"""A simple multi-agent env baseline with N-agent social dilemma problem.

This script demonstrates running the following policies in competition for the above problem:
    (1) heuristic policy of repeating the same move
    (2) heuristic policy of beating the last opponent move
    (3) LSTM/feedforward PPO policies
    (4) LSTM policy with custom entropy loss
"""

# config related imports
from omegaconf import OmegaConf
import marl_trainers.configs.config_paths as cfg_pth
# argument parsing related import statements
import argparse
import os
# social dilemma collusion related import statement
# Note: The `PettingZooEnv` wrapper needs import from rllib library only
# from tianshou.env import PettingZooEnv
from collusion_dilemma.envs.collusion_dilemma_envs import CollusionDilemmaEnv
# rllib related import statements
import random
import ray
from ray import air, tune
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.rllib.algorithms.ppo import (
    PPO,
    PPOConfig,
    PPOTF2Policy,
    PPOTF1Policy,
    PPOTorchPolicy,
)
from ray.rllib.env import PettingZooEnv
from ray.rllib.examples.policy.rock_paper_scissors_dummies import (
    BeatLastHeuristic,
    AlwaysSameHeuristic,
)
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.tune.registry import register_env

# import tensorflow and pytorch modules
tf1, tf, tfv = try_import_tf()
torch, _ = try_import_torch()

def config_loader(config_path):
    return OmegaConf.load(config_path)

# load the specific single configuration file for carrying out the experiment
# TODO: uncomment the configuration to select amongst homogenous or heterogenous agents
# TODO: change the gov_rek flag in the configuration file for loading the governance layer

# configs = config_loader(cfg_pth.PPO_BASELINE_HMG_TRAINER_CONFIG) # Option 1
# configs = config_loader(cfg_pth.PPO_BASELINE_HTR_TRAINER_CONFIG) # Option 2
# configs = config_loader(cfg_pth.PPO_LARGE_HMG_TRAINER_CONFIG) # Option 3
configs = config_loader(cfg_pth.PPO_LARGE_HTR_TRAINER_CONFIG) # Option 4


def env_creator(config):
    env = CollusionDilemmaEnv(num_agents=config.num_agents,
                              gov_rek=config.gov_rek,
                              hetero_prob=config.hetero_prob,
                              eps_len=config.eps_len,
                              zero_mean=config.zero_mean,)
    return env


register_env(configs.env_name, lambda config: PettingZooEnv(env_creator(configs)))


def run_same_policy(args, stop):
    """Use the same PPO policy for all the agents."""
    config = (
        PPOConfig()
        .environment(args.env_name)
        .framework(args.framework)
        )

    # results = tune.Tuner(
    #     "PPO", param_space=config, run_config=air.RunConfig(stop=stop, verbose=1)
    # ).fit()

    # if args.as_test:
    #     check_learning_achieved(results, args.stop_reward)

    results = ray.tune.run('PPO', name='PPO_LARGE_HTR_ZERO_MEAN_GOV_TRAINER', config=config, stop={ 'timesteps_total': 12_000 }, verbose=1)
    


def run_heuristic_vs_learned(args, use_lstm=True, algorithm="PPO"):
    """Run heuristic policies vs a learned agent.
    """

    def select_policy(agent_id, episode, **kwargs):
        if agent_id == "player_0":
            return "learned"
        else:
            return random.choice(args.policy_choices)

    config = (
        AlgorithmConfig(algo_class=algorithm)
        .environment(args.env_name)
        .framework(args.framework)
        .rollouts(
            num_rollout_workers=0,
            num_envs_per_worker=4,
            rollout_fragment_length=64,
        )
        .training(
            train_batch_size=args.train_batch_size,
            gamma=args.gamma,
        )
        .multi_agent(
            policies={
                "always_same": PolicySpec(policy_class=AlwaysSameHeuristic),
                "beat_last": PolicySpec(policy_class=BeatLastHeuristic),
                "learned": PolicySpec(
                    config=AlgorithmConfig.overrides(
                        model={"use_lstm": use_lstm},
                        framework_str=args.framework,
                    )
                ),
            },
            policy_mapping_fn=select_policy,
            policies_to_train=["learned"],
        )
        .reporting(metrics_num_episodes_for_smoothing=args.metric_smoothing_eps_len)
        # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
        .resources(num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0")))
    )

    algo = config.build()

    for itr_count in range(args.stop_iters):
        results = algo.train()
        
        # Timesteps reached.
        if "policy_always_same_reward" not in results["hist_stats"]:
            reward_diff = 0
            continue
        
        reward_diff = sum(results["hist_stats"]["episode_reward"])
        avg_total_reward = sum( [a/b for a,b in zip(results["hist_stats"]["episode_reward"], results["hist_stats"]["episode_lengths"])] )
        dom_agent_reward = sum( [a/b for a,b in zip(results["hist_stats"]["policy_beat_last_reward"], results["hist_stats"]["episode_lengths"])] )

        print("Iteration Count: " + str(itr_count))
        print("Average accumulated rewards: " + str(avg_total_reward))
        print("Dominant agent average rewards: " + str(dom_agent_reward))
        
        if results["timesteps_total"] > args.stop_timesteps:
            break
        # Reward (difference) reached -> all good, return.
        elif reward_diff > args.stop_reward:
            return

    # Reward (difference) not reached: Error if `as_test`.
    if args.as_test:
        raise ValueError(
            "Desired reward difference ({}) not reached! Only got to {}.".format(
                args.stop_reward, reward_diff
            )
        )


def run_with_custom_entropy_loss(config, stop):
    """Example of customizing the loss function of an existing policy.
    This performs about the same as the default loss does."""

    policy_cls = {
        "torch": PPOTorchPolicy,
        "tf": PPOTF1Policy,
        "tf2": PPOTF2Policy,
    }[config.framework]

    class EntropyPolicy(policy_cls):
        def loss_fn(policy, model, dist_class, train_batch):
            logits, _ = model(train_batch)
            action_dist = dist_class(logits, model)
            if config.framework == "torch":
                # Required by PGTorchPolicy's stats fn.
                model.tower_stats["policy_loss"] = torch.tensor([0.0])
                policy.policy_loss = torch.mean(
                    -0.1 * action_dist.entropy()
                    - (
                        action_dist.logp(train_batch["actions"])
                        * train_batch["advantages"]
                    )
                )
            else:
                policy.policy_loss = -0.1 * action_dist.entropy() - tf.reduce_mean(
                    action_dist.logp(train_batch["actions"]) * train_batch["advantages"]
                )
            return policy.policy_loss

    class EntropyLossPPO(PPO):
        @classmethod
        def get_default_policy_class(cls, config):
            return EntropyPolicy

    run_heuristic_vs_learned(config, use_lstm=True, algorithm=EntropyLossPPO)


if __name__ == "__main__":
    # parsing the configuration dictionary and initializing rllib executor
    ray.init()

    stop = {
        "training_iteration": configs.stop_iters,
        "timesteps_total": configs.stop_timesteps,
        "episode_reward_mean": configs.stop_reward,
    }

    run_same_policy(configs, stop=stop)
    print("run_same_policy: PPO")

    # run_heuristic_vs_learned(configs, use_lstm=True)
    # print("run_heuristic_vs_learned (w/ lstm): ok.")

    # run_with_custom_entropy_loss(configs, stop=stop)
    # print("run_with_custom_entropy_loss: ok.")
    ray.shutdown()
