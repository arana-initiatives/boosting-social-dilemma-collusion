# config related imports
from omegaconf import OmegaConf
import marl_trainers.configs.config_paths as cfg_pth
# tianshou related import statements
import gymnasium as gym
from gymnasium.spaces import Box, Discrete
import torch, numpy as np, torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import tianshou as ts
from tianshou.env import PettingZooEnv
from collusion_dilemma.envs.collusion_dilemma_envs import CollusionDilemmaEnv
from tianshou.utils.net.common import Net

def config_loader(config_path):
    return OmegaConf.load(config_path)

def get_env(config):
    return PettingZooEnv(CollusionDilemmaEnv(num_agents=config.num_agents,
                                            hetero_prob=config.hetero_prob,
                                            eps_len=config.eps_len,))

def dqn_trainer(config):
    logger = ts.utils.TensorboardLogger(SummaryWriter(config.logging_dir))
    train_envs = ts.env.DummyVectorEnv([lambda: get_env(config) for _ in range(config.train_num)])
    test_envs = ts.env.DummyVectorEnv([lambda: get_env(config) for _ in range(config.test_num)])

    env = get_env(config)
    state_shape = env.observation_space.shape or env.observation_space.n
    action_shape = env.action_space.shape or env.action_space.n
    net = Net(state_shape=state_shape, action_shape=action_shape, hidden_sizes=config.hidden_sizes)
    optim = torch.optim.Adam(net.parameters(), lr=config.learning_rate)

    policy = ts.policy.DQNPolicy(net, optim, config.gamma,
                                 config.n_step,
                                 target_update_freq=config.target_freq)
    train_collector = ts.data.Collector(policy,
                                        train_envs,
                                        ts.data.VectorReplayBuffer(config.buffer_size,
                                                                   config.train_num),
                                        exploration_noise=True)
    test_collector = ts.data.Collector(policy, test_envs, exploration_noise=True)

    result = ts.trainer.offpolicy_trainer(
    policy, train_collector, test_collector,
    config.epochs,
    config.step_per_epoch,
    config.step_per_collect,
    config.test_num,
    config.batch_size,
    update_per_step=1 / config.step_per_collect,
    train_fn=lambda epoch, env_step: policy.set_eps(config.eps_train),
    test_fn=lambda epoch, env_step: policy.set_eps(config.eps_test),
    stop_fn=lambda mean_rewards: mean_rewards >= env.spec.reward_threshold,
    logger=logger)
    print(f'Finished training! Use {result["duration"]}')

if __name__ == "__main__":
    # change below specified path variable for executing different files
    config = config_loader(cfg_pth.DQN_BASELINE_HMG_TRAINER_CONFIG)
    dqn_trainer(config)
