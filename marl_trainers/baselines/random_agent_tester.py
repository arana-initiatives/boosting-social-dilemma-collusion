from tianshou.env import PettingZooEnv
from collusion_dilemma.envs.collusion_dilemma_envs import CollusionDilemmaEnv
from tianshou.data import Collector
from tianshou.env import DummyVectorEnv
from tianshou.policy import RandomPolicy, MultiAgentPolicyManager



def random_agent_tester(num_agents=2, render=.1):
    env = PettingZooEnv(CollusionDilemmaEnv(num_agents=num_agents))
    agent_policies = [RandomPolicy() for idx in range(num_agents)]
    policy = MultiAgentPolicyManager(agent_policies, env)
    env = DummyVectorEnv([lambda: env])
    collector = Collector(policy, env)
    result = collector.collect(n_episode=1, render=render)


if __name__ == "__main__":
    random_agent_tester(num_agents=3, render=.1)
