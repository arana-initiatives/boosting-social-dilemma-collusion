# general import statements
import re
import numpy as np
from collusion_dilemma.common.collusion_payoffs import CollusionPayoffs
# for acessing multi-dimensional array elements
import functools
from functools import reduce
import operator

# environment design related import statements
from gymnasium.spaces import Box, Discrete
from pettingzoo.utils.env import ParallelEnv
from pettingzoo.test import parallel_api_test

def _get_agent_names(num_agents):
    return [ f"agent_{agent_number}" for agent_number in range(num_agents) ]

class CollusionDilemmaEnv(ParallelEnv):
    metadata = {
        "name": "collusion_dilemma_v0",
    }

    def __init__(self, num_agents=2, hetero_prob=0., eps_len=5, coop_flag=True):
        self.num_actions = 2 # fixed binary action choices: "to collude, or not to"
        self._num_agents = num_agents
        self._hetero_prob = hetero_prob
        self.eps_len = eps_len
        self.coop_flag = coop_flag
        self.possible_agents = _get_agent_names(self._num_agents)
        self.observation_spaces = Box(low=0.0, high=self.num_actions - 1, shape=(self._num_agents, ), dtype=np.float32)
        self.payoff_vector = CollusionPayoffs(self._num_agents, self._hetero_prob).agent_payoffs
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )
        # keeps track of current observed state & episode interactions
        self.current_state = None
        self.current_step = 0


    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents[:]
        self.current_timestep = 0
        self.state = np.random.randint(0, self.num_actions, size=self._num_agents) # represents all the taken actions
        observations = {agent: self.state[self.agent_name_mapping[agent]] for agent in self.agents}
        infos = {agent: {} for agent in self.agents}
        return observations, infos

    def step(self, actions):
        
        if not actions:
            self.agents = []
            return {}, {}, {}, {}, {}
        
        # extracting action values from the dictionary and populate states
        self.state = np.zeros(self._num_agents, dtype=np.int16)
        for agent_name, agent_action in actions.items():
            self.state[int(re.findall(r'\d+', agent_name)[0])] = agent_action

        rewards = { id: reduce(operator.getitem, self.state, self.payoff_vector)[index] \
                    for id, index in self.agent_name_mapping.items() } 

        observations = {
            self.agents[i]: self.state[i]
            for i in range(len(self.agents))
        }

        # custom environment boilerplate code for design compatibility
        terminations = {agent: False for agent in self.agents}
        self.current_timestep += 1
        env_truncation = self.current_timestep >= self.eps_len
        truncations = {agent: env_truncation for agent in self.agents}
        infos = {agent: {} for agent in self.agents}
        if env_truncation:
            self.agents = []

        return observations, rewards, terminations, truncations, infos

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return Discrete(self.num_actions)

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return Discrete(self.num_actions)

if __name__ == "__main__":
    env = CollusionDilemmaEnv()
    parallel_api_test(env, num_cycles=1_000_000)
