# general import statements
import re
import gymnasium
import numpy as np
from collusion_dilemma.common.collusion_payoffs import CollusionPayoffs
# for acessing multi-dimensional array elements
import functools
from functools import reduce
import operator

# environment design related import statements
from gymnasium.spaces import Box, Discrete
from pettingzoo.utils.env import ParallelEnv, AECEnv
from pettingzoo.test import parallel_api_test
from pettingzoo.utils import agent_selector

def _get_agent_names(num_agents):
    return [ f"agent_{agent_number}" for agent_number in range(num_agents) ]

class CollusionDilemmaParallelEnv(ParallelEnv):
    metadata = {
        "name": "collusion_dilemma_parallel_v0",
    }

    def __init__(self, num_agents=2, hetero_prob=0., eps_len=5, gov_rek=False):
        self.gov_rek = gov_rek
        self.num_actions = 2 # fixed binary action choices: "to collude, or not to"
        self._num_agents = num_agents
        self._hetero_prob = hetero_prob
        self.eps_len = eps_len
        self.possible_agents = _get_agent_names(self._num_agents)
        self.observation_spaces = Box(low=0.0, high=self.num_actions - 1, shape=(self._num_agents, ), dtype=np.float32)
        self.payoff_vector = CollusionPayoffs(self._num_agents, self._hetero_prob, self.eps_len, self.gov_rek).agent_payoffs
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )
        # keeps track of current observed state & episode interactions
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


class CollusionDilemmaEnv(AECEnv):
    metadata = {
        "name": "collusion_dilemma_v0",
    }

    def __init__(self, num_agents=2, hetero_prob=0., eps_len=10, gov_rek=False, zero_mean=False):
        self.zero_mean = zero_mean
        self.gov_rek = gov_rek
        self.num_actions = 2 # fixed binary action choices: "to collude, or not to"
        self._num_agents = num_agents
        self._hetero_prob = hetero_prob
        self.eps_len = eps_len
        self.possible_agents = _get_agent_names(self._num_agents)
        self._observation_spaces = { agent: Discrete(self.num_actions) for agent in self.possible_agents }
        self._action_spaces = { agent: Discrete(self.num_actions) for agent in self.possible_agents }
        self.payoff_vector = CollusionPayoffs(self._num_agents, self._hetero_prob, self.eps_len, self.gov_rek, self.zero_mean).agent_payoffs
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )
        self.agents = self.possible_agents[:]
        self.state = np.random.randint(0, self.num_actions-1, size=self._num_agents) # represents all the taken actions
        self.observations = {agent: self.state[self.agent_name_mapping[agent]] for agent in self.agents}
        self.rewards = {agent: 0 for agent in self.agents}
        # keeps track of current observed state & episode interactions
        self.current_step = 0
        self.current_agent = None
        self.current_action = None


    def reset(self, seed=None, options=None):
        
        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0 for agent in self.agents}
        self.state = np.random.randint(0, self.num_actions-1, size=self._num_agents) # represents all the taken actions
        self.observations = {agent: self.state[self.agent_name_mapping[agent]] for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self._cumulative_rewards = {agent: 0. for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        # keeps track of current observed state & episode interactions
        self.current_timestep = 0
        self.current_agent = None
        self.current_action = None
        # agent selection metric values
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()

    def observe(self, agent):
        return np.array(self.observations[agent])

    def step(self, action):
        # TODO: Find exact error reason for action attaining out of bound values
        if action: # action value clipping for compatibility sanity
            if action > 1:
                action = 1
            elif action < 0:
                action = 0

        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            self._was_dead_step(action)
            return
        
        agent = self.agent_selection
        self.current_agent = agent
        self.current_action = action

        self._cumulative_rewards[agent] = 0
        self.state[self.agent_name_mapping[agent]] = action
        
        if self._agent_selector.is_last():
            # rewards for all agents are placed in the .rewards dictionary
            
            self.rewards = { id: reduce(operator.getitem, self.state, self.payoff_vector)[index] \
                    for id, index in self.agent_name_mapping.items() }

            self.current_step += 1
            # The truncations dictionary must be updated for all players.
            self.truncations = {
                agent: self.current_step >= self.eps_len for agent in self.agents
            }

            self.observations = {
                self.agents[i]: self.state[i]
                for i in range(len(self.agents))
            }
        else:
            # necessary so that observe() returns a reasonable observation at all times.
            self.state[self.agent_name_mapping[agent]] = action
            # no rewards are allocated until both players give an action
            self._clear_rewards()

        # selects the next agent.
        self.agent_selection = self._agent_selector.next()
        # Adds .rewards to ._cumulative_rewards
        self._accumulate_rewards()

        # custom environment boilerplate code for design compatibility
        terminations = {agent: False for agent in self.agents}
        env_truncation = self.current_timestep >= self.eps_len
        truncations = {agent: env_truncation for agent in self.agents}
        infos = {agent: {} for agent in self.agents}

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return self._observation_spaces[agent]

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return self._action_spaces[agent]
    
    def render(self):
        print(self.current_agent, self.current_action, self.state, self.observations, self.rewards, self._cumulative_rewards)


if __name__ == "__main__":
    env = CollusionDilemmaParallelEnv()
    parallel_api_test(env, num_cycles=10_000_000)
