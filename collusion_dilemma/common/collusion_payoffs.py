# import statements
import random
import numpy as np
from collusion_dilemma.common.constants import *

# for acessing multi-dimensional array elements
from functools import reduce
import operator


def _update_base_payoffs(payoff_values, append_value):
    return np.insert(payoff_values, obj=0, values=append_value, axis=-1)

def _add_payoff_noise(payoff_values, noise, const_noise=False):
    if const_noise:
        return payoff_values + round(random.uniform(-noise, noise),3)
    return payoff_values + np.round(np.random.uniform(low=-noise, high=noise, size=payoff_values.shape), 3) 

def _get_pbrs_factor(payoff_values, eps_len, num_agents):
    max_reward_arr = np.ones(num_agents, dtype=np.int8)
    return sum(reduce(operator.getitem, max_reward_arr, payoff_values)) / eps_len

def _add_temperature_tuning(payoff_values, temperature_val):
    # apply this function to payoff matrix before adding the noise
    zero_masks, non_zero_masks = payoff_values == 0., payoff_values != 0.
    payoff_values[zero_masks] = temperature_val
    payoff_values[non_zero_masks] = payoff_values[non_zero_masks] - temperature_val
    return payoff_values

class CollusionPayoffs():
    def __init__(self, num_agents=2, hetero_prob=0., eps_len=1, gov_rek=False):
        self.num_agents = num_agents
        self.hetero_prob = hetero_prob
        self.agent_payoffs = self.generate_payoff_vector(self.num_agents, self.hetero_prob)

    def generate_payoff_vector(self, num_agents, hetero_prof_ref):
        col_pay_vector_strat_a = np.copy(BASE_COLLUSION_PAYOFFS)
        col_pay_vector_strat_b = np.copy(BASE_COLLUSION_PAYOFFS)
        if hetero_prof_ref > 0: # default payoff matrix starts with one heterogeneous agent
            col_pay_vector_strat_a = np.copy(HETERO_COLLUSION_PAYOFFS)
            col_pay_vector_strat_b = np.copy(HETERO_COLLUSION_PAYOFFS)

        for kth_agent in range(num_agents-2):
            hetero_prob_ = np.random.rand()
            if hetero_prob_ <= hetero_prof_ref:
                base_agent_payoffs = HETERO_AGENT_PAYOFFS
            else:
                base_agent_payoffs = BASE_AGENT_PAYOFFS    

            col_pay_vector_strat_a = _update_base_payoffs(col_pay_vector_strat_a[np.newaxis, ...], base_agent_payoffs['no_collusion'])
            col_pay_vector_strat_b = _update_base_payoffs(col_pay_vector_strat_b[np.newaxis, ...], base_agent_payoffs['collusion'])
            col_pay_vector_strat_a = np.append(col_pay_vector_strat_a, col_pay_vector_strat_b, axis=0)
            col_pay_vector_strat_b = np.copy(col_pay_vector_strat_a)
        
        return col_pay_vector_strat_a

if __name__ == "__main__":
    # homogeneous collusion payoffs
    collusion_payoffs = CollusionPayoffs(num_agents=4)
    print(collusion_payoffs.agent_payoffs, collusion_payoffs.agent_payoffs.shape, '\n', \
          collusion_payoffs.agent_payoffs[1][1][1][1], collusion_payoffs.agent_payoffs[1][1][1][1].shape, '\n')
    
    # heterogeneous collusion payoffs
    collusion_payoffs = CollusionPayoffs(num_agents=4, hetero_prob=0.5)
    print(collusion_payoffs.agent_payoffs, collusion_payoffs.agent_payoffs.shape, '\n',  \
          collusion_payoffs.agent_payoffs[1][1][1][1], collusion_payoffs.agent_payoffs[1][1][1][1].shape, '\n')

    # testing reward retrieval from `agent_payoffs` 
    actions_arr = np.random.randint(0, 2, size=collusion_payoffs.num_agents)
    print(reduce(operator.getitem, actions_arr, collusion_payoffs.agent_payoffs), actions_arr)
