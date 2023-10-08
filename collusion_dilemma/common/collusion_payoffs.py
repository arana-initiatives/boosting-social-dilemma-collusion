# import statements
import random
import numpy as np
from collusion_dilemma.common.constants import *
from copy import deepcopy

# for acessing multi-dimensional array elements
from functools import reduce
import operator


def _update_base_payoffs(payoff_values, append_value):
    return np.insert(payoff_values, obj=0, values=append_value, axis=-1)

def _add_payoff_noise(payoff_values, noise, const_noise=False):
    if const_noise:
        return payoff_values + round(random.uniform(-noise, noise),3)
    return payoff_values + np.round(np.random.uniform(low=-noise, high=noise, size=payoff_values.shape), 3) 

def _get_pbrs_factor(eps_len, num_agents, hetero_prob):
    assert hetero_prob < 1.
    max_expct_agent_rwd = 1
    if hetero_prob > 0:
        max_expct_agent_rwd = 1*(1-hetero_prob) + 2*hetero_prob

    return max_expct_agent_rwd / np.sqrt(eps_len * num_agents)

def _add_temperature_tuning(payoff_values, temperature_val):
    # apply this function to payoff matrix before adding the noise
    zero_masks, non_zero_masks = payoff_values < payoff_values.max() / 2, payoff_values >= payoff_values.max() / 2
    payoff_values[zero_masks] = payoff_values[zero_masks] - temperature_val
    payoff_values[non_zero_masks] = payoff_values[non_zero_masks] + temperature_val
    return payoff_values

def _generate_sparse_payoffs(payoff_values):
    sparse_masks = np.ones(payoff_values.shape, dtype=bool)
    base_idx = [x - 1 for x in list(payoff_values.shape)[:len(list(payoff_values.shape))-1]]
    idx_vals = []
    for i in range(payoff_values.shape[-1]):
        temp_base_idx = deepcopy(base_idx)
        temp_base_idx.append(i)
        idx_vals.append(tuple(temp_base_idx))
    
    for idx_val in idx_vals:
        sparse_masks[idx_val] = False
    payoff_values[sparse_masks] = 0
    return payoff_values
    


class CollusionPayoffs():
    def __init__(self,
                 num_agents=2,
                 hetero_prob=0.,
                 eps_len=1,
                 gov_rek=False,
                 zero_mean=True,
                 sparse=False,
                 noise=0.1,
                 const_noise=False,
                 temperature=0.1,
                 ):
        self.num_agents = num_agents
        self.hetero_prob = hetero_prob
        self.eps_len = eps_len
        self.gov_rek = gov_rek
        self.zero_mean = zero_mean
        self.noise = noise
        self.const_noise = const_noise
        self.temperature = temperature
        self.sparse = sparse
        self.pbrs_factor = _get_pbrs_factor(self.eps_len, self.num_agents, self.hetero_prob)
        self.agent_payoffs = self.generate_payoff_vector()        

    def generate_payoff_vector(self):
        col_pay_vector_strat_a = np.copy(BASE_COLLUSION_PAYOFFS)
        col_pay_vector_strat_b = np.copy(BASE_COLLUSION_PAYOFFS)

        if self.sparse:
            col_pay_vector_strat_a = np.copy(SPARSE_BASE_COLLUSION_PAYOFFS)
            col_pay_vector_strat_b = np.copy(SPARSE_BASE_COLLUSION_PAYOFFS)

        if self.gov_rek:
            col_pay_vector_strat_a = np.copy(BASE_COLLUSION_PAYOFFS) + self.pbrs_factor * np.copy(NON_ZERO_MEAN_KERNEL)
            col_pay_vector_strat_b = np.copy(BASE_COLLUSION_PAYOFFS) + self.pbrs_factor * np.copy(NON_ZERO_MEAN_KERNEL)
            if self.sparse:
                col_pay_vector_strat_a = np.copy(SPARSE_BASE_COLLUSION_PAYOFFS) + self.pbrs_factor * np.copy(NON_ZERO_MEAN_KERNEL)
                col_pay_vector_strat_b = np.copy(SPARSE_BASE_COLLUSION_PAYOFFS) + self.pbrs_factor * np.copy(NON_ZERO_MEAN_KERNEL)
            if self.zero_mean:
                col_pay_vector_strat_a = np.copy(BASE_COLLUSION_PAYOFFS) + self.pbrs_factor * np.copy(ZERO_MEAN_KERNEL)
                col_pay_vector_strat_b = np.copy(BASE_COLLUSION_PAYOFFS) + self.pbrs_factor * np.copy(ZERO_MEAN_KERNEL)
                if self.sparse:
                    col_pay_vector_strat_a = np.copy(SPARSE_BASE_COLLUSION_PAYOFFS) + self.pbrs_factor * np.copy(ZERO_MEAN_KERNEL)
                    col_pay_vector_strat_b = np.copy(SPARSE_BASE_COLLUSION_PAYOFFS) + self.pbrs_factor * np.copy(ZERO_MEAN_KERNEL)

        if self.hetero_prob > 0: # default payoff matrix starts with one heterogeneous agent
            col_pay_vector_strat_a = np.copy(HETERO_COLLUSION_PAYOFFS)
            col_pay_vector_strat_b = np.copy(HETERO_COLLUSION_PAYOFFS)
            if self.sparse:
                col_pay_vector_strat_a = np.copy(SPARSE_HETERO_COLLUSION_PAYOFFS)
                col_pay_vector_strat_b = np.copy(SPARSE_HETERO_COLLUSION_PAYOFFS)

            if self.gov_rek:
                col_pay_vector_strat_a = np.copy(HETERO_COLLUSION_PAYOFFS) + self.pbrs_factor * np.copy(NON_ZERO_MEAN_KERNEL)
                col_pay_vector_strat_b = np.copy(HETERO_COLLUSION_PAYOFFS) + self.pbrs_factor * np.copy(NON_ZERO_MEAN_KERNEL)
                if self.sparse:
                    col_pay_vector_strat_a = np.copy(SPARSE_HETERO_COLLUSION_PAYOFFS) + self.pbrs_factor * np.copy(NON_ZERO_MEAN_KERNEL)
                    col_pay_vector_strat_b = np.copy(SPARSE_HETERO_COLLUSION_PAYOFFS) + self.pbrs_factor * np.copy(NON_ZERO_MEAN_KERNEL)
                if self.zero_mean:
                    col_pay_vector_strat_a = np.copy(HETERO_COLLUSION_PAYOFFS) + self.pbrs_factor * np.copy(ZERO_MEAN_KERNEL)
                    col_pay_vector_strat_b = np.copy(HETERO_COLLUSION_PAYOFFS) + self.pbrs_factor * np.copy(ZERO_MEAN_KERNEL)
                if self.sparse:
                    col_pay_vector_strat_a = np.copy(SPARSE_HETERO_COLLUSION_PAYOFFS) + self.pbrs_factor * np.copy(ZERO_MEAN_KERNEL)
                    col_pay_vector_strat_b = np.copy(SPARSE_HETERO_COLLUSION_PAYOFFS) + self.pbrs_factor * np.copy(ZERO_MEAN_KERNEL)

        for kth_agent in range(self.num_agents-2):
            hetero_prob_ = np.random.rand()
            if hetero_prob_ <= self.hetero_prob:
                base_agent_payoffs = HETERO_AGENT_PAYOFFS
            else:
                base_agent_payoffs = BASE_AGENT_PAYOFFS

            if self.gov_rek:
                base_agent_payoffs = {'no_collusion': base_agent_payoffs['no_collusion'],
                                      'collusion': base_agent_payoffs['collusion'] + self.pbrs_factor * 1}
                if self.zero_mean:
                    base_agent_payoffs = {'no_collusion': base_agent_payoffs['no_collusion'] + self.pbrs_factor * -1 ,
                                          'collusion': base_agent_payoffs['collusion'] + self.pbrs_factor * 1}

            col_pay_vector_strat_a = _update_base_payoffs(col_pay_vector_strat_a[np.newaxis, ...], base_agent_payoffs['no_collusion'])
            col_pay_vector_strat_b = _update_base_payoffs(col_pay_vector_strat_b[np.newaxis, ...], base_agent_payoffs['collusion'])
            
            col_pay_vector_strat_a = np.append(col_pay_vector_strat_a, col_pay_vector_strat_b, axis=0)
            col_pay_vector_strat_b = np.copy(col_pay_vector_strat_a)

        if self.temperature > 0:
            col_pay_vector_strat_a = _add_temperature_tuning(col_pay_vector_strat_a, self.temperature)
        
        if self.noise > 0:
            col_pay_vector_strat_a = _add_payoff_noise(col_pay_vector_strat_a, self.noise, self.const_noise)
        
        if self.sparse:
            col_pay_vector_strat_a = _generate_sparse_payoffs(col_pay_vector_strat_a)
        return np.round(col_pay_vector_strat_a, 3)

if __name__ == "__main__":
    # homogeneous collusion payoffs
    collusion_payoffs = CollusionPayoffs(num_agents=4, gov_rek=True, zero_mean=True, sparse=True)
    print(collusion_payoffs.agent_payoffs, collusion_payoffs.agent_payoffs.shape, '\n', \
          collusion_payoffs.agent_payoffs[1][1][1][1], collusion_payoffs.agent_payoffs[1][1][1][1].shape, '\n')
    
    # heterogeneous collusion payoffs
    collusion_payoffs = CollusionPayoffs(num_agents=4, hetero_prob=0.5, sparse=True)
    print(collusion_payoffs.agent_payoffs, collusion_payoffs.agent_payoffs.shape, '\n',  \
          collusion_payoffs.agent_payoffs[1][1][1][1], collusion_payoffs.agent_payoffs[1][1][1][1].shape, '\n')

    # testing reward retrieval from `agent_payoffs` 
    actions_arr = np.random.randint(0, 2, size=collusion_payoffs.num_agents)
    print(reduce(operator.getitem, actions_arr, collusion_payoffs.agent_payoffs), actions_arr)
