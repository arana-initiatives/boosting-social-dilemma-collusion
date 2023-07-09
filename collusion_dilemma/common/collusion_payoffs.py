# import statements
import numpy as np
from collusion_dilemma.common.constants import *

# for acessing multi-dimensional array elements
from functools import reduce
import operator


def _update_base_payoffs(payoff_values, append_value):
    return np.insert(payoff_values, obj=0, values=append_value, axis=-1)

class CollusionPayoffs():
    def __init__(self, num_agents=2, hetero_prob=0.):
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
