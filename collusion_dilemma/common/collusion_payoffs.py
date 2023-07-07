# import statements
import numpy as np
from collusion_dilemma.common.constants import *


def _update_base_payoffs(payoff_values, append_value):
    return np.insert(payoff_values, obj=0, values=append_value, axis=-1)

class CollusionPayoffs():
    def __init__(self, num_agents=3, hetero_mode=False):
        self.num_agents = num_agents
        self.hetero_mode = hetero_mode
        self.agent_payoffs = self.generate_payoff_vector(self.num_agents, self.hetero_mode)

    def generate_payoff_vector(self, num_agents, hetero_mode):
        col_pay_vector_strat_a = np.copy(BASE_COLLUSION_PAYOFFS)
        col_pay_vector_strat_b = np.copy(BASE_COLLUSION_PAYOFFS)
        if hetero_mode:
            col_pay_vector_strat_a = np.copy(HETERO_AGENT_PAYOFFS)
            col_pay_vector_strat_b = np.copy(HETERO_AGENT_PAYOFFS)

        for kth_agent in range(num_agents-2):
            base_agent_payoffs = BASE_AGENT_PAYOFFS
            if hetero_mode:
                base_agent_payoffs = HETERO_AGENT_PAYOFFS

            col_pay_vector_strat_a = _update_base_payoffs(col_pay_vector_strat_a[np.newaxis, ...], base_agent_payoffs['no_collusion'])
            col_pay_vector_strat_b = _update_base_payoffs(col_pay_vector_strat_b[np.newaxis, ...], base_agent_payoffs['collusion'])
            col_pay_vector_strat_a = np.append(col_pay_vector_strat_a, col_pay_vector_strat_b, axis=0)
            col_pay_vector_strat_b = np.copy(col_pay_vector_strat_a)
        
        return col_pay_vector_strat_a

if __name__ == "__main__":
    collusion_payoffs = CollusionPayoffs(num_agents=4)
    print(collusion_payoffs.agent_payoffs, collusion_payoffs.agent_payoffs.shape, '\n', \
          collusion_payoffs.agent_payoffs[1][1][1][1], collusion_payoffs.agent_payoffs[1][1][1][1].shape)

