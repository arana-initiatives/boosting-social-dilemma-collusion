import numpy as np

# partially sparse reward signals in collusion dilemma problem
BASE_COLLUSION_PAYOFFS = np.array([[[0, 0], [0, 1]], [[1, 0], [1, 1]]], dtype=np.float32)
HETERO_COLLUSION_PAYOFFS = np.array([[[0, 0], [0, 2]], [[1, 0], [1, 2]]], dtype=np.float32)
BASE_AGENT_PAYOFFS = {'no_collusion': 0., 'collusion': 1.} # for heterogeneous setup, other agents are base agents
HETERO_AGENT_PAYOFFS = {'no_collusion': 0., 'collusion': 2.} # for heterogeneous setup, only one agent dominates
