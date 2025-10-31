
import synapse.da.dgd as dgd

import numpy as np


def test_dgd():

    iteration_num = 200
    num_agents = 6
    w = dgd.generate_random_doubly_stoch_mat((num_agents, num_agents))
    f = lambda x: x * x
    x = np.ones(num_agents)

    for i in range(iteration_num):
        iter_x = x.copy()
        for a in range(num_agents):
            
            x[a] = dgd.iteration_step(
                w,
                iter_x,
                dgd.grad_f(f),
                num_agents,
                a,
                i
            )
    
    assert all([v >= -0.0001 and v <= 0.0001 for v in x])


def test_dgd_vector_x():

    iteration_num = 100
    num_agents = 6
    w = dgd.generate_random_doubly_stoch_mat((num_agents, num_agents))
    x = np.array([np.ones(num_agents) for i in range(num_agents)])

    for i in range(iteration_num):
        iter_x = x.copy()
        for a in range(num_agents):

            if a % 2 == 0:
                f_i = lambda x: x[a] * x[a] - x[0]
            else:
                f_i = lambda x: -x[a] + x[5]

            x[a] = dgd.iteration_step(
                w,
                iter_x,
                dgd.grad_f(f_i),
                num_agents,
                a,
                i
            )

    assert True

