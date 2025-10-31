# Distributed Gradient Descent
# Core Formula for unctronstrainted DGD 
# x_i^(k+1) = sum j to m(w_ij*x_i^k - alpha(k)grad(f_i(x_i_^k)))
#
# Ref: https://doi.org/10.1109/TAC.2008.2009515 
#

import numpy as np
from jax import grad

def iteration_step(w, x, f_grad, m, i, k):
    return np.sum([w[i][j]*x[j] - 1/(k+1) * f_grad(x[i]) for j in range(m)], axis=0)

def grad_f(f):
    return grad(f)

def generate_linear_desc_array(n):
    return np.arange(1, 0, step=-1/n)

def generate_random_doubly_stoch_mat(dim):
    x = np.random.random(dim)
    row_sum = None
    col_sum = None

    iteration = 0
    while ((np.any(row_sum != 1)) | (np.any(col_sum != 1))) and iteration <= 500:
        x /= x.sum(0)
        x = x / x.sum(1)[:, np.newaxis]
        row_sum = x.sum(1)
        col_sum = x.sum(0)
        iteration += 1

    return x


