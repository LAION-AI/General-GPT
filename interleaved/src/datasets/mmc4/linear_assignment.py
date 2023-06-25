'''
From: https://github.com/allenai/mmc4/blob/main/scripts/linear_assignment.py

code for computing linear assignments using lapjv
'''

import numpy as np
import unittest
from numpy import array, dstack, float32, float64, linspace, meshgrid, random, sqrt
from scipy.spatial.distance import cdist
from lapjv import lapjv


def base_solve(W, max_dummy_cost_value=1000):
    '''
    Gives hungarian solve for a non-square matrix. it's roughly from:

    NOTE: this ** MINIMIZES COST **. So, if you're handing sims, make sure to negate them!

    https://github.com/jmhessel/multi-retrieval/blob/master/bipartite_utils.py

    returns i_s, j_s, cost such that:
    for i, j in zip(i_s, j_s)

    are the (i, j) row column entries selected.

    cost is sum( cost[i, j] for i, j in zip(i_s, j_s) )

    '''
    if np.sum(np.abs(W)) > max_dummy_cost_value:
        print('Warning, you values in your matrix may be too big, please raise max_dummy_cost_value')


    orig_shape = W.shape
    if orig_shape[0] != orig_shape[1]:
        if orig_shape[0] > orig_shape[1]:
            pad_idxs = [[0, 0], [0, W.shape[0]-W.shape[1]]]
            col_pad = True
        else:
            pad_idxs = [[0, W.shape[1]-W.shape[0]], [0, 0]]
            col_pad = False
        W = np.pad(W, pad_idxs, 'constant', constant_values=max_dummy_cost_value)

    sol, _, cost = lapjv(W)

    i_s = np.arange(len(sol))
    j_s = sol[i_s]

    sort_idxs = np.argsort(-W[i_s, j_s])
    i_s, j_s = map(lambda x: x[sort_idxs], [i_s, j_s])

    if orig_shape[0] != orig_shape[1]:
        if col_pad:
            valid_idxs = np.where(j_s < orig_shape[1])[0]
        else:
            valid_idxs = np.where(i_s < orig_shape[0])[0]
        i_s, j_s = i_s[valid_idxs], j_s[valid_idxs]

    m_cost = 0.0
    for i, j in zip(i_s, j_s):
        m_cost += W[i, j]

    return i_s, j_s, m_cost