import numpy as np

from numpy.random import choice
from scipy.linalg import lstsq


def set_random_views_to_zero(x, r, nb_fts, nb_views):

    m = len(x)

    x_copy = x.copy()
    zero_ids = choice(range(m*nb_views), int(r*m*nb_views))

    for i in zero_ids:
        id_point = i // nb_views
        id_view = i % nb_views
        x_copy[id_point][id_view*nb_fts:(id_view+1)*nb_fts] = 0

    return x_copy


def recontruct_views(M, L):

    R, res, k, _ = lstsq(L.T, M.T, check_finite=False)

    return np.dot(R.T, L), sum(res), k

