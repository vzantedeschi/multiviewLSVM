import numpy as np

from numpy.random import choice
from scipy.linalg import lstsq


def set_random_views_to_value(x, r, nb_fts, nb_views, value=float('NaN')):

    m = len(x)

    x_copy = x.copy()
    zero_ids = choice(range(m*nb_views), int(r*m*nb_views))

    if type(value) == float:

        for i in zero_ids:
            id_point = i // nb_views
            id_view = i % nb_views
            x_copy[id_point][id_view*nb_fts:(id_view+1)*nb_fts] = value
    else:

        for i in zero_ids:
            id_point = i // nb_views
            id_view = i % nb_views
            x_copy[id_point][id_view*nb_fts:(id_view+1)*nb_fts] = value[id_view]

    return x_copy

def set_random_blocks_to_value(dict_x, r, nb_fts, value=float('NaN')):

    m = len(x[0])

    zero_ids = choice(range(m*nb_views), int(r*m*nb_views))

    if type(value) == float:

        for i in zero_ids:
            id_point = i // nb_views
            id_view = i % nb_views
            x[id_view][id_point] = value

    else:

        for i in zero_ids:
            id_point = i // nb_views
            id_view = i % nb_views
            x[id_view][id_point] = value[id_view]

    return x


def recontruct_views(M):

    R, res, k, _ = lstsq(L.T, M.T, check_finite=False)

    return np.dot(R.T, L), sum(res), k