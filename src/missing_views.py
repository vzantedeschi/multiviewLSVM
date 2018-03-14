import numpy as np

from numpy.random import choice

def set_random_views_to_value(x, r, value=float('NaN'), sym=False):

    m = len(x)
    nb_views = x.shape[2]
    
    x_copy = x.copy()
    zero_ids = choice(range(m*nb_views), int(r*m*nb_views))

    if type(value) == float:

        for i in zero_ids:
            id_point = i // nb_views
            id_view = i % nb_views
            x_copy[id_point, :, id_view] = value

            if sym:
                x_copy[:, id_point, id_view] = value

    else:

        for i in zero_ids:
            id_point = i // nb_views
            id_view = i % nb_views
            x_copy[id_point, :, id_view] = value[id_view]

            if sym:
                x_copy[:, id_point, id_view] = value[id_view]

    return x_copy