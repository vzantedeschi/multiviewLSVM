import numpy as np

from numpy.random import choice

def set_random_views_to_value(x, r, r_type="none", sym=False):

    m = len(x)
    nb_views = x.shape[2]

    x_copy = x.copy()
    zero_ids = choice(range(m*nb_views), int(r*m*nb_views))

    if r_type in ["means", "none", "reconstruction"]:
        value = float("NaN")
    else:
        value = 0.

    for i in zero_ids:
        id_point = i // nb_views
        id_view = i % nb_views
        x_copy[id_point, :, id_view] = value

        if sym:
            x_copy[:, id_point, id_view] = value

    if r_type == "means":
        means = np.nanmean(x_copy, axis=0)

        for v in range(nb_views):
            nans_inds = np.isnan(x_copy[:, :, v])[:, 0]
            x_copy[nans_inds, :, v] = means[:, v]

    return x_copy