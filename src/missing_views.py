import numpy as np

from numpy.linalg import inv
from numpy.random import choice

def set_random_views_to_value(x, y, r, r_type="none", sym=False):

    m = len(x)
    nb_views = x.shape[2]

    x_copy = x.copy()
    y_copy = y.copy()

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

    # drop points whose features are all nans
    inds = ~np.isnan(x_copy).all(axis=(1,2))
    x_copy = x_copy[inds]
    y_copy = y_copy[inds]

    return x_copy, y_copy

def laplacian_reconstruction(x, y, kernel, x2=None, y2=None):

    x_copy = x.copy()
    y_copy = y.copy()

    if x2 is not None:
        x2_copy = x2.copy()
        y2_copy = y2.copy()

    # select biggest view to be the principal view (only on train)
    nan_per_view = np.sum(np.isnan(x_copy), axis=(0, 1))
    v = np.argmin(nan_per_view)

    # drop points whose principal view is NaN
    inds = ~np.isnan(x_copy[:, :, v])[:, 0]
    x_copy = x_copy[inds]
    y_copy = y_copy[inds]

    if x2 is not None:
        inds = ~np.isnan(x2_copy[:, :, v])[:, 0]
        x2_copy = x2_copy[inds]
        y2_copy = y2_copy[inds]

        x_copy = np.vstack((x_copy, x2_copy))

    # compute principal view gram
    ref_gram = kernel(x_copy[:, :, v], x_copy[:, :, v])
    ref_laplacian = np.diag(np.sum(ref_gram, axis=1)) - ref_gram
    grams = []

    for view in range(len(nan_per_view)):
        if view == v:
            grams.append(ref_gram)
        else:
            m_inds = np.isnan(x_copy[:, :, view])[:, 0]
            c_inds = ~m_inds

            K_cc = kernel(x_copy[c_inds, :, view], x_copy[c_inds, :, view])
            L_cm = ref_laplacian[c_inds][:, m_inds]
            L_mm = ref_laplacian[m_inds][:, m_inds]
            L_mm_inv = inv(L_mm)

            view_gram = np.full(ref_gram.shape, np.nan)

            view_gram[np.ix_(c_inds, c_inds)] = K_cc
            view_gram[np.ix_(c_inds, m_inds)] = -np.dot(K_cc, np.dot(L_cm, L_mm_inv))
            view_gram[np.ix_(m_inds, c_inds)] = -np.dot(L_mm_inv, np.dot(L_cm.T, K_cc))
            view_gram[np.ix_(m_inds, m_inds)] = np.dot(L_mm_inv, np.dot(L_cm.T, np.dot(K_cc, np.dot(L_cm, L_mm_inv))))

            # check symmetry
            assert np.allclose(view_gram, view_gram.T, atol=1e-8)

            grams.append(view_gram)

    gram_views = np.dstack(grams)
    assert np.all(gram_views), np.isnan(gram_views)

    if x2 is None:
        return gram_views, y_copy

    else:
        return gram_views, y_copy, y2_copy

