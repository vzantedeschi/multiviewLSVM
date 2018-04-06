import numpy as np

from numpy.linalg import inv
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

def laplacian_reconstruction(x, kernel=None, x2=None):

    x_copy = x.copy()

    if x2 is not None:
        x2_copy = x2.copy()

    # select biggest view to be the principal view (only on train)
    nan_per_view = np.sum(np.isnan(x_copy), axis=(0, 1))
    v = np.argmin(nan_per_view)

    # drop points whose principal view is NaN
    inds = ~np.isnan(x_copy[:, :, v])[:, 0]
    x_copy = x_copy[inds]   

    if x2 is not None:
        inds2 = ~np.isnan(x2_copy[:, :, v])[:, 0]
        x2_copy = x2_copy[inds2]

        x_copy = np.vstack((x_copy, x2_copy))

    # compute principal view gram
    if kernel is not None:
        ref_gram = kernel(x_copy[:, :, v], x_copy[:, :, v])
    else:
        x_copy = x_copy[:, inds]
        ref_gram = x_copy[:, :, v]

    ref_laplacian = np.diag(np.sum(ref_gram, axis=1)) - ref_gram
    grams = []

    for view in range(len(nan_per_view)):
        if view == v:
            grams.append(ref_gram)
        else:  
            m_inds = np.isnan(x_copy[:, :, view])[:, 0]
            c_inds = ~m_inds

            if kernel is not None:
                K_cc = kernel(x_copy[c_inds, :, view], x_copy[c_inds, :, view])
            else:
                K_cc = x_copy[c_inds][:, c_inds, view]

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

    if x2 is None:
        return gram_views, inds

    else:
        return gram_views, inds, inds2

