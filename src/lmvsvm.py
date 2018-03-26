import numpy as np

import liblinearutil as liblin
from scipy.linalg import lstsq

from src.utils import select_from_multiple_views, select_landmarks, multiview_kernels

def get_kernels(x, lands, kernel=None):

    if kernel is None:
        x = select_from_multiple_views(x, lands)
    else:
        x = multiview_kernels(x, lands, kernel)

    return x

def recontruct_views(proj_sample, proj_landmarks):

    nb_views = proj_landmarks.shape[2]

    L = np.hstack([proj_landmarks[:, :, v] for v in range(nb_views)])
    M = np.hstack([proj_sample[:, :, v] for v in range(nb_views)])

    R = missing_lstsq(L, M)

    return np.dot(R, L)

def missing_lstsq(A, B):
    """ Code adapted from http://alexhwilliams.info/itsneuronalblog/2018/02/26/censored-lstsq/
    Solves least squares problem subject to missing data.

    Note: uses a broadcasted solve for speed.

    Args
    ----
    A (ndarray) : l x lv matrix
    B (ndarray) : m x lv matrix

    Returns
    -------
    X (ndarray) : m x l matrix that minimizes norm(M*(B - XA))
    """
    mask = ~np.isnan(B) 

    X = np.empty((B.shape[0], A.shape[0]))
    for i in range(B.shape[0]):
        m = mask[i] # drop rows where mask is zero
        X[i] = lstsq(A[:, m].T, B[i, m], check_finite=False)[0]
    return X

def train(x, y, c, params='-s 2 -B 1 -q'):
    return liblin.train(y.tolist(), x.tolist(), '-c {} '.format(c) + params)

def predict(x, y, model, classify=True):

    p_label, p_acc, p_vals = liblin.predict(y.tolist(), x.tolist(), model, "-q")

    if classify:
        return p_label

    return p_vals

