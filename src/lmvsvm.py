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

    R, res, k, _ = lstsq(L.T, M.T, check_finite=False)

    return np.dot(R.T, L), sum(res), k

def train(x, y, c, params='-s 2 -B 1 -q'):
    return liblin.train(y.tolist(), x.tolist(), '-c {} '.format(c) + params)

def predict(x, y, model, classify=True):

    p_label, p_acc, p_vals = liblin.predict(y.tolist(), x.tolist(), model, "-q")

    if classify:
        return p_label

    return p_vals

