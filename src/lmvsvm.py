import numpy as np

import liblinearutil as liblin
from scipy.linalg import lstsq
from scipy.optimize import leastsq, minimize

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

    R, _ = missing_lstsq(L, M)

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
        m = mask[i]
        X[i] = lstsq(A[:, m].T, B[i, m], check_finite=False)[0]
    return X, mask

def train(x, y, c, params='-s 2 -B 1 -q'):
    return liblin.train(y.tolist(), x.tolist(), '-c {} '.format(c) + params)

def predict(x, y, model, classify=True):

    p_label, p_acc, p_vals = liblin.predict(y.tolist(), x.tolist(), model, "-q")

    if classify:
        return p_label

    return p_vals

# -------------------------------------------------------------- ALTERNATING LEARNING R, theta

# def r_obj_function(R, i, x, y, lands, svm, mask, c1, c2): 
#     l = len(lands)

#     m_i = mask[i]
#     cost = c2*sum((x[i, m_i] - np.dot(R, lands[:, m_i]).T)**2)
#     _, _, svm_pred = liblin.predict(y[i:i], [np.dot(R, lands).tolist()], svm, "-q")
#     cost += c1*max(0, 1 - np.asarray(svm_pred)[0, y[i]])

#     return cost

def r_obj_function(R, x, y, lands, svm, mask, c1, c2): 
    m = len(x)
    l = len(lands)

    r_2d = R.reshape(m, l)
    _, _, svm_pred = liblin.predict(y, np.dot(r_2d, lands).tolist(), svm, "-q")

    cost = c2*sum((x - np.dot(r_2d, lands))[mask]**2)
    cost += c1*sum(np.max(0, 1 - np.asarray(svm_pred)[0, y]))
    
    # for i in range(len(x)):
    #     m_i = mask[i]
    #     cost += c2*sum((x[i, m_i] - np.dot(r_2d[i], lands[:, m_i]).T)**2)
    #     cost += c1*max(0, 1 - np.asarray(svm_pred)[0, y[i]])

    return cost

def mean_class_losses(svm, x, y, mask, r_costs):
    pass


def alternating_train(x, y, lands, c1, c2=1, params='-s 2 -B 1 -q'):

    nb_views = lands.shape[2]

    L = np.hstack([lands[:, :, v] for v in range(nb_views)])
    M = np.hstack([x[:, :, v] for v in range(nb_views)])

    l = len(lands)
    m = len(x)

    r0, mask = missing_lstsq(L, M)
    s0 = np.dot(r0, L)

    sample = s0.copy()
    R = r0.copy()

    y_list = y.tolist()
    svm = liblin.train(y.tolist(), sample.tolist(), '-c {} '.format(c1) + params)

    it = 0
    while True:
        it += 1

        res = minimize(r_obj_function, R.flatten(), args=(M, y, L, svm, mask, c1, c2), options={'disp':  True})
        # for i in range(len(x)):
        #     r_i = minimize(r_obj_function, R[i, :], args=(i, M, y, L, svm, mask, c1, c2), options={'disp':  False})
        #     R[i] = r_i.x
        #     cost += r_i.fun

        print(r_i.fun)
        R = res.x.reshape(m, l)
        sample =  np.dot(R, L)

        svm = liblin.train(y.tolist(), sample.tolist(), '-c {} '.format(c1) + params)

        _, p_acc, _ = liblin.predict(y, sample.tolist(), svm, "-q")
        print(p_acc)
        if it == 1:
            break
    return svm 

def alternating_predict(x, y, proj_landmarks, model, classify=True):

    sample = recontruct_views(x, proj_landmarks)

    p_label, p_acc, p_vals = liblin.predict(y.tolist(), sample.tolist(), model, "-q")

    if classify:
        return p_label

    return p_vals