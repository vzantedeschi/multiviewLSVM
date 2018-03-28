import numpy as np
from svmutil import *

from src.utils import multiview_kernels
# ----------------------------------------------------------------------- MULTIVIEW BASELINES

def get_kernels(x1, x2=None, inds=None, kernel=None):

    if kernel is None:
        x = select_from_multiple_views(x1, inds)
    else:
        x = multiview_kernels(x1, x2, kernel)

    return x

def one_vs_all_svm_train(train_x, train_y, c, params='-s 0 -t 4 -b 1 -q'):
    models = []

    nb_classes = max(train_y) + 1

    for cl in range(nb_classes):

        y = train_y.copy()
        y[train_y == cl] = 1
        y[train_y != cl] = -1

        # add serial number
        x = np.c_[np.arange(len(y))+1, train_x]

        model = svm_train(y.tolist(), x.tolist(), '-c {} '.format(c) + params)
        models.append(model)

    return models

def svr_train(train_x, train_y, c, params='-s 3 -t 4 -b 1 -q'):

    # add serial number
    x = np.c_[np.arange(len(train_y))+1, train_x]

    model = svm_train(train_y.tolist(), x.tolist(), '-c {} '.format(c) + params)

    return model

def one_vs_all_svm_predict(test_x, test_y, models):

    scores = []

    for cl, model in enumerate(models):

        y = test_y.copy()
        y[test_y == cl] = 1
        y[test_y != cl] = -1
        
        # add serial number
        x = np.c_[np.arange(len(y))+1, test_x]

        p_label, _, p_vals = svm_predict(y.tolist(), x.tolist(), model, "-q")
        scores.append(p_vals)

    scores = np.hstack(scores)
    labels = np.argmax(scores, axis=1)
    return labels, scores[:, labels]

def svr_predict(test_x, test_y, model):

    _, _, p_vals = svm_predict(test_y.tolist(), test_x.tolist(), model, "-q")

    return p_vals

def train(x, y, c, classify=True):
    models = []
    for _, x_m in x.items():

        if classify:
            view_models = one_vs_all_svm_train(x_m, y, c)

        else:
            view_models = svr_train(x_m, y, c)

        models.append(view_models)

    return models

def predict(x, y, models, classify=True):

    scores = []

    if classify:
        predictions = []

        for v, view_models in enumerate(models):

            p_label, p_vals = one_vs_all_svm_predict(x[v], y, view_models)
            predictions.append(p_label)
            scores.append(p_vals)

        predictions = np.vstack(predictions).astype(int)
        m = len(y)
        most_frequent_label = np.empty(m)

        for i in range(m):
            most_frequent_label[i] = np.argmax(np.bincount(predictions[:, i]))

        return most_frequent_label

    else:

        for v, view_model in enumerate(models):

            p_vals = svr_predict(x[v], y, view_model)
            scores.append(p_vals)

        return np.mean(np.hstack(scores), axis=1)