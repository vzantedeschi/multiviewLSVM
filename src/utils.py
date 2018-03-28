#/usr/bin/python

# ---------------------------------------------------------------------- IMPORTS
import argparse
import csv
import numpy as np
import random
import os

from numpy import linalg as LA
from liblinearutil import *
from scipy.sparse import csr_matrix
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import normalize, scale


def normalized_mse(y, y_pred):

    return mean_squared_error(y, y_pred) / LA.norm(y)

# -------------------------------------------------------------- I/0 FUNCTIONS

def make_directory(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def dict_to_csv(my_dict,header,filename):

    make_directory(os.path.dirname(filename))

    with open(filename, 'w') as csv_file:
        writer = csv.writer(csv_file)
        
        writer.writerow(header)
        for key, value in my_dict.items():
            writer.writerow([key, value])

def dict_to_array(l):
    data,indices,indptr = [],[],[0]

    ptr = 0

    for d in l:

        for k,v in d.items():
            
            data.append(v)
            indices.append(k-1)

        ptr += len(d.keys())
        indptr.append(ptr)

    return csr_matrix((data, indices, indptr))

def array_to_dict(a,**kwargs):

    results = []
    r,c = a.shape

    try:
        a = a.tolil()
    except:
        pass

    if kwargs:
        clusters = kwargs['clusters']
        L = kwargs['land']

        for i in range(r):
            k = clusters[i]
            results.append({})
            for j in range(c):
                results[i][k*L+j+1] = float(a[i,j])
            # results[i][k*L+c+1] = 1
    else:

        for i in range(r):
            results.append({})
            for j in range(c):
                results[i][j+1] = float(a[i,j])
            # results[i][c+1] = 1

    return results

# ----------------------------------------------------------------------
def select_landmarks(x, n):
    m = len(x)

    return x[random.sample(range(m), min(m, n))]

def select_from_multiple_views(x, inds):
    r = x[:, inds, :]
    return r

def get_view_dict(x):
    d = {}

    for v in range(x.shape[2]):
        d[v] = x[:, :, v]
        
    return d

def twod_array(x):

    return np.hstack([x[:, :, v] for v in range(x.shape[2])])


def multiview_kernels(x1, x2, kernel, gamma=None):

    matrices = []
    nb_views = x1.shape[2]

    for v in range(nb_views):
        c1_inds = ~np.isnan(x1[:, :, v])[:, 0]
        c2_inds = ~np.isnan(x2[:, :, v])[:, 0]

        view_m = np.full((len(x1), len(x2)), np.nan)
        view_m[np.ix_(c1_inds, c2_inds)] = kernel(x1[c1_inds, :, v], x2[c2_inds, :, v], gamma=gamma)
        matrices.append(view_m)

    return np.dstack(matrices)

# ----------------------------------------------------------------- SPLITTERS

def splits_generator(x, CV=3, sets=None):

    if sets is not None:
        for p in range(3):

            train_inds = sets[p][0]
            test_inds = sets[p][1]
            val_inds = sets[p][2]

            yield train_inds, val_inds, test_inds

    else:
        splitter = KFold(n_splits=CV)
        for train_inds, val_inds in splitter.split(x):

            yield train_inds, val_inds, None


# ----------------------------------------------------------------- DATASET LOADERS
from scipy.io import loadmat
import csv

DATAPATH = "./datasets/"

def load_flower17(process=None):

    dist_mat1 = loadmat(os.path.join(DATAPATH, "flower17", "distancematrices17gcfeat06"))
    dist_mat2 = loadmat(os.path.join(DATAPATH, "flower17", "distancematrices17itfeat08"))

    dist_matrices = dict(dist_mat1, **dist_mat2)

    matrix, mean_fts = [], []
    for k, val in dist_matrices.items():
        if not k.startswith("__"):

            if process:
                val = process(val)

            matrix.append(val[:, :, None])

    matrix = np.dstack(matrix)

    assert matrix.shape == (1360, 1360, 7), matrix.shape

    splits = loadmat(os.path.join(DATAPATH, "flower17", "datasplits"))

    sets = []
    for i in range(1, 4):
        s = (splits["trn{}".format(i)]-1, splits["tst{}".format(i)]-1, splits["val{}".format(i)]-1)
        sets.append(list(map(np.squeeze, s)))

    indices = list(range(1360))
    labels = np.asarray([i // 80 for i in indices])

    return labels, sets, matrix

def load_sarcos(id_task=1):

    train_mat = loadmat(os.path.join(DATAPATH, "sarcos", "sarcos_inv"))
    test_mat = loadmat(os.path.join(DATAPATH, "sarcos", "sarcos_inv_test"))

    train_array, test_array = train_mat['sarcos_inv'], test_mat['sarcos_inv_test']

    assert train_array.shape == (44484, 28), train_array.shape
    assert test_array.shape == (4449, 28), test_array.shape

    train_x, train_y = np.dstack([train_array[:, i*7:(i+1)*7][:, :, None] for i in range(3)]), train_array[:, 20+id_task]
    test_x, test_y = np.dstack([test_array[:, i*7:(i+1)*7][:, :, None] for i in range(3)]), test_array[:, 20+id_task]

    return train_x, np.squeeze(train_y), test_x, np.squeeze(test_y)

def load_uwave():

    with open(os.path.join(DATAPATH, "uwave", "UWaveGestureLibraryAll_TRAIN"), 'r') as f_train:

        reader = csv.reader(f_train, quoting=csv.QUOTE_NONNUMERIC)
        train = np.asarray(list(reader))
        train_y, train_x = train[:, 0].astype(int) - 1, train[:, 1:]
        train_x = np.dstack([train_x[:, 315*i:315*(i+1)] for i in range(3)])

    with open(os.path.join(DATAPATH, "uwave", "UWaveGestureLibraryAll_TEST"), 'r') as f_test:

        reader = csv.reader(f_test, quoting=csv.QUOTE_NONNUMERIC)
        test = np.asarray(list(reader))
        test_y, test_x = test[:, 0].astype(int) - 1, test[:, 1:]
        test_x = np.dstack([test_x[:, 315*i:315*(i+1)] for i in range(3)])


    assert train_x.shape == (896, 315, 3), train_x.shape
    assert test_x.shape == (3582, 315, 3), test_x.shape

    return train_x, train_y, test_x, test_y

def csv_to_dict(filename):

    with open(filename, 'r') as csv_file:
        reader = csv.reader(csv_file)
        next(reader,None)
        my_dict = {row[0]:eval(row[1]) for row in reader}

    return my_dict

# ------------------------------------------------------------------- ARG PARSER

def get_args(prog, dataset="flower17", strategy="lmvsvm", view_rec_type="zeros"):

    parser = argparse.ArgumentParser(prog=prog, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-s", "--strategy", dest='strategy', default=strategy,
                        help='choice of learner')
    parser.add_argument("-r", "--reconstr", dest='view_rec_type', default=view_rec_type,
                        help='choice of reconstruction technique for missing views')

    return parser.parse_args()