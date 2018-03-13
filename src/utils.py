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

def select_from_multiple_views(x, inds, lands, nb_views, nb_insts):
    r = np.hstack([x[inds][:, lands, v] for v in range(nb_views)])
    assert r.shape == (len(inds), len(lands)*nb_views)
    return r

def get_view_blocks(x, inds_1, inds_2, nb_views):
    d = {}

    for v in range(nb_views):
        d[v] = x[inds_1][:, inds_2, v]
        
    return d

def multiview_kernels(x1, x2, kernel, nb_views, gamma=None):

    matrices = {}
    feats_per_view = x1.shape[1] // nb_views

    for v in range(nb_views):
        view_m = kernel(x1[:, v*feats_per_view: (1+v)*feats_per_view], x2[:, v*feats_per_view: (1+v)*feats_per_view], gamma=gamma)

        matrices[v] = view_m

    return matrices

def get_landmarks_projections(distances, land_ids, nb_views, nb_insts):
    return np.hstack([distances[land_ids + v*nb_insts][:, land_ids] for v in range(nb_views)])

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
            mean_fts.append(np.mean(val, axis=0))

    matrix = np.dstack(matrix)
    mean_fts = np.vstack(mean_fts)

    assert mean_fts.shape == (7, 1360), mean_fts.shape
    assert matrix.shape == (1360, 1360, 7), matrix.shape

    splits = loadmat(os.path.join(DATAPATH, "flower17", "datasplits"))

    sets = []
    for i in range(1, 4):
        s = (splits["trn{}".format(i)]-1, splits["tst{}".format(i)]-1, splits["val{}".format(i)]-1)
        sets.append(list(map(np.squeeze, s)))

    indices = list(range(1360))
    labels = np.asarray([i // 80 for i in indices])

    return indices, labels, sets, matrix, mean_fts

def load_sarcos(id_task=1):

    train_mat = loadmat(os.path.join(DATAPATH, "sarcos", "sarcos_inv"))
    test_mat = loadmat(os.path.join(DATAPATH, "sarcos", "sarcos_inv_test"))

    train_array, test_array = train_mat['sarcos_inv'], test_mat['sarcos_inv_test']

    assert train_array.shape == (44484, 28), train_array.shape
    assert test_array.shape == (4449, 28), test_array.shape

    train_x, train_y = train_array[:, :21], train_array[:, 20+id_task]
    test_x, test_y = test_array[:, :21], test_array[:, 20+id_task]

    return train_x, np.squeeze(train_y), test_x, np.squeeze(test_y)

def load_uwave():

    with open(os.path.join(DATAPATH, "uwave", "UWaveGestureLibraryAll_TRAIN"), 'r') as f_train:

        reader = csv.reader(f_train, quoting=csv.QUOTE_NONNUMERIC)
        train = np.asarray(list(reader))
        train_y, train_x = train[:, 0] - 1, train[:, 1:]

    with open(os.path.join(DATAPATH, "uwave", "UWaveGestureLibraryAll_TEST"), 'r') as f_test:

        reader = csv.reader(f_test, quoting=csv.QUOTE_NONNUMERIC)
        test = np.asarray(list(reader))
        test_y, test_x = test[:, 0] - 1, test[:, 1:]

    assert train_x.shape == (896, 945), train_x.shape
    assert test_x.shape == (3582, 945), test_x.shape

    return train_x, train_y, test_x, test_y

def csv_to_dict(filename):

    with open(filename, 'r') as csv_file:
        reader = csv.reader(csv_file)
        next(reader,None)
        my_dict = {row[0]:eval(row[1]) for row in reader}

    return my_dict

# ------------------------------------------------------------------- ARG PARSER

def get_args(prog,dataset_name="svmguide1",nb_clusters=1,nb_landmarks=10,kernel="linear",pca=False):

    parser = argparse.ArgumentParser(prog=prog,formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-d", "--dataset", dest='dataset_name', default=dataset_name,
                        help='dataset name')
    parser.add_argument("-l", "--nblands", type=int, dest='nb_landmarks', default=nb_landmarks,
                        help='number of landmarks')
    parser.add_argument("-o", "--normalize", dest='norm', action="store_true",
                        help='if set, the dataset is normalized')
    parser.add_argument("-k", "--kernel", dest='kernel', default=kernel, choices=kernels.keys(),
                        help='choice of projection function')

    return parser.parse_args()