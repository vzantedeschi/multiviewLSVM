#/usr/bin/python

# ---------------------------------------------------------------------- IMPORTS
import argparse
import csv
import numpy as np
import random
import os

from liblinearutil import *
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize,scale

from src.kernels import kernels

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
    return np.hstack([x[inds + v*nb_insts][:, lands] for v in range(nb_views)])

# ----------------------------------------------------------------- DATASET LOADERS
DATAPATH = "./datasets/"

def load_flower17(process=None):

    from scipy.io import loadmat

    dist_mat1 = loadmat(os.path.join(DATAPATH, "flower17", "distancematrices17gcfeat06"))
    dist_mat2 = loadmat(os.path.join(DATAPATH, "flower17", "distancematrices17itfeat08"))

    dist_matrices = dict(dist_mat1, **dist_mat2)

    matrix = []
    for k, val in dist_matrices.items():
        if not k.startswith("__"):

            if process:
                val = process(val)

            matrix.append(val)
    matrix = np.vstack(matrix)
    assert matrix.shape == (1360 * 7, 1360), matrix.shape

    splits = loadmat(os.path.join(DATAPATH, "flower17", "datasplits"))

    sets = []
    for i in range(1, 4):
        s = (splits["trn{}".format(i)]-1, splits["tst{}".format(i)]-1, splits["val{}".format(i)]-1)
        sets.append(list(map(np.squeeze, s)))

    indices = list(range(1360))
    labels = np.asarray([i // 80 for i in indices])

    return indices, labels, sets, matrix

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