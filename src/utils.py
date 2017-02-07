#/usr/bin/python

# ---------------------------------------------------------------------- IMPORTS
import argparse
import random

from liblinearutil import *
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize

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

    if kwargs:
        clusters = clustering(a,kwargs['clusterer'])
        L = kwargs['land']

        for i in range(r):
            k = clusters[i]
            results.append({})
            for j in range(c):
                e = a[i,j]
                if e != 0:
                    results[i][k*L+j+1] = float(e)
            # results[i][k*L+r+1] = 1
    else:

        for i in range(r):

            results.append({})
            for j in range(c):
                e = a[i,j]
                if e != 0:
                    results[i][j+1] = float(e)
            # results[i][r+1] = 1
    return results

# ----------------------------------------------------------------------
def select_landmarks(x,n,norm=False):
    m = len(x)
    landmarks = dict_to_array(random.sample(x,min(m,n)))

    if norm:
        normalize(landmarks)

    return landmarks.transpose()

def project(x,landmarks,clusterer=None,norm=False):
    
    x_arr = dict_to_array(x)

    if norm:
        normalize(x_arr)

    # project on landmark space
    projection = x_arr.dot(landmarks)

    if clusterer:
        return array_to_dict(projection,clusterer=clusterer,land=landmarks.shape[0])
    else:
        return array_to_dict(projection)

def clustering(x,clusterer):
    try:
        return clusterer.predict(x)
    except:
        clusterer.fit(x)
        return clusterer.labels_

# ----------------------------------------------------------------- DATASET LOADERS
def load_dataset(name):
    if name == "svmguide1":
        train_y, train_x = svm_read_problem('./datasets/svmguide1')
        test_y, test_x = svm_read_problem('./datasets/svmguide1.t')
    return train_y,train_x,test_y,test_x

# ------------------------------------------------------------------- ARG PARSER

def get_args(prog,dataset_name="svmguide1",nb_clusters=1):

    parser = argparse.ArgumentParser(prog=prog,formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-d", "--dataset", dest='dataset_name', default=dataset_name,
                        help='dataset name')
    parser.add_argument("-n", "--nbclusters", type=int, dest='nb_clusters', default=nb_clusters,
                        help='number of clusters')
    parser.add_argument("-s", "--savemodel", dest='save_model', action="store_true",
                        help='if set, the learned model is saved')

    return parser.parse_args()