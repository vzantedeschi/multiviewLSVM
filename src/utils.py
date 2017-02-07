#/usr/bin/python

# ---------------------------------------------------------------------- IMPORTS
import argparse
import numpy as np
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
    else:

        for i in range(r):

            results.append({})
            for j in range(c):
                e = a[i,j]
                if e != 0:
                    results[i][j+1] = float(e)

    return results

# ----------------------------------------------------------------------
def select_landmarks(x,n):
    m = x.shape[0]
    landmarks = x[random.sample(range(m),min(m,n))]

    return landmarks.transpose()

def project(x,landmarks,clusterer=None):

    # project on landmark space
    projection = x.dot(landmarks)

    if clusterer:
        return array_to_dict(projection,clusterer=clusterer,land=landmarks.shape[1])
    else:
        return array_to_dict(projection)

def clustering(x,clusterer):
    try:
        return clusterer.predict(x)
    except:
        clusterer.fit(x)
        return clusterer.labels_

# ----------------------------------------------------------------- DATASET LOADERS
DATAPATH = "./datasets/"

def load_csr_matrix(filename):
    with open(filename,'r') as in_file:
        data,indices,indptr = [],[],[0]

        labels = []
        ptr = 0

        for line in in_file:
            line = line.split(None, 1)
            if len(line) == 1: 
                line += ['']
            label, features = line
            labels.append(float(label))

            f_list = features.split()
            for f in f_list:

                k,v = f.split(':')
                data.append(float(v))
                indices.append(float(k)-1)

            ptr += len(f_list)
            indptr.append(ptr)

        return labels,csr_matrix((data, indices, indptr))

def load_dataset(name,norm=False):

    if name == "svmguide1":
        train_path = DATAPATH+name
        test_path = DATAPATH+name+'.t'

    elif name == "ijcnn1":
        train_path = DATAPATH+name+'.tr'
        test_path = DATAPATH+name+'.t'

    elif name == "mnist":
        train_path = DATAPATH+name+'_train.csv.sparse'
        test_path = DATAPATH+name+'_test.csv.sparse'

    train_y,train_x = load_csr_matrix(train_path)
    
    test_y,test_x = load_csr_matrix(test_path)

    if norm:
        return train_y,normalize(train_x),test_y,normalize(test_x)
    else:
        return train_y,train_x,test_y,test_x

# ------------------------------------------------------------------- ARG PARSER

def get_args(prog,dataset_name="svmguide1",nb_clusters=1,nb_landmarks=10):

    parser = argparse.ArgumentParser(prog=prog,formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-d", "--dataset", dest='dataset_name', default=dataset_name,
                        help='dataset name')
    parser.add_argument("-n", "--nbclusters", type=int, dest='nb_clusters', default=nb_clusters,
                        help='number of clusters')
    parser.add_argument("-l", "--nblands", type=int, dest='nb_landmarks', default=nb_landmarks,
                        help='number of landmarks')
    parser.add_argument("-s", "--savemodel", dest='save_model', action="store_true",
                        help='if set, the learned model is saved')

    return parser.parse_args()