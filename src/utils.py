#/usr/bin/python

# ---------------------------------------------------------------------- IMPORTS
import argparse
import csv
import datetime
import logging
import numpy as np
import os

from scipy.spatial.distance import pdist
from sklearn.preprocessing import normalize

PATH = "./datasets/"
GAMMA_PATH = PATH+"gammas.csv"
DATASETS = ["iris01","iris02","iris12","sonar","ionosphere"]

def compute_gamma(sample):
    dists = pdist(sample)
    assert len(dists) == len(sample)*(len(sample)-1)/2
    mean_dist = np.average(dists)
    return mean_dist

# ---------------------------------------------------------------------- LOGGER

def get_logger(filename):
    log_format = '[%(asctime)s] %(levelname)s - %(message)s'
    log_formatter = logging.Formatter(log_format)

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    logger_path = filename + datetime.date.today().strftime("%Y-%m-%d")

    handler = logging.FileHandler(logger_path)
    handler.setFormatter(log_formatter)
    logger.addHandler(handler)

    return logger,logger_path

# ------------------------------------------------------------------- DATASETS

def load_iris_dataset(excluded_class=2):
    classes = [0,1,2]
    classes.remove(excluded_class)

    from sklearn.datasets import load_iris
    # load dataset
    iris = load_iris()
    # select instances of classes 0 and 1
    Y = iris.target[iris.target!=excluded_class]
    X = iris.data[iris.target!=excluded_class]

    m = X.shape[0]
    assert m == Y.shape[0]
    Y = Y.reshape(m,1)

    Y[Y==classes[0]] = -1
    Y[Y==classes[1]] = 1
    return X,Y

def load_dataset(dataset_name,get_params=True):
    param_dict = {'c':[10**i for i in range(-1,5)]}
    gammas = load_gammas()

    if dataset_name == "iris01":
        x,y = load_iris_dataset(excluded_class=2)
    elif dataset_name == "iris02":
        x,y = load_iris_dataset(excluded_class=1)
    elif dataset_name == "iris12":
        x,y = load_iris_dataset(excluded_class=0)
    else:
        dataset = np.loadtxt(PATH+dataset_name+".txt")
        if dataset_name == "sonar":
            x,y = np.split(dataset,[-1],axis=1)
            param_dict = {'c':[10**i for i in range(1,6)]}
        elif dataset_name == "ionosphere":
            x,y = np.split(dataset,[-1],axis=1)
            param_dict = {'c':[10**i for i in range(-1,4)]}
        else:
            raise Exception("Unknown dataset: please implement a loader.")

    if get_params:
        return normalize(x),y,gammas[dataset_name],param_dict
    else:
        return normalize(x),y

# ------------------------------------------------------------------- ARG PARSER

def get_args(prog,dataset_name):

    parser = argparse.ArgumentParser(prog=prog,formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-d", "--dataset", dest='dataset_name', default=dataset_name,
                        help='dataset directory')

    return parser.parse_args()

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

def csv_to_dict(filename):

    with open(filename, 'r') as csv_file:
        reader = csv.reader(csv_file)
        next(reader,None)
        my_dict = {row[0]:eval(row[1]) for row in reader}

    return my_dict

def ndarray_to_bytes(array):
    return array.dumps()

def string_to_ndarray(string):
    return np.ma.loads(eval(string))

def save_gammas(gammas_dict):
    import numbers

    for key,value in gammas_dict.items():
        assert key in DATASETS
        assert isinstance(value,numbers.Real)

    dict_to_csv(gammas_dict,["gammas for normalized samples"],GAMMA_PATH)

def load_gammas():
    return csv_to_dict(GAMMA_PATH)


# ------------------------------------------------------------------ MAIN
if __name__ == "__main__":
    gammas = {}

    for d in DATASETS:
        sample,_, = load_dataset(d,get_params=False)
        gammas[d] = compute_gamma(sample)

    save_gammas(gammas)