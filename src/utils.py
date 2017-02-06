#/usr/bin/python

# ---------------------------------------------------------------------- IMPORTS
import argparse
import csv
import datetime
import logging
import numpy as np
import os

from scipy.spatial.distance import pdist
from sklearn.preprocessing import scale


PATH = "./datasets/"
GAMMA_PATH = PATH+"gammas.csv"
DATASETS = ["iris01","iris02","iris12","sonar","ionosphere","heart-statlog","liver"]

def compute_gamma(sample):
    dists = pdist(sample)
    assert len(dists) == len(sample)*(len(sample)-1)/2
    mean_dist = np.average(dists)
    return mean_dist

def most_frequent_list_dict(d_list):
    result = {}
    for d in d_list:
        for k,v in d.items():
            try:
                result[k].append(v)
            except:
                result[k] = [v]

    for k,v in result.items():
        result[k] = max(set(v),key=v.count)

    return result

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

def load_iris_dataset(excluded_class=None):
    from sklearn.datasets import load_iris
    # load dataset
    iris = load_iris()

    if excluded_class is None:

        return iris.data,iris.target

    else:
        classes = [0,1,2]
        classes.remove(excluded_class)

        # select instances of classes 0 and 1
        Y = iris.target[iris.target!=excluded_class]
        X = iris.data[iris.target!=excluded_class]

        m = X.shape[0]
        assert m == Y.shape[0]

        Y[Y==classes[0]] = -1
        Y[Y==classes[1]] = 1
        return X,Y

def load_mnist_dataset():
    # from keras.datasets import mnist

    import os

    proxy = 'http://cache.univ-st-etienne.fr:3128'

    os.environ['http_proxy'] = proxy 
    os.environ['HTTP_PROXY'] = proxy
    os.environ['https_proxy'] = proxy
    os.environ['HTTPS_PROXY'] = proxy

    # return mnist.load_data()

    from sklearn.datasets import fetch_mldata
    mnist = fetch_mldata('MNIST original', data_home="datasets/")
    return mnist.data,mnist.target

def load_dataset(dataset_name,get_params=False):
    param_dict = {'c':[10**i for i in range(-3,10)]}
    gammas = load_gammas()

    if dataset_name == "iris01":
        x,y = load_iris_dataset(excluded_class=2)
    elif dataset_name == "iris02":
        x,y = load_iris_dataset(excluded_class=1)
    elif dataset_name == "iris12":
        x,y = load_iris_dataset(excluded_class=0)
    elif dataset_name == "mnist":
        x,y = load_mnist_dataset()
    else:
        dataset = np.loadtxt(PATH+dataset_name+".txt")
        if dataset_name == "sonar":
            x,y = np.split(dataset,[-1],axis=1)
        elif dataset_name == "ionosphere":
            x,y = np.split(dataset,[-1],axis=1)
        elif dataset_name == "heart-statlog":
            y,x = np.split(dataset,[1],axis=1)
        elif dataset_name == "liver":
            x,y = np.split(dataset,[-1],axis=1)
            y[y==2] = -1
        else:
            raise Exception("Unknown dataset: please implement a loader.")

    if get_params:
        return scale(x),y,gammas[dataset_name],param_dict
    else:
        return scale(x),y

def load_train_test(dataset_name):
    if dataset_name == "svmguide1":
        conv = {i: (lambda s: float(s.decode().split(':')[1])) for i in range(1,5)}
        train_name = dataset_name
        test_name = dataset_name+".t"
    elif dataset_name == "ijcnn1":
        train_name = dataset_name+".tr"
        test_name = dataset_name+".t"
        # conv = {lambda s: {int(s.decode().split(':')[0]):float(s.decode().split(':')[1])}}
        conv = {i: (lambda s: float(s.decode().split(':')[1]) or 0) for i in range(1,14)}

        train = np.loadtxt(PATH+train_name,converters=conv)
        test = np.loadtxt(PATH+test_name,converters=conv)

        train_y,train_x = np.split(train,[1],axis=1)
        test_y,test_x = np.split(test,[1],axis=1)

        train_y[train_y==0] = -1
        test_y[test_y==0] = -1

    return scale(train_x),scale(test_x),train_y,test_y

# ------------------------------------------------------------------- ARG PARSER

def get_args(prog,dataset_name,nb_clusters,distance):

    parser = argparse.ArgumentParser(prog=prog,formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-d", "--dataset", dest='dataset_name', default=dataset_name,
                        help='dataset name')
    parser.add_argument("-n", "--nbclusters", type=int, dest='nb_clusters', default=nb_clusters,
                        help='number of clusters')
    parser.add_argument("-s", "--savemodel", dest='save_model', action="store_true",
                        help='if set, the learned model is saved')
    parser.add_argument("-m", "--distance", type=str, dest='distance', default=distance,
                        choices=['euclidean','mst'], help='distance measure between centroids and landmarks')

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

    dict_to_csv(gammas_dict,["gammas for standardized samples"],GAMMA_PATH)

def load_gammas():
    return csv_to_dict(GAMMA_PATH)

# ------------------------------------------------------------------ MAIN
if __name__ == "__main__":
    gammas = {}

    for d in DATASETS:
        sample,_ = load_dataset(d,get_params=False)
        gammas[d] = compute_gamma(sample)

    save_gammas(gammas)