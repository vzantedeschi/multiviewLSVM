#/usr/bin/python

# ---------------------------------------------------------------------- IMPORTS

import datetime
import logging
import numpy as np

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

def make_directory(dir_path):
    import os
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

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
    Y = Y.reshape(1,m)

    Y[Y==classes[0]] = -1
    Y[Y==classes[1]] = 1
    return X,Y

# -------------------------------------------------------------------- DATA MANAGER

def accuracy_rate(y_pred,y):
    return np.sum(y==y_pred)/y.shape[1]