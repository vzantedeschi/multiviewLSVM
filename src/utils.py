#/usr/bin/python

# ---------------------------------------------------------------------- IMPORTS

import csv
import datetime
import logging
import numpy as np
import os

def make_directory(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

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

# -------------------------------------------------------------- I/0 FUNCTIONS

def dict_to_csv(my_dict,filename):

    make_directory(os.path.dirname(filename))

    with open(filename, 'w') as csv_file:
        writer = csv.writer(csv_file)
        
        for key, value in my_dict.items():
            writer.writerow([key, value])
            print(key,value)

def csv_to_dict(filename):

    with open(filename, 'rb') as csv_file:
        reader = csv.reader(csv_file)
        mydict = dict(reader)