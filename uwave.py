import numpy as np
import time
from statistics import mean, stdev

from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import KFold

from liblinearutil import *

from src.baseline import *
from src.utils import dict_to_csv, load_uwave, multiview_kernels

DATASET = "uwave"
kname = "rbf"

c_range = [10**i for i in range(-3, 4)]

CV = 3
PATH = "results/view/{}/svm-per-view".format(DATASET)

print("learning on {} with SVM per view".format(DATASET))

# datasets
X, Y, test_X, test_Y = load_uwave()

# tuning
t1 = time.time()

tuning_acc = {}.fromkeys(c_range, 0.)    

splitter = KFold(n_splits=CV)

for train_index, val_index in splitter.split(X):

    train_x, val_x = X[train_index], X[val_index]
    train_y, val_y = Y[train_index], Y[val_index]

    train_matrix = multiview_kernels(train_x, rbf_kernel, 3)
    val_matrix = multiview_kernels(val_x, rbf_kernel, 3)

    # tuning     
    for c in c_range:
        models = train_svm_per_view(train_matrix, train_y, 8, 3, c)
        pred = predict_svm_per_view(val_matrix, val_y, 3, models)
        tuning_acc[c] += accuracy_score(val_y, pred)

best_C = max(tuning_acc, key=tuning_acc.get)

t2 = time.time()
print("tuning time:", t2-t1)

# training
train_matrix = multiview_kernels(X, rbf_kernel, 3)
models = train_svm_per_view(train_matrix, Y, 8, 3, best_C)

t3 = time.time()
print("training time:", t3-t2)

test_matrix = multiview_kernels(test_X, rbf_kernel, 3)
pred = predict_svm_per_view(test_matrix, test_Y, 3, models)

t4 = time.time()
print("testing time:", t4-t3)

dict_to_csv({'accuracy': accuracy_score(test_Y, pred), 'error': 0., 'train_time': t3-t2, 'test_time': t4-t3},["nb_iter={},cv={}".format(1, CV)], PATH+".csv")
