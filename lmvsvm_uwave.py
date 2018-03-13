import numpy as np
import time
from statistics import mean, stdev

from sklearn.model_selection import KFold
from liblinearutil import *

from src.kernels import get_kernel
from src.projection import multiview_project
from src.utils import dict_to_csv, load_uwave, select_landmarks

DATASET = "uwave"
kname = "rbf"

landmarks = [10, 50, 100, 200, 400, 500, 600, 684]
c_range = [10**i for i in range(-3, 4)]

ITER = 5
CV = 3
PATH = "results/view/{}/lmvsvm".format(DATASET)

print("learning on {} with LMVSVM".format(DATASET))

# datasets
X, Y, test_X, test_Y = load_uwave()

# get kernel

kernel = get_kernel(kname, None)

acc_list, acc_std_list = [], []
train_time_list = []
test_time_list = []

for L in landmarks:

    accuracies = []
    train_times = []
    test_times = []

    for it in range(ITER):

        # tuning
        t1 = time.time()

        tuning_acc = {}.fromkeys(c_range, 0.)    

        splitter = KFold(n_splits=CV)

        for train_index, val_index in splitter.split(X):

            train_x, val_x = X[train_index], X[val_index]
            train_y, val_y = Y[train_index], Y[val_index]

            lands = select_landmarks(train_x, L)
            proj_train_x = multiview_project(train_x, lands, kernel, 3)
            proj_val_x = multiview_project(val_x, lands, kernel, 3)

            for c in c_range:

                model = train(train_y.tolist(), proj_train_x.tolist(), '-c {} -s 2 -B 1 -q'.format(c))
                _, p_acc, _ = predict(val_y.tolist(), val_x.tolist(), model, '-q')
                tuning_acc[c] += p_acc[0]

        best_C = max(tuning_acc, key=tuning_acc.get)

        t2 = time.time()
        print("tuning time:", t2-t1)
        
        # training
        proj_train_x = multiview_project(X, lands, kernel, 3)
        model = train(Y.tolist(), proj_train_x.tolist(), '-c {} -s 2 -B 1 -q'.format(best_C))

        t3 = time.time()
        print("training time:", t3-t2)

        proj_test_x = multiview_project(test_X, lands, kernel, 3)
        p_label, p_acc, p_val = predict(test_Y.tolist(), proj_test_x.tolist(), model, '-q')

        t4 = time.time()
        print("testing time:", t4-t3)

        accuracies.append(p_acc[0])
        train_times.append(t3-t2)
        test_times.append(t4-t3)

    acc_list.append(mean(accuracies))
    acc_std_list.append(stdev(accuracies))
    train_time_list.append(mean(train_times))
    test_time_list.append(mean(test_times))

dict_to_csv({'accuracy': acc_list, 'error': acc_std_list, 'train_time': train_time_list, 'test_time': test_time_list, 'landmarks': landmarks},["nb_iter={},cv={}".format(ITER, CV)], PATH+".csv")
