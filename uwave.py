import numpy as np
import time
from statistics import mean, stdev

from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import rbf_kernel

from src.lmvsvm import *
from src.utils import dict_to_csv, load_uwave, select_landmarks, splits_generator, twod_array

DATASET = "uwave"

landmarks = [10, 50, 100, 200, 400, 500, 597]
# landmarks = [200]
c_range = [10**i for i in range(-3, 4)]

ITER = 5
CV = 3
PATH = "results/{}/kernel-lmvsvm".format(DATASET)

print("learning on {} with LMVSVM. results saved in {}".format(DATASET, PATH))

# datasets
X, Y, test_X, test_Y = load_uwave()

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

        for train_inds, val_inds, _ in splits_generator(X, CV, None):

            train_x, train_y = X[train_inds], Y[train_inds]
            val_x, val_y = X[val_inds], Y[val_inds]

            lands = select_landmarks(train_x, L)
            k_train_x = twod_array(get_kernels(train_x, lands, kernel=rbf_kernel))
            k_val_x = twod_array(get_kernels(val_x, lands, kernel=rbf_kernel))

            for c in c_range:

                model = train(k_train_x, train_y, c)
                pred = predict(k_val_x, val_y, model)
                tuning_acc[c] += accuracy_score(pred, val_y)

        best_C = max(tuning_acc, key=tuning_acc.get)

        t2 = time.time()
        print("tuning time:", t2-t1)
        
        # training
        k_train_val_x = twod_array(get_kernels(X, lands, kernel=rbf_kernel))
        model = train(k_train_val_x, Y, best_C)

        t3 = time.time()
        print("training time:", t3-t2)

        k_test_x = twod_array(get_kernels(test_X, lands, kernel=rbf_kernel))
        pred = predict(k_test_x, test_Y, model)

        t4 = time.time()
        print("testing time:", t4-t3)

        accuracies.append(accuracy_score(pred, test_Y)*100)
        print(accuracy_score(pred, test_Y)*100)
        train_times.append(t3-t2)
        test_times.append(t4-t3)

    acc_list.append(mean(accuracies))
    acc_std_list.append(stdev(accuracies))
    train_time_list.append(mean(train_times))
    test_time_list.append(mean(test_times))

dict_to_csv({'accuracy': acc_list, 'error': acc_std_list, 'train_time': train_time_list, 'test_time': test_time_list, 'landmarks': landmarks},["nb_iter={},cv={}".format(ITER, CV)], PATH+".csv")
