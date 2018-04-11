import numpy as np
import time
from statistics import mean, stdev

from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import rbf_kernel

from src.lmvsvm import *
from src.missing_views import set_random_views_to_value
from src.utils import dict_to_csv, load_flower17, get_args, get_view_dict, twod_array, splits_generator, load_uwave

args = get_args(__file__)
recons = args.reconstr

DATASET = "uwave"
L = 200

ratios_missing = [0.05*i for i in range(1, 11)]
c_range = [10**i for i in range(-3, 4)]
ratios_missing = [0.3]
c_range = [1]

X, Y, test_X, test_Y = load_uwave()

ITER = 2
PATH = "results/{}/missing/lmvsvm/{}".format(DATASET, recons)

print("learning on {}, missing views completed by {}. results saved in {}".format(DATASET, recons, PATH))

acc_list = []
std_list = []
times = []

for r in ratios_missing:
    print(r, "\n")

    accuracies = []
    rec_times = []

    for train_inds, val_inds, _ in splits_generator(X, ITER, None):

        train_x, train_y = X[train_inds], Y[train_inds]
        val_x, val_y = X[val_inds], Y[val_inds]

        lands = select_landmarks(train_x, L)
        
        k_train_x = get_kernels(train_x, lands, rbf_kernel)
        k_val_x = get_kernels(val_x, lands, rbf_kernel)

        # erase some views from data
        k_train_x = set_random_views_to_value(k_train_x, r, recons)
        k_val_x = set_random_views_to_value(k_val_x, r, recons)

        t1 = time.time()

        if recons == "reconstruction":
            land_projections = get_kernels(lands, lands, rbf_kernel)
            k_train_x = recontruct_views(k_train_x, land_projections)
            k_val_x = recontruct_views(k_val_x, land_projections)

        t2 = time.time()

        # tuning     
        tuning_acc = {}.fromkeys(c_range, 0.)
        for c in c_range:
            model = train(k_train_x, train_y, c)
            pred = predict(k_val_x, val_y, model)

            tuning_acc[c] += accuracy_score(pred, val_y)
            print(tuning_acc[c])

        best_C = max(tuning_acc, key=tuning_acc.get)

        # training

        k_train_val_x = np.vstack((k_train_x, k_val_x))
        model = train(k_train_val_x, Y, best_C)

        k_test_x = get_kernels(test_X, lands, rbf_kernel)
        k_test_x = set_random_views_to_value(k_test_x, r, recons)

        t3 = time.time()

        if recons == "reconstruction":
            k_test_x = recontruct_views(k_test_x, land_projections)

        t4 = time.time()

        pred = predict(k_test_x, test_Y, model)

        accuracies.append(accuracy_score(pred, test_Y)*100)
        rec_times.append(t2-t1+t4-t3)

    acc_list.append(mean(accuracies))
    std_list.append(stdev(accuracies))
    times.append(mean(rec_times))

dict_to_csv({'accuracy':acc_list,'error':std_list,'time':times,'ratios':ratios_missing},["nb_iter={}".format(ITER)],PATH+".csv")
