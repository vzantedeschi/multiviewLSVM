import numpy as np
import time
from statistics import mean, stdev

from sklearn.metrics import accuracy_score

from src.missing_views import set_random_views_to_value, laplacian_reconstruction
from src.lmvsvm import *
from src.utils import dict_to_csv, load_flower17, get_args, get_view_dict, twod_array, splits_generator, multiview_kernels, rbf_kernel

args = get_args(__file__)
recons = args.reconstr

DATASET = "flower17"
L = 200

ratios_missing = [0.05*i for i in range(1, 11)]
c_range = [10**i for i in range(-3, 4)]
ratios_missing = [0.3]
c_range = [1]

# datasets
Y, sets, X = load_flower17(rbf_kernel)

ITER = 2
PATH = "results/{}/missing/lmvsvm/{}".format(DATASET, recons)

print("learning on flower with LMVSVM, missing views completed by {} completion. results saved in {}".format(recons, PATH))

acc_list = []
std_list = []
time_list = []

for r in ratios_missing:
    print(r, "\n")
    mean_accuracies = []

    for i in range(ITER):

        accuracies = []
        times = []

        # cross-validation
        for train_inds, val_inds, test_inds in splits_generator(Y, 3, sets):

            train_x, train_y = X[train_inds], Y[train_inds]
            val_x, val_y = X[val_inds], Y[val_inds]
            test_x, test_y = X[test_inds], Y[test_inds]

            lands = select_landmarks(train_x, L, inds=True)
            
            k_train_x = get_kernels(train_x, lands)
            k_val_x = get_kernels(val_x, lands)

            # erase some views from data
            k_train_x = set_random_views_to_value(k_train_x, r, recons)
            k_val_x = set_random_views_to_value(k_val_x, r, recons)

            t1 = time.time()

            if recons == "reconstruction":
                land_projections = get_kernels(X[lands], lands)
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

            k_train_val_x = np.vstack((k_train_x, k_val_x))
            k_train_val_y = np.hstack((train_y, val_y))

            model = train(k_train_val_x, k_train_val_y, best_C)

            k_test_x = get_kernels(test_x, lands)
            k_test_x = set_random_views_to_value(k_test_x, r, recons)

            t3 = time.time()

            if recons == "reconstruction":
                k_test_x = recontruct_views(k_test_x, land_projections)

            t4 = time.time()

            pred = predict(k_test_x, test_y, model)

            acc = accuracy_score(pred, test_y)*100
            print(acc)
            accuracies.append(acc)
            times.append(t2-t1+t4-t3)

        mean_accuracies.append(mean(accuracies))

    acc_list.append(mean(mean_accuracies))
    std_list.append(stdev(mean_accuracies))
    time_list.append(mean(times))

dict_to_csv({'accuracy':acc_list,'error':std_list,'time':time_list,'ratios':ratios_missing},["nb_iter={},cv={}".format(ITER,3)],PATH+".csv")
