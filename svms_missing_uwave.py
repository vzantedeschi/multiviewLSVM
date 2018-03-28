import numpy as np
import time
from statistics import mean, stdev

from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import rbf_kernel

from src.missing_views import set_random_views_to_value, laplacian_reconstruction
from src.svms import *
from src.utils import dict_to_csv, load_flower17, get_args, get_view_dict, twod_array, splits_generator, load_uwave

CV = 3

# ratios_missing = [0.05*i for i in range(1, 11)]
# c_range = [10**i for i in range(-3, 4)]

ratios_missing = [0.05]
c_range = [1]

kernel = rbf_kernel
sets = None
X, Y, test_X, test_Y = load_uwave()

value = 0.

ITER = 10
PATH = "results/view/uwave/missing/svms/laplacian"

print("learning on uwave with SVMs, missing views completed by Laplacian completion")

acc_list = []
std_list = []
train_time_list = []
test_time_list = []

for r in ratios_missing:
    print(r, "\n")
    mean_accuracies = []

    for i in range(ITER):

        accuracies = []
        train_times = []
        test_times = []

        for train_inds, val_inds, _ in splits_generator(X, CV, sets):

            train_x, train_y = X[train_inds], Y[train_inds]
            val_x, val_y = X[val_inds], Y[val_inds]

            # erase some views from training
            inc_train_x, train_y = set_random_views_to_value(train_x, train_y, r, r_type="none")

            # kernelize and reconstruct views
            k_train_x, inc_train_y = laplacian_reconstruction(inc_train_x, train_y, rbf_kernel)
            k_train_x = get_view_dict(k_train_x)
            k_val_x = get_view_dict(get_kernels(val_x, train_x, kernel=kernel))

            t1 = time.time()

            # tuning     
            tuning_acc = {}.fromkeys(c_range, 0.)
            for c in c_range:
                model = train(k_train_x, inc_train_y, c)
                pred = predict(k_val_x, val_y, model)

                tuning_acc[c] = accuracy_score(pred, val_y)
            best_C = max(tuning_acc, key=tuning_acc.get)

            t2 = time.time()
            print("tuning time:", t2-t1)

            # training
            inc_val_x, val_y = set_random_views_to_value(val_x, val_y, r, r_type="none")
            train_val_x = np.vstack((inc_train_x, inc_val_x))
            train_val_y = np.hstack((train_y, val_y))
            k_train_val_x, inc_train_val_y = laplacian_reconstruction(train_val_x, train_val_y, rbf_kernel)
            k_train_val_x = get_view_dict(k_train_val_x)


            model = train(k_train_val_x, inc_train_val_y, best_C)

            t3 = time.time()
            print("training time:", t3-t2)

            k_test_x = get_view_dict(get_kernels(test_X, train_x, kernel=kernel))

            pred = predict(k_test_x, test_Y, model)

            t4 = time.time()
            print("testing time:", t4-t3)

            acc = accuracy_score(pred, test_Y)*100
            print(acc)
            accuracies.append(acc)
            train_times.append(t3-t2)
            test_times.append(t4-t3)

        mean_accuracies.append(mean(accuracies))
        print(mean(accuracies))

    acc_list.append(mean(mean_accuracies))
    std_list.append(stdev(mean_accuracies))
    train_time_list.append(mean(train_times))
    test_time_list.append(mean(test_times))

dict_to_csv({'accuracy':acc_list,'error':std_list,'train_time':train_time_list,'test_time':test_time_list,'ratios':ratios_missing},["nb_iter={},cv={}".format(ITER,3)],PATH+".csv")
