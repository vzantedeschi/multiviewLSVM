import numpy as np
import time
from statistics import mean, stdev

from sklearn.metrics import accuracy_score

from src.kernels import rbf_kernel
from src.missing_views import set_random_views_to_value, laplacian_reconstruction
from src.svms import *
from src.utils import dict_to_csv, load_flower17, get_args, get_view_dict, twod_array, splits_generator, multiview_kernels

CV = 3

ratios_missing = [0.05*i for i in range(1, 11)]
c_range = [10**i for i in range(-3, 4)]

# ratios_missing = [0.05]
# c_range = [1]

# datasets
Y, sets, X = load_flower17(rbf_kernel)

ITER = 10
PATH = "results/view/flower17/missing/svms/laplacian"

print("learning on flower with SVMs, missing views completed by Laplacian completion")

acc_list = []
std_list = []
time_list = []

for r in ratios_missing:
    print(r, "\n")
    mean_accuracies = []

    for i in range(ITER):

        accuracies = []
        times = []

        # erase some views from training 
        x = set_random_views_to_value(X, r, r_type="none", sym=True)

        t0 = time.time()
        # kernelize and reconstruct views
        k_x, mask = laplacian_reconstruction(x)
        t10 = time.time()

        # cross-validation
        for train_inds, val_inds, test_inds in splits_generator(Y, CV, sets):

            train_val_inds = np.hstack((train_inds,val_inds))

            train_y = Y[train_inds[mask[train_inds]]]
            val_y = Y[val_inds[mask[val_inds]]]
            test_y = Y[test_inds[mask[test_inds]]]
            train_val_y = Y[train_val_inds[mask[train_val_inds]]]

            train_inds = mask[train_inds]
            val_inds = mask[val_inds]
            test_inds = mask[test_inds]
            train_val_inds = mask[train_val_inds]

            k_train_x = get_view_dict(k_x[np.ix_(train_inds,train_inds)])
            k_val_x = get_view_dict(k_x[np.ix_(val_inds,train_inds)])

            t1 = time.time()

            # tuning     
            tuning_acc = {}.fromkeys(c_range, 0.)
            for c in c_range:
                model = train(k_train_x, train_y, c)
                pred = predict(k_val_x, val_y, model)

                tuning_acc[c] = accuracy_score(pred, val_y)

            best_C = max(tuning_acc, key=tuning_acc.get)

            t2 = time.time()
            print("tuning time:", t2-t1)

            # training
            k_train_val_x = get_view_dict(k_x[np.ix_(train_val_inds,train_val_inds)])
            model = train(k_train_val_x, train_val_y, best_C)

            t3 = time.time()
            print("training time:", t3-t2)

            k_test_x = get_view_dict(k_x[np.ix_(test_inds,train_val_inds)])

            pred = predict(k_test_x, test_y, model)

            t4 = time.time()
            print("testing time:", t4-t3)

            acc = accuracy_score(pred, test_y)*100
            print(acc)
            accuracies.append(acc)
            times.append(t10-t0)

        mean_accuracies.append(mean(accuracies))
        print(mean(accuracies))

    acc_list.append(mean(mean_accuracies))
    std_list.append(stdev(mean_accuracies))
    time_list.append(mean(times))

dict_to_csv({'accuracy':acc_list,'error':std_list,'time':time_list,'ratios':ratios_missing},["nb_iter={},cv={}".format(ITER,CV)],PATH+".csv")
