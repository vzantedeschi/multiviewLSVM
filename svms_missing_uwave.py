import numpy as np
import time
from statistics import mean, stdev

from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import rbf_kernel

from src.missing_views import set_random_views_to_value, laplacian_reconstruction
from src.svms import *
from src.utils import dict_to_csv, load_flower17, get_args, get_view_dict, twod_array, splits_generator, load_uwave, multiview_kernels

CV = 3

ratios_missing = [0.05*i for i in range(1, 11)]
c_range = [10**i for i in range(-3, 4)]

#ratios_missing = [0.05]
#c_range = [1]

X, Y, test_X, test_Y = load_uwave()

ITER = 2
PATH = "results/view/uwave/missing/svms/none"

print("learning on uwave with SVMs, missing views completed by Laplacian completion")

acc_list = []
std_list = []
time_list = []

for r in ratios_missing:
    print(r, "\n")
    accuracies = []
    times = []

    for i in range(ITER):

        print(X.shape, Y.shape)
        # erase some views from training 
        x = set_random_views_to_value(X, r, r_type="none")
        test_x = set_random_views_to_value(test_X, r, r_type="none")

        # kernelize and reconstruct views
        t0 = time.time()
        # k_x, mask, mask2 = laplacian_reconstruction(x, rbf_kernel, test_x)
        mask, mask2 = np.ones(len(Y), dtype=bool), np.ones(len(test_Y), dtype=bool)
        
        k_x = multiview_kernels(np.vstack((x, test_x)), np.vstack((x, test_x)), rbf_kernel)
        y, test_y = Y[mask], test_Y[mask2]

        t10 = time.time()

        # cross-validation
        for train_inds, val_inds, _ in splits_generator(y, CV, None):

            train_y = y[train_inds]
            val_y = y[val_inds]

            k_train_x = get_view_dict(k_x[np.ix_(train_inds,train_inds)])
            k_val_x = get_view_dict(k_x[np.ix_(val_inds,train_inds)])

            t1 = time.time()

            # tuning     
            tuning_acc = {}.fromkeys(c_range, 0.)
            for c in c_range:
                model = train(k_train_x, train_y, c)
                pred = predict(k_val_x, val_y, model)

                tuning_acc[c] += accuracy_score(pred, val_y)

        best_C = max(tuning_acc, key=tuning_acc.get)

        t2 = time.time()
        print("tuning time:", t2-t1)

        # training
        train_val_inds = np.hstack((train_inds,val_inds))
        k_train_val_x = get_view_dict(k_x[np.ix_(train_val_inds,train_val_inds)])
        model = train(k_train_val_x, y[train_val_inds], best_C)

        t3 = time.time()
        print("training time:", t3-t2)

        test_inds = np.arange(len(test_y))+len(y)
        k_test_x = get_view_dict(k_x[np.ix_(test_inds,train_val_inds)])

        pred = predict(k_test_x, test_y, model)

        t4 = time.time()
        print("testing time:", t4-t3)

        acc = accuracy_score(pred, test_y)*100
        print(acc)
        accuracies.append(acc)
        times.append(t10-t0)

    acc_list.append(mean(accuracies))
    std_list.append(stdev(accuracies))
    time_list.append(mean(times))

dict_to_csv({'accuracy':acc_list,'error':std_list,'times':time_list,'ratios':ratios_missing},["nb_iter={},cv={}".format(ITER,3)],PATH+".csv")
