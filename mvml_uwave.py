import itertools
import numpy as np
import time
from statistics import mean, stdev

from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import rbf_kernel

from src.mvml import one_vs_all_mvml_train, one_vs_all_mvml_predict, get_kernels
from src.utils import dict_to_csv, load_uwave, normalized_mse, multiview_kernels, splits_generator, get_view_dict

DATASET = "uwave"
appr_levels = [l/896 for l in [10, 50, 100, 200]]
eta_range = [10**i for i in range(-3, 3)]
lambda_range = [10**i for i in range(-8, 2)]
# appr_levels = [0.06]
# lambda_range = [1]
# eta_range = [1]

CV = 3
ITER = 4
PATH = "results/view/{}/mvml2".format(DATASET)

print("learning on {} with MVML".format(DATASET))

# datasets
X, Y, test_X, test_Y = load_uwave()

acc_list, std_list = [], []
train_time_list = []
test_time_list = []

for a in appr_levels:

    accuracies = []
    train_times = []
    test_times = []

    for it in range(ITER):

        # tuning
        t1 = time.time()

        tuning_acc = {}.fromkeys(itertools.product(lambda_range, eta_range), 0.)

        for train_inds, val_inds, _ in splits_generator(X, CV, None):

            train_y, val_y = Y[train_inds], Y[val_inds]

            train_x = get_view_dict(get_kernels(X[train_inds], X[train_inds], kernel=rbf_kernel))
            val_x = get_view_dict(get_kernels(X[val_inds], X[train_inds], kernel=rbf_kernel))

            for l in lambda_range:
                for e in eta_range:

                    mvml = one_vs_all_mvml_train(train_x, train_y, 8, l, e, a)
                    pred = one_vs_all_mvml_predict(val_x, mvml)
                    p_acc = accuracy_score(val_y, pred)

                    tuning_acc[(l,e)] += p_acc

        best_l, best_e = max(tuning_acc, key=tuning_acc.get)

        t2 = time.time()
        print("tuning time:", t2-t1)

        # training

        train_val_x = get_view_dict(get_kernels(X, X, kernel=rbf_kernel))
        mvml = one_vs_all_mvml_train(train_val_x, Y, 8, best_l, best_e, a)

        t3 = time.time()
        print("training time:", t3-t2)

        test_x = get_view_dict(get_kernels(test_X, X, kernel=rbf_kernel))
        
        pred = one_vs_all_mvml_predict(test_x, mvml)
        p_acc = accuracy_score(test_Y, pred)

        t4 = time.time()
        print("testing time:", t4-t3)

        accuracies.append(p_acc*100)
        train_times.append(t3-t2)
        test_times.append(t4-t3)

    acc_list.append(mean(accuracies))
    std_list.append(stdev(accuracies))
    train_time_list.append(mean(train_times))
    test_time_list.append(mean(test_times))

dict_to_csv({'accuracy':acc_list,'error':std_list,'train_time':train_times,'test_time':test_times},["nb_iter={},cv={}".format(ITER, CV)], PATH+".csv")
