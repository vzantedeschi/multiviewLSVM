import itertools
import numpy as np
import time
from statistics import mean, stdev

from sklearn.metrics import accuracy_score

from src.kernels import rbf_kernel
from src.mvml import *
from src.utils import dict_to_csv, load_flower17, get_view_dict

DATASET = "flower17"
appr_levels = [l/1020 for l in [10, 50, 100]]
eta_range = [10**i for i in range(-3, 3)]
lambda_range = [10**i for i in range(-8, 2)]
# appr_levels = [10/1020]
# lambda_range = [10**-3, 1]
# eta_range = [1]

ITER = 4
PATH = "results/view/{}/mvml".format(DATASET)

print("learning on {} with MVML".format(DATASET))

# datasets
Y, sets, X = load_flower17()

X = rbf_kernel(X)

acc_list = []
std_list = []
train_time_list = []
test_time_list = []

for a in appr_levels:

    mean_accuracies = []

    for i in range(ITER):

        accuracies = []
        train_times = []
        test_times = []

        for p in range(3):

            train_inds = sets[p][0]
            test_inds = sets[p][1]
            val_inds = sets[p][2]

            train_y, val_y = Y[train_inds], Y[val_inds]

            train_x = get_view_dict(get_kernels(X[train_inds], inds=train_inds))
            val_x = get_view_dict(get_kernels(X[val_inds], inds=train_inds))
            t1 = time.time()

            # tuning     
            tunin_acc = {}.fromkeys(itertools.product(lambda_range, eta_range), 0.)

            for l in lambda_range:
                for e in eta_range:

                    mvml = one_vs_all_mvml_train(train_x, train_y, 17, l, e, a)
                    pred = one_vs_all_mvml_predict(val_x, mvml)
                    p_acc = accuracy_score(val_y, pred)

                    tunin_acc[(l,e)] += p_acc

            best_l, best_e = max(tunin_acc, key=tunin_acc.get)
            t2 = time.time()
            print("tuning time:", t2-t1)
            # training

            train_val_inds = np.hstack((train_inds, val_inds))

            train_val_x = get_view_dict(get_kernels(X[train_val_inds], inds=train_val_inds))
            train_val_y = Y[train_val_inds]

            mvml = one_vs_all_mvml_train(train_val_x, train_val_y, 17, best_l, best_e, a)

            t3 = time.time()
            print("training time:", t3-t2)

            test_x = get_view_dict(get_kernels(X[test_inds], inds=train_val_inds))
            test_y = Y[test_inds]

            pred = one_vs_all_mvml_predict(test_x, mvml)
            p_acc = accuracy_score(test_y, pred)

            t4 = time.time()
            print("testing time:", t4-t3)

            accuracies.append(p_acc*100)
            train_times.append(t3-t2)
            test_times.append(t4-t3)

        mean_accuracies.append(mean(accuracies))
        print(mean(accuracies))

    acc_list.append(mean(mean_accuracies))
    std_list.append(stdev(mean_accuracies))
    train_time_list.append(mean(train_times))
    test_time_list.append(mean(test_times))

dict_to_csv({'accuracy':acc_list,'error':std_list,'train_time':train_time_list,'test_time':test_time_list,'appr_levels':appr_levels},["nb_iter={},cv={}".format(ITER,3)],PATH+".csv")
