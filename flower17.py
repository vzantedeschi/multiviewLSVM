import numpy as np
import time
from statistics import mean, stdev

from sklearn.metrics import accuracy_score

from src.baseline import *
from src.kernels import rbf_kernel
from src.missing_views import set_random_blocks_to_value
from src.utils import dict_to_csv, load_flower17, get_view_blocks

DATASET = "flower17"
C_RANGE = [10**i for i in range(-3, 4)]
# ratios_missing = [0.05*i for i in range(10)]

PATH = "results/view/{}/svm-per-view".format(DATASET)

print("learning on {} with a SVM per view".format(DATASET))

# datasets
indices, labels, sets, dist_matrices, _ = load_flower17(rbf_kernel)

# acc_list = []
# std_list = []
# train_time_list = []
# test_time_list = []

# for r in ratios_missing:

#     mean_accuracies = []

#     for i in range(ITER):

accuracies = []
train_times = []
test_times = []

for p in range(3):

    train_inds = sets[p][0]
    test_inds = sets[p][1]
    val_inds = sets[p][2]

    train_x, train_y = get_view_blocks(dist_matrices, train_inds, train_inds, 7), labels[train_inds]
    val_x, val_y = get_view_blocks(dist_matrices, val_inds, train_inds, 7), labels[val_inds]

    # erase some views from data
    # incomplete_train_x = set_random_blocks_to_value(train_x, r, L, 7, 0)
    # incomplete_val_x = set_random_blocks_to_value(val_x, r, L, 7, 0)
    # incomplete_test_x = set_random_blocks_to_value(test_x, r, L, 7, 0)

    t1 = time.time()

    tuning_accs = {}.fromkeys(C_RANGE, 0.)

    # tuning     
    for c in C_RANGE:
        models = train_svm_per_view(train_x, train_y, 7, c)
        pred = predict_svm_per_view(val_x, val_y, 7, models)
        tuning_accs[c] += accuracy_score(val_y, pred)

    best_C = max(tuning_accs, key=tuning_accs.get)
    t2 = time.time()
    print("tuning time:", t2-t1)

    # training
    train_val_inds = np.hstack((train_inds, val_inds))
    train_val_x, train_val_y = get_view_blocks(dist_matrices, train_val_inds, train_val_inds, 7), labels[train_val_inds]
    models = train_svm_per_view(train_val_x, train_val_y, 7, best_C)
    print(best_C)

    t3 = time.time()
    print("training time:", t3-t2)

    test_x, test_y = get_view_blocks(dist_matrices, test_inds, train_val_inds, 7), labels[test_inds]
    pred = predict_svm_per_view(test_x, test_y, 7, models)

    t4 = time.time()
    print("testing time:", t4-t3)

    accuracies.append(accuracy_score(pred, test_y)*100)
    train_times.append(t3-t2)
    test_times.append(t4-t3)

    # acc_list.append(mean(mean_accuracies))
    # std_list.append(stdev(mean_accuracies))
    # train_time_list.append(mean(train_times))
    # test_time_list.append(mean(test_times))

dict_to_csv({'accuracy':mean(accuracies),'error':stdev(accuracies),'train_time':mean(train_times),'test_time':mean(test_times)},["nb_iter={},cv={}".format(1,3)],PATH+".csv")
