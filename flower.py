import numpy as np
import time
from statistics import mean, stdev

from sklearn.metrics import accuracy_score

from src.lmvsvm import train, predict, get_kernels
from src.utils import dict_to_csv, load_flower17, select_landmarks, select_from_multiple_views, rbf_kernel, twod_array

DATASET = "flower17"
landmarks = [10, 50, 100, 200, 400, 500, 597]
C_RANGE = [10**i for i in range(-3, 4)]

ITER = 5
PATH = "results/{}/lmvsvm".format(DATASET)

print("learning on {} with L3SVM. results saved in {}".format(DATASET, PATH))

# datasets
labels, sets, dist_matrices = load_flower17(rbf_kernel)

acc_list = []
std_list = []
train_time_list = []
test_time_list = []

for L in landmarks:

    mean_accuracies = []

    for i in range(ITER):

        accuracies = []
        train_times = []
        test_times = []

        for p in range(3):

            train_inds = sets[p][0]
            test_inds = sets[p][1]
            val_inds = sets[p][2]

            lands = select_landmarks(train_inds, L)

            train_x, train_y = twod_array(get_kernels(dist_matrices[train_inds], lands)), labels[train_inds]
            val_x, val_y = twod_array(get_kernels(dist_matrices[val_inds], lands)), labels[val_inds]
            test_x, test_y = twod_array(get_kernels(dist_matrices[test_inds], lands)), labels[test_inds]

            t1 = time.time()

            # tuning     
            tuning_acc = {}.fromkeys(C_RANGE, 0.)

            for c in C_RANGE:
                model = train(train_x, train_y, c)
                pred = predict(val_x, val_y, model)

                tuning_acc[c] = accuracy_score(pred, val_y)

            best_C = max(tuning_acc, key=tuning_acc.get)

            t2 = time.time()
            print("tuning time:", t2-t1)

            # training
            train_val_y = np.hstack((train_y, val_y))
            train_val_x = np.vstack((train_x, val_x))

            model = train(train_val_x, train_val_y, best_C)

            t3 = time.time()
            print("training time:", t3-t2)

            pred = predict(test_x, test_y, model)

            t4 = time.time()
            print("testing time:", t4-t3)

            accuracies.append(accuracy_score(pred, test_y)*100)
            train_times.append(t3-t2)
            test_times.append(t4-t3)

        mean_accuracies.append(mean(accuracies))

    acc_list.append(mean(mean_accuracies))
    std_list.append(stdev(mean_accuracies))
    train_time_list.append(mean(train_times))
    test_time_list.append(mean(test_times))

dict_to_csv({'accuracy':acc_list,'error':std_list,'train_time':train_time_list,'test_time':test_time_list,'landmarks':landmarks},["nb_iter={},cv={}".format(ITER,3)],PATH+".csv")
