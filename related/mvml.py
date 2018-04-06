import numpy as np

"""
    Copyright (C) 2018  Riikka Huusari

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.


This file contains algorithms for Multi-View Metric Learning (MVML) as introduced in

Riikka Huusari, Hachem Kadri and Cécile Capponi:
Multi-View Metric Learning in Vector-Valued Kernel Spaces
in International Conference on Artificial Intelligence and Statistics (AISTATS) 2018

Usage (see also demo.py for a more detailed example):
    create a MVML object via:
        mvml = MVML(kernel_dict, label_vector, regression_parameter_list, nystrom_param)
    learn the model:
        A, g, w = mvml.learn_mvml()
    predict with the model:
        predictions = predict_mvml(test_kernel_dict, g, w)

(parameter names as in the paper)

Code is tested with Python 3.5.2 and numpy 1.12.1
"""

class MVML:

    def __init__(self, kernels, labels, regression_params, nystrom_param):

        """
        :param kernels: dictionary for kernel matrices of all the views, keys should be numbers [0,...,views-1]
        :param labels: array of length n_samples containing the classification/regression labels for training data
        :param regression_params: array/list of regression parameters, first for basic regularization, second for
                                  regularization of A (not necessary if A is not learned)
        :param nystrom_param: value between 0 and 1 indicating level of nyström approximation; 1 = no approximation
        """

        # calculate nyström approximation (if used)

        self.nystrom_param = nystrom_param
        n = kernels[0].shape[0]
        self.n_approx = int(np.floor(nystrom_param * n))  # number of samples in approximation, equals n if no approx.

        if nystrom_param < 1:
            self._calc_nystrom(kernels)
        else:
            self.U_dict = kernels

        self.Y = labels
        self.reg_params = regression_params

    def learn_mvml(self, learn_A=1, learn_w=0, n_loops=6):

        """
        :param learn_A: choose if A is learned or not: 1 - yes (default); 2 - yes, sparse; 3 - no (MVML_Cov); 4 - no (MVML_I)
        :param learn_w: choose if w is learned or not: 0 - no (uniform 1/views, default setting), 1 - yes
        :param n_loops: maximum number of iterations in MVML, usually something like default 6 is already converged
        :return: A (metrcic matrix - either fixed or learned), g (solution to learning problem), w (weights - fixed or learned)
        """

        views = len(self.U_dict)
        n = self.U_dict[0].shape[0]
        lmbda = self.reg_params[0]
        if learn_A < 3:
            eta = self.reg_params[1]

        # ========= initialize A =========

        # positive definite initialization (with multiplication with the U matrices if using approximation)
        A = np.zeros((views * self.n_approx, views * self.n_approx))
        if learn_A < 3:
            for v in range(views):
                if self.nystrom_param < 1:
                    A[v * self.n_approx:(v + 1) * self.n_approx, v * self.n_approx:(v + 1) * self.n_approx] = \
                        np.dot(np.transpose(self.U_dict[v]), self.U_dict[v])
                else:
                    A[v * self.n_approx:(v + 1) * self.n_approx, v * self.n_approx:(v + 1) * self.n_approx] = np.eye(n)
        # otherwise initialize like this if using MVML_Cov
        elif learn_A == 3:
            for v in range(views):
                for vv in range(views):
                    if self.nystrom_param < 1:
                        A[v * self.n_approx:(v + 1) * self.n_approx, vv * self.n_approx:(vv + 1) * self.n_approx] = \
                            np.dot(np.transpose(self.U_dict[v]), self.U_dict[vv])
                    else:
                        A[v * self.n_approx:(v + 1) * self.n_approx, vv * self.n_approx:(vv + 1) * self.n_approx] = \
                            np.eye(n)
        # or like this if using MVML_I
        elif learn_A == 4:
            for v in range(views):
                if self.nystrom_param < 1:
                    A[v * self.n_approx:(v + 1) * self.n_approx, v * self.n_approx:(v + 1) * self.n_approx] = \
                        np.eye(self.n_approx)
                else:
                    # it might be wise to make a dedicated function for MVML_I if using no approximation
                    # - numerical errors are more probable this way using inverse
                    A[v * self.n_approx:(v + 1) * self.n_approx, v * self.n_approx:(v + 1) * self.n_approx] = \
                        np.linalg.pinv(self.U_dict[v])  # U_dict holds whole kernels if no approx

        # ========= initialize w, allocate g =========
        w = (1 / views) * np.ones((views, 1))
        g = np.zeros((views * self.n_approx, 1))

        # ========= learn =========
        loop_counter = 0
        while True:

            if loop_counter > 0:
                g_prev = np.copy(g)
                A_prev = np.copy(A)
                w_prev = np.copy(w)

            # ========= update g =========

            # first invert A
            try:
                A_inv = np.linalg.pinv(A + 1e-09 * np.eye(views * self.n_approx))
            except np.linalg.linalg.LinAlgError:
                try:
                    A_inv = np.linalg.pinv(A + 1e-06 * np.eye(views * self.n_approx))
                except ValueError:
                    return A_prev, g_prev
            except ValueError:
                return A_prev, g_prev

            # then calculate g (block-sparse multiplications in loop) using A_inv
            for v in range(views):
                for vv in range(views):
                    A_inv[v * self.n_approx:(v + 1) * self.n_approx, vv * self.n_approx:(vv + 1) * self.n_approx] = \
                        w[v] * w[vv] * np.dot(np.transpose(self.U_dict[v]), self.U_dict[vv]) + \
                        lmbda * A_inv[v * self.n_approx:(v + 1) * self.n_approx,
                                      vv * self.n_approx:(vv + 1) * self.n_approx]
                g[v * self.n_approx:(v + 1) * self.n_approx, 0] = np.dot(w[v] * np.transpose(self.U_dict[v]), self.Y)

            try:
                g = np.dot(np.linalg.pinv(A_inv), g)  # here A_inv isn't actually inverse of A (changed in above loop)
            except np.linalg.linalg.LinAlgError:
                g = np.linalg.solve(A_inv, g)

            # ========= check convergence =========

            if learn_A > 2 and learn_w != 1:  # stop at once if only g is to be learned
                break

            if loop_counter > 0:

                # convergence criteria
                g_diff = np.linalg.norm(g - g_prev) / np.linalg.norm(g_prev)
                A_diff = np.linalg.norm(A - A_prev, ord='fro') / np.linalg.norm(A_prev, ord='fro')
                if g_diff < 1e-4 and A_diff < 1e-4:
                    break

            if loop_counter >= n_loops:  # failsafe
                break

            # ========= update A =========
            if learn_A == 1:
                A = self._learn_A(A, g, lmbda, eta)
            elif learn_A == 2:
                A = self._learn_blocksparse_A(A, g, views, self.n_approx, lmbda, eta)

            # ========= update w =========
            if learn_w == 1:
                Z = np.zeros((n, views))
                for v in range(views):
                    Z[:, v] = np.dot(self.U_dict[v], g[v * self.n_approx:(v + 1) * self.n_approx]).ravel()
                w = np.dot(np.linalg.pinv(np.dot(np.transpose(Z), Z)), np.dot(np.transpose(Z), self.Y))

            loop_counter += 1

        return A, g, w

    def predict_mvml(self, test_kernels, g, w):

        """
        :param test_kernels: dictionary of test kernels (as the dictionary of kernels in __init__)
        :param g: g, learning solution that is learned in learn_mvml
        :param w: w, weights for combining the solutions of views, learned in learn_mvml
        :return: (regression) predictions, array of size  test_samples*1
        """

        views = len(self.U_dict)
        t = test_kernels[0].shape[0]

        X = np.zeros((t, views * self.n_approx))
        for v in range(views):
            if self.nystrom_param < 1:
                X[:, v * self.n_approx:(v + 1) * self.n_approx] = w[v] * \
                                                                  np.dot(test_kernels[v][:, 0:self.n_approx],
                                                                         self.W_sqrootinv_dict[v])
            else:
                X[:, v * self.n_approx:(v + 1) * self.n_approx] = w[v] * test_kernels[v]

        return np.dot(X, g)

    def _calc_nystrom(self, kernels):

        # calculates the nyström approximation for all the kernels in the given dictionary

        self.U_dict = {}
        self.W_sqrootinv_dict = {}

        for v in range(len(kernels)):

            kernel = kernels[v]

            E = kernel[:, 0:self.n_approx]
            W = E[0:self.n_approx, :]
            Ue, Va, _ = np.linalg.svd(W)
            vak = Va[0:self.n_approx]
            inVa = np.diag(vak ** (-0.5))
            U_v = np.dot(E, np.dot(Ue[:, 0:self.n_approx], inVa))
            self.U_dict[v] = U_v
            self.W_sqrootinv_dict[v] = np.dot(Ue[:, 0:self.n_approx], inVa)

    def _learn_A(self, A, g, lmbda, eta):

        # basic gradient descent

        stepsize = 0.5
        if stepsize*eta >= 0.5:
            stepsize = 0.9*(1/(2*eta))  # make stepsize*eta < 0.5

        loops = 0
        not_converged = True
        while not_converged:

            A_prev = np.copy(A)

            A_pinv = np.linalg.pinv(A)
            A = (1-2*stepsize*eta)*A + stepsize*lmbda*np.dot(np.dot(A_pinv, g), np.dot(np.transpose(g), A_pinv))

            if loops > 0:
                prev_diff = diff
            diff = np.linalg.norm(A - A_prev) / np.linalg.norm(A_prev)

            if loops > 0 and prev_diff > diff:
                A = A_prev
                stepsize = stepsize*0.1

            if diff < 1e-5:
                not_converged = False

            if loops > 10:
                not_converged = False

            loops += 1

        return A

    def _learn_blocksparse_A(self, A, g, views, m, lmbda, eta):

        # proximal gradient update method

        converged = False
        rounds = 0

        L = lmbda * np.linalg.norm(np.dot(g, g.T))
        # print("L ", L)

        while not converged and rounds < 100:

            # no line search - this has worked well enough experimentally
            A = self._proximal_update(A, views, m, L, g, lmbda, eta)

            # convergence
            if rounds > 0:
                A_diff = np.linalg.norm(A - A_prev) / np.linalg.norm(A_prev)

                if A_diff < 1e-3:
                    converged = True

            A_prev = np.copy(A)

            rounds += 1

        return A

    def _proximal_update(self, A_prev, views, m, L, D, lmbda, gamma):

        # proximal update

        # the inverse is not always good to compute - in that case just return the previous one and end the search
        try:
            A_prev_inv = np.linalg.pinv(A_prev)
        except np.linalg.linalg.LinAlgError:
            try:
                A_prev_inv = np.linalg.pinv(A_prev + 1e-6 * np.eye(views * m))
            except np.linalg.linalg.LinAlgError:
                return A_prev
            except ValueError:
                return A_prev
        except ValueError:
            return A_prev

        if np.any(np.isnan(A_prev_inv)):
            # just in case the inverse didn't return a proper solution (happened once or twice)
            return A_prev

        A_tmp = A_prev + (lmbda / L) * np.dot(np.dot(A_prev_inv.T, D), np.dot(np.transpose(D), A_prev_inv.T))

        # if there is one small negative eigenvalue this gets rid of it
        try:
            val, vec = np.linalg.eigh(A_tmp)
        except np.linalg.linalg.LinAlgError:
            return A_prev
        except ValueError:
            return A_prev
        val[val < 0] = 0

        A_tmp = np.dot(vec, np.dot(np.diag(val), np.transpose(vec)))
        A_new = np.zeros((views*m, views*m))

        # proximal update, group by group (symmetric!)
        for v in range(views):
            for vv in range(v + 1):
                if v != vv:
                    if np.linalg.norm(A_tmp[v * m:(v + 1) * m, vv * m:(vv + 1) * m]) != 0:
                        multiplier = 1 - gamma / (2 * np.linalg.norm(A_tmp[v * m:(v + 1) * m, vv * m:(vv + 1) * m]))
                        if multiplier > 0:
                            A_new[v * m:(v + 1) * m, vv * m:(vv + 1) * m] = multiplier * A_tmp[v * m:(v + 1) * m,
                                                                                               vv * m:(vv + 1) * m]
                            A_new[vv * m:(vv + 1) * m, v * m:(v + 1) * m] = multiplier * A_tmp[vv * m:(vv + 1) * m,
                                                                                               v * m:(v + 1) * m]
                else:
                    if (np.linalg.norm(A_tmp[v * m:(v + 1) * m, v * m:(v + 1) * m])) != 0:
                        multiplier = 1 - gamma / (np.linalg.norm(A_tmp[v * m:(v + 1) * m, v * m:(v + 1) * m]))
                        if multiplier > 0:
                            A_new[v * m:(v + 1) * m, v * m:(v + 1) * m] = multiplier * A_tmp[v * m:(v + 1) * m,
                                                                                             v * m:(v + 1) * m]

        return A_new



# ------------------------------------------------------------------------------- MY PATCH
from src.utils import multiview_kernels, select_from_multiple_views

def one_vs_all_mvml_train(train_x, train_y, nb_classes, l, e, a, *args):
    models = {}

    for c in range(nb_classes):

        y = train_y.copy()
        y[train_y == c] = 1
        y[train_y != c] = -1

        clf = MVML(train_x, y, [l, e], nystrom_param=a)
        A, g, w = clf.learn_mvml()

        models[c] = {"clf": clf, "a":A, "g":g, "w":w}

    return models

def one_vs_all_mvml_predict(test_x, models, *args):

    predictions = []

    for k, values in models.items():
        predictions.append(values["clf"].predict_mvml(test_x, values["g"], values["w"]))

    return np.argmax(np.hstack(predictions), axis=1)

def get_kernels(x1, x2=None, inds=None, kernel=None):

    if kernel is None:
        x = select_from_multiple_views(x1, inds)
    else:
        x = multiview_kernels(x1, x2, kernel)

    return x
