import h5py
import numpy as np


def import_data(every=False):
    if every:
        electrodes = 25
    else:
        electrodes = 22
    X, y = [], []
    for i in range(9):
        A01T = h5py.File('data/A0' + str(i + 1) + 'T_slice.mat', 'r')
        X1 = np.copy(A01T['image'])
        X.append(X1[:, :electrodes, :])
        y1 = np.copy(A01T['type'])
        y1 = y1[0, 0:X1.shape[0]:1]
        y.append(np.asarray(y1, dtype=np.int32))

    for subject in range(9):
        delete_list = []
        for trial in range(288):
            if np.isnan(X[subject][trial, :, :]).sum() > 0:
                delete_list.append(trial)
        X[subject] = np.delete(X[subject], delete_list, 0)
        y[subject] = np.delete(y[subject], delete_list)
    y = [y[i] - np.min(y[i]) for i in range(len(y))]
    return X, y


def train_test_subject(X, y, train_all=True, standardize=True):

    l = np.random.permutation(len(X[0]))
    X_test = X[0][l[:50], :, :]
    y_test = y[0][l[:50]]

    if train_all:
        X_train = np.concatenate((X[0][l[50:], :, :], X[1], X[2], X[3], X[4], X[5], X[6], X[7], X[8]))
        y_train = np.concatenate((y[0][l[50:]], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8]))

    else:
        X_train = X[0][l[50:], :, :]
        y_train = y[0][l[50:]]

    X_train_mean = X_train.mean(0)
    X_train_var = np.sqrt(X_train.var(0))

    if standardize:
        X_train -= X_train_mean
        X_train /= X_train_var
        X_test -= X_train_mean
        X_test /= X_train_var

    X_train = np.transpose(X_train, (0, 2, 1))
    X_test = np.transpose(X_test, (0, 2, 1))

    return X_train, X_test, y_train, y_test


def train_test_total(X, y, standardize=True):

    X_total = np.concatenate((X[0], X[1], X[2], X[3], X[4], X[5], X[6], X[7], X[8]))
    y_total = np.concatenate((y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8]))

    l = np.random.permutation(len(X_total))
    X_test = X_total[l[:50], :, :]
    y_test = y_total[l[:50]]
    X_train = X_total[l[50:], :, :]
    y_train = y_total[l[50:]]

    X_train_mean = X_train.mean(0)
    X_train_var = np.sqrt(X_train.var(0))

    if standardize:
        X_train -= X_train_mean
        X_train /= X_train_var
        X_test -= X_train_mean
        X_test /= X_train_var

    X_train = np.transpose(X_train, (0, 2, 1))
    X_test = np.transpose(X_test, (0, 2, 1))

    return X_train, X_test, y_train, y_test
