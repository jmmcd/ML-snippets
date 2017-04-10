#!/usr/bin/env python

"""Demos for density-based one-class classification. Some are copied
from http://air.ug/~jquinn/software/lsanomaly.html.

"""

import numpy as np
import pylab as plt
import scipy.io

from dbocc import SingleGaussianDensity, NaiveBayesDensity, MultivariateGaussianDensity, NegativeMeanDistance, DensityBasedOneClassClassifier, KernelDensity



def toy_data():
    # normal training set
    X = np.array([
        [1.0, 1.0],
        [1.1, 1.1],
        [1.0, 1.2],
        [1.1, 1.3],
        [1.0, 1.1],
        [1.4, 1.1],
        [1.1, 1.2]
    ])

    # test set with some normal, some anomaly
    X_test = np.array([
        [1.1, 1.1],
        [1.0, 1.2],
        [1.1, 1.3],
        [1.0, 1.1],
        [1.1, 1.35],
        [1.6, 0.9]
    ])
    # labels for test set
    y_test = np.array([False, False, False, False, False, True])

    return X, X_test, y_test

def process_wbcd():
    d = np.genfromtxt("wbcd.dat", delimiter=",")

    # discard the '?' values in wdbc
    d = d[~np.isnan(d).any(axis=1)]

    # shuffle
    np.random.seed(0)
    np.random.shuffle(d)
    dX = d[:, 1:-1] # discard the first column (ids) and the last (labels)
    dy = d[:, -1]
    dy = dy > 2.5 # in wdbc, 2 = benign, 4 = malignant

    # separate into normal and anomaly
    dX0 = dX[~dy]
    dX1 = dX[dy]
    dy0 = dy[~dy]
    dy1 = dy[dy]

    split = 0.5
    idx = int(split * len(dX0))

    # train_X is half of the normal class
    train_X = dX0[:idx]
    # test set is the other half of the normal class and all of the anomaly class
    test_X = np.concatenate((dX0[idx:], dX1))
    test_y = np.concatenate((dy0[idx:], dy1))

    return train_X, test_X, test_y

def process_server():
    """In this dataset (available from the Coursera Machine Learning
    course, lesson 9 -- copyright Andrew Ng), we have a training set X
    and a validation set Xval, yval. It represents features of a
    network/compute server, eg CPU usage and network traffic. In
    Coursera the validation set is used to select a value for the
    threshold. But for our purposes we'll call it a test set
    instead."""
    d = scipy.io.loadmat("coursera_ml_ex8data2.mat")
    train_X = d["X"]
    test_X = d["Xval"]
    test_y = d["yval"].astype('bool') # Matlab saves as ints
    test_y.shape = (test_y.shape[0],) # of shape (100, 1)
    return train_X, test_X, test_y

def test():

    # several functions that will give us a dataset (X_train, X_test, y_test)
    fns = (toy_data, process_wbcd, process_server)

    denss = [
        SingleGaussianDensity,
        NaiveBayesDensity,
        MultivariateGaussianDensity,
        KernelDensity,
        NegativeMeanDistance
    ]

    scalers = [
        preprocessing.StandardScaler,
        preprocessing.MinMaxScaler,
        None
    ]

    for data_fn in fns:
        train_X, test_X, test_y = data_fn()
        test_X0 = test_X[~test_y]
        test_X1 = test_X[test_y]

        for dens in denss:
            for scaler in scalers:
                print(data_fn.__name__)
                print("-----------------")
                print(dens.__name__)
                if scaler:
                    print(scaler.__name__)
                else:
                    print("No scaling")

                c = DensityBasedOneClassClassifier(dens=dens(), scaler=scaler)
                c.fit(train_X)

                # visualise
                d0 = c.score_samples(test_X0)
                d1 = c.score_samples(test_X1)
                plt.hist((d0, d1), bins=30)
                plt.savefig("hist_" + data_fn.__name__ + "_" + dens.__name__ + ".png")
                plt.close()

                # predict and evaluate
                yhat = c.predict(test_X)
                acc = np.mean(yhat == test_y)

                # thanks Loi!
                yhat_X0 = c.predict(test_X0)
                acc_X0 = np.mean(yhat_X0 == False)
                yhat_X1 = c.predict(test_X1)
                acc_X1 = np.mean(yhat_X1 == True)

                print("acc: %.2f" % acc)
                print("acc X0: %.2f" % acc_X0)
                print("acc X1: %.2f" % acc_X1)
                print()

test()
