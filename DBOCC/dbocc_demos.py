#!/usr/bin/env python

"""Demos for density-based one-class classification.

"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score, roc_curve, auc
import random

# baselines
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest

# our methods
from dbocc import SingleGaussianDensity, NaiveBayesDensity, MultivariateGaussianDensity, NegativeMeanDistance, KernelDensity, DensityBasedOneClassClassifier




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
    y_test = np.where(y_test, 1, -1)

    return X, X_test, y_test


def toy_data_with_zero_variance():
    # normal training set
    X = np.array([
        [1.0, 1.0, 1.0],
        [1.1, 1.1, 1.0],
        [1.0, 1.2, 1.0],
        [1.1, 1.3, 1.0],
        [1.0, 1.1, 1.0],
        [1.4, 1.1, 1.0],
        [1.1, 1.2, 1.0]
    ])

    # test set with some normal, some anomaly
    X_test = np.array([
        [1.1, 1.1, 1.0],
        [1.0, 1.2, 1.0],
        [1.1, 1.3, 1.0],
        [1.0, 1.1, 1.0],
        [1.1, 1.35, 1.0],
        [1.6, 0.9, 1.0]
    ])
    # labels for test set
    y_test = np.array([False, False, False, False, False, True])
    y_test = np.where(y_test, 1, -1)

    return X, X_test, y_test


def process_wbcd():
    d = np.genfromtxt("wbcd.dat", delimiter=",")

    # discard the '?' values in wdbc
    d = d[~np.isnan(d).any(axis=1)]

    # shuffle
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
    test_y = np.where(test_y, 1, -1)

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
    test_y = np.where(test_y, 1, -1)
    return train_X, test_X, test_y


def actigraphy(whichuser=1):
    """Some actigraphy data not yet made publicly available."""
    train_X = np.load("actigraphy_standardised_train_X.npy")
    train_y = np.load("actigraphy_train_y.npy")
    test_X = np.load("actigraphy_standardised_test_X.npy")
    test_y = np.load("actigraphy_test_y.npy")
    train_X = train_X[train_y == whichuser]
    test_y = test_y != whichuser
    test_y = np.where(test_y, 1, -1)
    return train_X, test_X, test_y


def test():

    random.seed(0)
    np.random.seed(0)

    # several functions that will give us a dataset of the form
    # (X_train, X_test, y_test)
    fns = (toy_data, process_wbcd, process_server,
           toy_data_with_zero_variance,)

    # fns = (
    #     lambda: actigraphy(whichuser=i) for i in range(1, 10)
    # )

    denss = [
        SingleGaussianDensity,
        NaiveBayesDensity,
        MultivariateGaussianDensity,
        KernelDensity,
        NegativeMeanDistance
    ]

    scalers = [
        None,
        preprocessing.MinMaxScaler,
        preprocessing.StandardScaler,
    ]

    clfs = {"DBOCC_" + type(dens).__name__ + "_None":
            DensityBasedOneClassClassifier(dens=dens(),
                                           scaler=scaler)
            for dens in denss
            for scaler in scalers}

    baselines = {
        "OneClassSVM": OneClassSVM(gamma=1e-5),
        # "EllipticEnvelope": EllipticEnvelope(), # TODO it crashes on some data: not full-rank, singular cov matrix
        "IsolationForest": IsolationForest()
    }

    clfs.update(baselines)

    for data_fn in fns:
        train_X, test_X, test_y = data_fn()
        test_X_normal = test_X[test_y == -1] # -1 for normal
        test_X_anomaly = test_X[test_y == 1] # 1 for anomaly

        for name in clfs:
            c = clfs[name]

            print(data_fn.__name__)
            print("-----------------")
            print(name)

            c.fit(train_X)

            if hasattr(c, "predict_log_proba"):
                yhat_prob = c.predict_log_proba(test_X)
                d0 = c.predict_log_proba(test_X_normal)
                d1 = c.predict_log_proba(test_X_anomaly)
            else:
                # our baselines have a decision_function, not a prob
                yhat_prob = c.decision_function(test_X)
                d0 = c.decision_function(test_X_normal)
                d1 = c.decision_function(test_X_anomaly)

            # visualise
            plt.hist((d0, d1), bins=30)
            plt.savefig("hist_" + type(data_fn).__name__ + "_" + name + ".png")
            plt.close()

            yhat = c.predict(test_X)
            acc = np.mean(yhat == test_y)

            # thanks Loi!
            yhat_X_normal = c.predict(test_X_normal)
            acc_X_normal = np.mean(yhat_X_normal == -1)
            yhat_X_anomaly = c.predict(test_X_anomaly)
            acc_X_anomaly = np.mean(yhat_X_anomaly == 1)

            # roc_auc_score needs true binary labels (0, 1) and
            # assumes that yhat_prob measures probability of class 1,
            # so we convert our normal (-1) to 1 and anomaly (1) to 0.
            auc_real = roc_auc_score(test_y == -1, yhat_prob)
            FPR, TPR, thresholds = roc_curve(test_y == -1, yhat == -1)
            auc_discrete = auc(FPR, TPR)

            print("acc: %.2f" % acc)
            print("acc X0: %.2f" % acc_X_normal)
            print("acc X1: %.2f" % acc_X_anomaly)
            print("auc_real: %.2f" % auc_real)
            print("auc_discrete: %.2f" % auc_discrete)
            print("")

test()
