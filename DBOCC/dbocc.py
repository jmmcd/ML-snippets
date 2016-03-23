#!/usr/bin/env python
from __future__ import print_function

# Copyright 2015-2016 James McDermott <jmmcd@jmmcd.net>

# Hereby licensed under the GPL v2 or any later version.


"""Density-based one-class classification, built using scikit-learn,
numpy and scipy components. This is mostly for educational porpoises.

One-class classification (OCC) is about training on a single class,
the "normal" class, usually because that's all that's available, but
then using the trained model to classify new data as either normal or
anomaly.

We provide density-based approaches to OCC. The idea is: we model the
density of the normal training data, and if a test point appears in a
low-density area of the space (lower than some threshold), then we
flag it as an anomaly.

There are several approaches to modelling density:

-Single Gaussian: uses a single Gaussian (with mean and
 variance). This is equivalent to just calculating the *distance* from
 the test point to the mean (centroid) of the training data.

-Independent Gaussians: each feature is modelled with a Gaussian, and
 density is the product of each feature's density.

-Multivariate Gaussian: the distribution is modelled by a multivariate
 Gaussian, which uses a mean for each feature plus a covariance
 matrix. See eg Andrew Ng Stanford ML course for the maths.

-Kernel density estimation (KDE): here the full joint distribution is
 modelled using kernel density.

Within KDE, we can choose the kernel and its parameters. To emulate
Cuong To and Elati (GECCO 2013), we can use a linear "pseudo kernel",
which is just the negative distance.

"""

from sklearn import preprocessing
from sklearn.neighbors import KernelDensity
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
import numpy as np
np.seterr(all='raise')
import scipy.spatial
import scipy.io

class SingleGaussianDensity:
    """A helper class which behaves like KDE, but models density as a
    Gaussian over the distance from the centroid. To be useful, the
    user needs to standardise features to have equal variance."""
    
    def __init__(self, metric="euclidean"):
        self.metric = metric
        
    def fit(self, X):
        self.mu = np.mean(X, axis=0)
        self.mu.shape = (1, len(self.mu))
        
    def score_samples(self, X):
        # distance from the mean
        dists = scipy.spatial.distance.cdist(X, self.mu, metric=self.metric)
        dists.shape = (dists.shape[0],)
        # a pt at mu will have zero distance, hence high density
        return scipy.stats.norm.pdf(dists, loc=0.0, scale=1.0)

class IndependentGaussiansDensity:
    """A helper class which behaves like KDE, but models density as a
    product of independent Gaussians."""
    
    def fit(self, X):
        self.mu = np.mean(X, axis=0)
        self.sigmasq = np.std(X, axis=0)
        
    def score_samples(self, X):
        return np.product(
            list(scipy.stats.norm.pdf(xi, loc=mui, scale=sigmasqi)
                 for xi, mui, sigmasqi in
                 zip(X.T, self.mu, self.sigmasq)), axis=0)
        
class MultivariateGaussianDensity:
    """A helper class which behaves like KDE, but models density with a
    single multivariate Gaussian."""
    
    def fit(self, X):
        self.mu = np.mean(X, axis=0)
        self.Sigma = np.cov(X, rowvar=0)
        
    def score_samples(self, X):
        result = scipy.stats.multivariate_normal.pdf(X, mean=self.mu, cov=self.Sigma)
        if X.shape[0] == 1:
            # multivariate_normal.pdf seems to squeeze, so we unsqueeze
            result = np.array([result])
        return result
    
class NegativeMeanDistance:
    """A helper class which behaves like KDE, but models "density" as
    negative mean distance. Distance behaves slightly differently to a
    kernel: a so-called linear kernel is only linear within the
    bandwidth, but goes non-linearly to zero outside. We use negative
    distance to preserve the sense, ie lower numbers are more
    anomalous, because a kernel is a similarity while a distance is a
    dissimilarity."""
    
    def __init__(self, metric="euclidean"):
        self.metric = metric
        
    def fit(self, X):
        self.X = X
        
    def score_samples(self, X):
        dists = scipy.spatial.distance.cdist(X, self.X, metric=self.metric)
        return -np.mean(dists, axis=1)


class DensityBasedOneClassClassifier:
    """A classifier for one-class classification based on estimating
    the density of the training set.

    Approaches to modelling density: single_gaussian,
    independent_gaussians, multivariate_gaussian, kernel.

    For kernel density, can also pass the kernel and bandwidth
    parameters.

    To use the negative mean distance approach, pass
    `kernel="linear_pseudo_kernel"`. Otherwise, `kernel` is the name
    of a kernel -- "gaussian", "linear", "tophat", "epanechnikov",
    "exponential", or "cosine", as accepted by
    KernelDensity. `bandwidth` is a parameter to that kernel. If you
    use "tophat" the effect is that density is defined as the number
    of points within a given radius.

    `metric` is the name of a metric as accepted by KernelDensity or
    by scipy.spatial.cdist, eg "euclidean".

    The `threshold` parameter sets the proportion of the (normal)
    training data which should be classified as normal.
    """

    def __init__(self,
                 threshold=0.95,
                 dens=None):

        self.threshold = threshold
        self.scaler = preprocessing.StandardScaler()
        if dens:
            self.dens = dens
        else:
            self.dens = IndependentGaussiansDensity()

    def fit(self, X):
        # scale
        self.scaler.fit(X)
        X = self.scaler.transform(X)
            
        # fit
        self.dens.fit(X)

        # transform relative threshold (eg 95%) to absolute
        dens = self.dens.score_samples(X)
        self.abs_threshold = np.percentile(dens, 100 * (1 - self.threshold))

    def score_samples(self, X):
        X = self.scaler.transform(X)
        # the score is in negative log-probability (for KDE), or in
        # density (for other density approaches), or in negative
        # distance (for NegativeMeanDistance)
        return self.dens.score_samples(X)

    def predict(self, X):
        dens = self.score_samples(X)
        return dens < self.abs_threshold


def toy_data():
    # normal training set
    X = np.array([
        [ 1.0, 1.0],
        [ 1.1, 1.1],
        [ 1.0, 1.2],
        [ 1.1, 1.3],
        [ 1.0, 1.1],
        [ 1.4, 1.1],
        [ 1.1, 1.2]
    ])

    # test set with some normal, some anomaly
    X_test = np.array([
        [ 1.1, 1.1],
        [ 1.0, 1.2],
        [ 1.1, 1.3],
        [ 1.0, 1.1],
        [ 1.1, 1.35],
        [ 1.6, 0.9]
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
    dX = d[:,1:-1] # discard the first column (ids) and the last (labels)
    dy = d[:,-1]
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
    
    for data_fn in fns:
        print(data_fn.__name__)
        print("-----------------")
        train_X, test_X, test_y = data_fn()
        test_X0 = test_X[~test_y]
        test_X1 = test_X[test_y]

        denss = [
            SingleGaussianDensity,
            IndependentGaussiansDensity,
            MultivariateGaussianDensity,
            KernelDensity,
            NegativeMeanDistance
        ]
        
        for dens in denss:
            c = DensityBasedOneClassClassifier(dens=dens())
            print(dens.__name__)
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
