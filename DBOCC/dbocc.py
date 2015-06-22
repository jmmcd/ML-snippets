#!/usr/bin/env python

# Copyright 2015 James McDermott <jmmcd@jmmcd.net>

# Hereby licensed under the GPL v2 or any later version.


"""
Density-based one-class classification, built using scikit-learn,
numpy and scipy components. This is mostly for educational porpoises.

One-class classification (OCC) is about training on a single class,
the "normal" class, usually because that's all that's available, but
then using the trained model to classify new data as either normal or
anomaly.

We provide three approaches to OCC:

Kernel density estimation: a new point is classified as an anomaly if
it is in a low-density region.

Mean distance: a new point is classified as an anomaly if its mean
distance to the training set is large. Where a kernel is a measure of
similarity between points, a distance is a measure of dissimilarity.

Centroid: a new point is classified as an anomaly if its distance to
the centroid of the training set is large.

"""

from sklearn import preprocessing
from sklearn.neighbors import KernelDensity
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
import numpy as np
np.seterr(all='raise')
import scipy.spatial


class CentroidBasedOneClassClassifier:
    """A simple classifier for one-class classification: standardise
    the data so that its mean is all zeros, then establish a threshold
    so as to contain (eg) 95% of the training data. Then new points at
    a distance larger than the threshold are classified as
    anomalies. One of the motivations is that it will be extremely
    fast both at training and classification time."""
    
    def __init__(self, threshold=0.95, metric="euclidean"):
        self.threshold = threshold
        self.scaler = preprocessing.StandardScaler()
        self.metric = metric

    def fit(self, X):
        # scale
        self.scaler.fit(X)
        X = self.scaler.transform(X)

        # because we are using StandardScaler, the centroid is a
        # vector of zeros, but we save it in shape (1, n) to allow
        # cdist to work happily later.
        self.centroid = np.zeros((1, X.shape[1]))

        # transform relative threshold (eg 95%) to absolute
        dists = self.get_density(X, scale=False) # no need to scale again
        self.abs_threshold = np.percentile(dists, 100 * self.threshold)

    def get_density(self, X, scale=True):
        if scale:
            X = self.scaler.transform(X)
        dists = scipy.spatial.distance.cdist(X, self.centroid, metric=self.metric)
        dists = np.mean(dists, axis=1)
        return dists

    def classify(self, X):
        dists = self.get_density(X)
        return dists > self.abs_threshold


class NegativeMeanDistance:
    """A small helper class which emulates the behaviour of KDE, but
    using negative mean distance. We use distance because it can
    behave slightly differently to a kernel: in particular, note that
    a so-called linear kernel is only linear within the bandwidth, but
    goes non-linearly to zero outside. We use negative distance to
    preserve the sense, ie lower numbers are more anomalous, because a
    kernel is a similarity while a distance is a dissimilarity."""
    
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

    There are two approaches as described above, kernel density
    estimation (using the `kernel` and `bandwidth` parameters), and
    distance (negative mean distance) against the training set.

    To use the distance approach, pass
    `kernel="really_linear"`. Otherwise, `kernel` is the name of a
    kernel -- "gaussian", "linear", "tophat", "epanechnikov",
    "exponential", or "cosine", as accepted by
    KernelDensity. `bandwidth` is a parameter to that kernel. If you
    use "tophat" the effect is that density is defined as the number
    of points within a given radius.

    `metric` is the name of a metric as accepted by KernelDensity or
    by scipy.spatial.cdist, eg "euclidean".

    The `threshold` parameter sets the proportion of the (normal)
    training data which should be classified as normal. The idea is
    that in OCC it's better to have a few false positives than any
    false negatives.

    A big drawback of kernel density estimation is that if you have a
    lot of training data, you have to store it all and calculate
    against it all during classification -- not just during training
    time. This is also true of the "distance" approach. The centroid
    approach above avoids this. Another possibility is to downsample
    -- fit a KDE with the training data, then draw a smaller number of
    samples from that, and use that as your training data. This will
    save time at classification time. However, I haven't tested
    whether it really works or not.  `should_downsample` and
    `downsample_count` are the parameters for it.

    """

    def __init__(self, threshold=0.95, kernel="gaussian", bandwidth=1.0,
                 metric="euclidean",
                 should_downsample=False, downsample_count=1000):

        self.should_downsample = should_downsample
        self.downsample_count = downsample_count
        self.threshold = threshold
        self.scaler = preprocessing.StandardScaler()
        if kernel == "really_linear":
            self.dens = NegativeMeanDistance(metric=metric)
        else:
            self.dens = KernelDensity(bandwidth=bandwidth, kernel=kernel, metric=metric)

    def fit(self, X):
        # scale
        self.scaler.fit(X)
        self.X = self.scaler.transform(X)

        # downsample?
        if self.should_downsample:
            self.X = self.downsample(self.X, self.downsample_count)

        # fit
        self.dens.fit(self.X)

        # transform relative threshold (eg 95%) to absolute
        dens = self.get_density(self.X, scale=False) # no need to scale again
        self.abs_threshold = np.percentile(dens, 100 * (1 - self.threshold))

    def get_density(self, X, scale=True):
        if scale:
            X = self.scaler.transform(X)
        # in negative log-prob (for KDE), in negative distance (for NegativeMeanDistance)
        return self.dens.score_samples(X)

    def classify(self, X):
        dens = self.get_density(X)
        return dens < self.abs_threshold # in both KDE and NMD, lower values are more anomalous

    def downsample(self, X, n):
        # we've already fit()ted, but we're worried that our X is so
        # large our classifier will be too slow in practice. we can
        # downsample by running a kde on X (this will be slow, but
        # happens only once), sampling from it, and then using those
        # points as the new X.
        if len(X) < n:
            return X
        kde = KernelDensity()
        kde.fit(X)
        return kde.sample(n)


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

def test():
    #train_X, test_X, test_y = toy_data()
    train_X, test_X, test_y = process_wbcd()
    test_X0 = test_X[~test_y]
    test_X1 = test_X[test_y]

    cs = {
        "density": DensityBasedOneClassClassifier(bandwidth=2, kernel="gaussian", metric="euclidean"),
        "distance": DensityBasedOneClassClassifier(kernel="really_linear"),
        "centroid": CentroidBasedOneClassClassifier()
    }

    for k in cs:
        print k
        c = cs[k]
        c.fit(train_X)
        d0 = c.get_density(test_X0)
        d1 = c.get_density(test_X1)
        plt.hist((d0, d1), bins=30)
        plt.savefig("hist_" + k + ".png")
        plt.close()

        yhat = c.classify(test_X)
        acc = np.mean(yhat == test_y)

        # thanks Loi!
        yhat_X0 = c.classify(test_X0)
        acc_X0 = np.mean(yhat_X0 == False)
        yhat_X1 = c.classify(test_X1)
        acc_X1 = np.mean(yhat_X1 == True)

        print "acc: %.2f" % acc
        print "acc X0: %.2f" % acc_X0
        print "acc X1: %.2f" % acc_X1
        print 

test()
