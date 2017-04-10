#!/usr/bin/env python

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

-Naive Bayes: each feature is modelled with an independent Gaussian,
 and density is the product of each feature's density.

-Multivariate Gaussian: the distribution is modelled by a multivariate
 Gaussian, which uses a mean for each feature plus a covariance
 matrix. See eg Andrew Ng Stanford ML course for the maths.

-Kernel density estimation (KDE): here the full joint distribution is
 modelled using kernel density. This is provided by Scikit-learn. For
 this option, we can choose the kernel and its parameters. The
 Gaussian kernel is the most common. To use "number of neighbours
 within a given distance" we can use the tophat kernel.

-To emulate Cuong To and Elati (GECCO 2013), we can use Negative Mean
 Distance. This has the same effect as KDE with a linear "pseudo
 kernel" calculated as the negative distance between the test point
 and each train point. It is distinct from a true linear kernel,
 because a linear kernel goes to zero (non-linearly) outside the
 bandwidth. We use negative distance to preserve the sense (a kernel
 is a similarity, whereas a distance is a dissimilarity).

"""

from __future__ import print_function

import numpy as np
np.seterr(all='raise')
import scipy.spatial
import scipy.stats
from sklearn import preprocessing
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt

class SingleGaussianDensity:
    """A helper class which behaves like KDE, but models density as a
    Gaussian over the distance from the centroid. To be useful, the
    user needs to standardise features to have equal variance."""

    def __init__(self, metric="euclidean"):
        self.mu = None
        self.metric = metric

    def fit(self, X):
        self.mu = np.mean(X, axis=0)
        self.mu.shape = (1, len(self.mu))
        dists = scipy.spatial.distance.cdist(X, self.mu, metric=self.metric)
        self.sd = np.max(dists)

    def score_samples(self, X):
        # distance from the mean
        dists = scipy.spatial.distance.cdist(X, self.mu, metric=self.metric)
        dists.shape = (dists.shape[0],)
        # a pt at mu will have zero distance, hence high density
        # we use a large sd so that pts at high distance won't cause an underflow
        return np.log(scipy.stats.norm.pdf(dists, loc=0.0, scale=self.sd))

class NaiveBayesDensity:
    """A helper class which behaves like KDE, but models density as a
    product of independent Gaussians."""

    def fit(self, X):
        self.mu = np.mean(X, axis=0)
        self.sigmasq = np.std(X, axis=0)

    def score_samples(self, X):
        return np.log(np.product(
            list(scipy.stats.norm.pdf(xi, loc=mui, scale=sigmasqi)
                 for xi, mui, sigmasqi in
                 zip(X.T, self.mu, self.sigmasq)), axis=0))

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
        return np.log(result)

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

    The `dens` parameter is a density object, such as the
    NaiveBayesDensity above, or the KernelDensity from
    sklearn.

    The `threshold` parameter sets the proportion of the (normal)
    training data which should be classified as normal.

    The `scaler` parameter gives a scaler object. Can pass `None`
    and then no scaling is performed.
    """

    def __init__(self,
                 threshold=0.95,
                 dens=None,
                 scaler=preprocessing.StandardScaler):

        self.threshold = threshold
        if scaler:
            self.scaler = scaler()
        else:
            self.scaler = None
        if dens:
            self.dens = dens
        else:
            self.dens = NaiveBayesDensity()

    def fit(self, X):
        # scale
        if self.scaler:
            self.scaler.fit(X)
            X = self.scaler.transform(X)

        # fit
        self.dens.fit(X)

        # transform relative threshold (eg 95%) to absolute
        dens = self.dens.score_samples(X)
        self.abs_threshold = np.percentile(dens, 100 * (1 - self.threshold))

    def score_samples(self, X):
        if self.scaler:
            X = self.scaler.transform(X)

        # the score is in log-probability (for density approaches), or
        # in negative distance (for NegativeMeanDistance)
        return self.dens.score_samples(X)

    def predict(self, X):
        dens = self.score_samples(X)
        return dens < self.abs_threshold
