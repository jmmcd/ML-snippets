#!/usr/bin/env python

# Copyright 2015-2017 James McDermott <jmmcd@jmmcd.net>
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

-Single Gaussian: uses a single Gaussian (with mean and scalar
 variance). This is equivalent to just calculating the *distance* from
 the test point to the mean (centroid) of the training data. If the
 features are standardised and decorrelated, this is equivalent to
 Mahalanobis distance.

-Naive Bayes: each feature is modelled with an independent Gaussian,
 and density is the product of each feature's density. This is
 equivalent to having a diagonal covariance matrix.

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
 is a similarity, whereas a distance is a dissimilarity). The mean
 distance is distinct from the distance to the centroid (as in Single
 Gaussian above).

Scikit-learn uses some different conventions for classifiers.

Binary classifiers: for SVM, it uses clf.decision_function() to get
real values, and then clf.predict() to get integer class labels.

But for eg Naive Bayes, it uses clf.predict_proba (or
predict_log_proba) to get real values, and then clf.predict().

For one-class classifiers (OneClassSVM, EllipticEnvelope,
IsolationForest), it uses clf.decision_function() to get real values,
and then clf.predict() to get class labels, always +1 for normal and
-1 for anomalies.

In our density-based OCC situation, the real values are probabilities
except in NegativeMeanDistance, but we will regard these as
pseudo-probabilities, so we use predict_log_proba. We return +1 for
normal and -1 for anomalies.

"""

from __future__ import print_function

import numpy as np
np.seterr(all='raise')
import scipy.spatial
import scipy.stats
from sklearn import preprocessing
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

class SingleGaussianDensity(GaussianMixture):
    """A helper class which behaves like KDE, but models density as a
    single spherical Gaussian: that is equivalent to putting a
    threshold on distance from the centroid. To be useful, the user
    needs to standardise features to have equal variance."""
    def __init__(self):
        super(SingleGaussianDensity, self).__init__(
            n_components=1,
            covariance_type='spherical'
        )

class NaiveBayesDensity(GaussianMixture):
    """A helper class which behaves like KDE, but models density as a
    product of independent Gaussians, ie a diagonal covariance
    matrix."""
    def __init__(self):
        super(NaiveBayesDensity, self).__init__(
            n_components=1,
            covariance_type='diag'
        )

class MultivariateGaussianDensity(GaussianMixture):
    """A helper class which behaves like KDE, but models density with a
    single multivariate Gaussian, ie a full covariance matrix."""
    def __init__(self):
        super(MultivariateGaussianDensity, self).__init__(
            n_components=1,
            covariance_type='full'
        )

class NegativeMeanDistance:
    """A helper class which behaves like KDE, but models "density" as
    negative mean distance. Distance behaves slightly differently to a
    kernel: a so-called linear kernel is only linear within the
    bandwidth, but goes non-linearly to zero outside. We use negative
    distance to preserve the sense, ie lower numbers are more
    anomalous, because a kernel is a similarity while a distance is a
    dissimilarity. We also allow user to set nneighbours, so we take NMD
    of these nearest neighbours only. This can help avoid an exaggerated
    effect of outliers. TODO: the threshold calculated with nneighbours
    in use may be wrong, because every point has a 0 as the first
    distance (distance to itself)."""

    def __init__(self, nneighbours=None, metric="euclidean"):
        self.metric = metric
        self.nneighbours = nneighbours

    def fit(self, X):
        self.X = X
        if self.nneighbours is not None:
            if self.nneighbours < 1 or self.nneighbours > len(self.X):
                raise ValueError("Invalid value for nneighbours")

    def score(self, X, y=None):
        # copied directly from sklearn kde.py
        return np.sum(self.score_samples(X))

    def score_samples(self, X):
        dists = scipy.spatial.distance.cdist(X, self.X, metric=self.metric)
        if self.nneighbours:
            # take the NMD of the nearest neighbours only, need to sort
            dists.sort()
            return -np.mean(dists[:,:self.nneighbours], axis=1)
        else:
            # take the NMD of all points in training set
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
        # eg 0.95 -> 0.05 -> 5 -> -467.3
        self.abs_threshold = np.percentile(dens, 100 * (1 - self.threshold))

    def predict_log_proba(self, X):
        if self.scaler:
            X = self.scaler.transform(X)

        # the score is in log-probability (for density approaches), or
        # in negative distance (for NegativeMeanDistance)
        return self.dens.score_samples(X)

    def predict(self, X):
        dens = self.predict_log_proba(X)
        # 1 for normal, -1 for anomaly
        return np.where(dens < self.abs_threshold, -1, 1)
