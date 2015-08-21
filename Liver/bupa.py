#!/usr/bin/env python

from __future__ import print_function
import numpy as np
import scipy.stats
import pandas as pd
import sklearn.linear_model
import sys

# James McDermott (c) 2015

def do_correlation(bupa):
    # gives almost no correlation: confirms that selector is not related to drinks
    print(np.corrcoef(bupa["x6"], bupa["x7"]))

def read_bupa(dichotomize_x6=None, discard_x7=True):
    # User shold download bupa.dat from
    # https://archive.ics.uci.edu/ml/datasets/Liver+Disorders
    bupa = pd.read_csv("bupa.dat", header=None, names="x1 x2 x3 x4 x5 x6 x7".split(" "))
    if dichotomize_x6 is None:
        pass
    else:
        bupa["x6"] = (bupa["x6"] > dichotomize_x6)
    if discard_x7:
        bupa.drop("x7", axis=1, inplace=True)
    else:
        bupa["x7"] = (bupa["x7"] == 2)
    return bupa

def contingency_table(bupa):
    a = bupa["x6"] == True
    b = bupa["x7"] == True
    print(bupa.groupby([a, b]).count().unstack()["x1"])
    
def get_Xy_train_test(Xy, randomise=True, test_proportion=0.3):
    """Take in a dataframe of numbers and split it into X (all columns up
    to last) and y (last column), then split it into training and
    testing subsets according to test_proportion. Shuffle if
    required."""
    Xy = Xy.values
    if randomise:
        np.random.shuffle(Xy)
    X = Xy[:,:-1].astype('float') # all columns but last
    y = Xy[:,-1].astype('int') # last column
    idx = int((1.0 - test_proportion) * len(y))
    train_X = X[:idx]
    train_y = y[:idx]
    test_X = X[idx:]
    test_y = y[idx:]
    return train_X, train_y, test_X, test_y

def fit_const(train_X, train_y, test_X, test_y):
    """Use the majority class of the y training values as a predictor."""
    mode = scipy.stats.mode(train_y)[0][0] # see help(scipy.stats.mode) for [0][0]
    yhat = np.ones(len(test_y)) * mode
    #print("Predicting constant", mode)
    acc = accuracy(test_y, yhat)
    #print("Accuracy =", acc)
    return acc
    

def fit_lr(train_X, train_y, test_X, test_y):
    """Use logistic regression as a baseline."""
    lr = sklearn.linear_model.LogisticRegression()
    lr.fit(train_X, train_y)
    yhat = lr.predict(test_X)
    #print("Using logistic regression")
    acc = accuracy(test_y, yhat)
    #print("Accuracy =", acc)
    return acc

def accuracy(y, yhat):
    return np.mean(y == yhat)

if __name__ == "__main__":

    reps = 30

    bupa = read_bupa(dichotomize_x6=5, discard_x7=False)
    contingency_table(bupa)
    do_correlation(bupa)
    print()
    
    print("Classifying x6 > 5")
    bupa = read_bupa(dichotomize_x6=5, discard_x7=True)
    results = [], []
    for i in range(reps):
        train_X, train_y, test_X, test_y = get_Xy_train_test(bupa)
        results[0].append(fit_const(train_X, train_y, test_X, test_y))
        results[1].append(fit_lr(train_X, train_y, test_X, test_y))
    print("Predict majority: mu/sigma/min/max")
    print("%.2f %.2f %.2f %.2f" % (np.mean(results[0]), np.std(results[0]), np.min(results[0]), np.max(results[0])))
    print()
    print("Logistic regression: mu/sigma/min/max")
    print("%.2f %.2f %.2f %.2f" % (np.mean(results[1]), np.std(results[1]), np.min(results[1]), np.max(results[1])))
    print()
        
    print("Classifying x7")
    bupa = read_bupa(dichotomize_x6=None, discard_x7=False)
    results = [], []
    for i in range(reps):
        train_X, train_y, test_X, test_y = get_Xy_train_test(bupa)
        results[0].append(fit_const(train_X, train_y, test_X, test_y))
        results[1].append(fit_lr(train_X, train_y, test_X, test_y))
    print("Predict majority: mu/sigma/min/max")
    print("%.2f %.2f %.2f %.2f" % (np.mean(results[0]), np.std(results[0]), np.min(results[0]), np.max(results[0])))
    print()
    print("Logistic regression: mu/sigma/min/max")
    print("%.2f %.2f %.2f %.2f" % (np.mean(results[1]), np.std(results[1]), np.min(results[1]), np.max(results[1])))
    print()

