#!/usr/bin/env python

import numpy as np
from scipy.stats import chi2
import matplotlib.pyplot as plt

def norm_of_each_row(x):
    # Euclidean norm of each row in a 2D array
    return np.array([np.linalg.norm(xi) for xi in x])

def uniform_in_hypersphere(d, n):
    """Generate n points uniformly distributed in a hypersphere of
    dimension d, centre 0, radius 1. The method is from
    http://math.stackexchange.com/q/87238. It generates points on the
    surface, then gives them a radius which is U**1/d where U is
    uniform on [0, 1]."""

    U = np.random.uniform(0, 1, (n,))
    X = np.random.normal(0, 1, (n, d))
    Rs = 1.0
    # The key equation, written out in element form, is:
    # p = np.array([Rs * Ui**(1.0/d) * Xi / np.linalg.norm(Xi) for
    #               Ui, Xi in zip(U, X)])
    # 
    # For speed we vectorise, so it is:
    p = (Rs * (U.reshape((-1, 1)) ** (1.0/d)) * X) / norm_of_each_row(X).reshape((-1, 1))
    return p

def uniform_in_hypersphere_tax_duin(d, n):
    """Generate n points uniformly distributed in a hypersphere of
    dimension d, centre 0, radius 1. The method is from Tax and Duin
    ("Uniform Object Generation for Optimizing One-class Classifiers",
    JMLR 2001, pp 155--173). It only draws (d x n) normal values,
    whereas the other method also requires an extra n uniform
    values. It achieves this by taking the d vectors and changing
    their radius using a chi^2 distribution.
    
    However, I have changed a power of 2/d to
    1/d -- I think the 2/d is incorrect, possibly a typo caused by
    bringing another power of 2 inside the brackets."""

    # x is a 2D array, shape (n, d)
    x = np.random.normal(0, 1, (n, d))

    # r^2 is a 1D array, shape (n,)
    r = norm_of_each_row(x)
    r_sq = r**2

    # chi2.cdf is a function: it maps r^2 to the range [0, 1]
    # ro^2 is a 1D array, shape (n,)
    ro_sq = chi2.cdf(r_sq, d)
    assert (0.0 <= ro_sq).all() and (ro_sq <= 1.0).all()

    # r' is a 1D array, shape (n,)
    # Note Tax and Duin have 2/d here, not 1/d. I think 1/d is correct.
    # Compare the 1/d in the other method.
    r_prime = ro_sq ** (1.0 / d) 
    assert len(r_prime) == len(x) == len(r_sq)

    # x' is a 2D array, shape (n, d).

    # For reading: the final equation, in element form, is:
    # x_prime = np.array([r_prime_i * xi / r_sq_i for (r_prime_i, xi, r_sq_i) in
    #                     zip(r_prime, x, r_sq)])
    #
    # For speed, we vectorise. Note reshape((-1, 1)) means reshape so
    # that the second dimension is 1 and the first dimension is the
    # original length.
    x_prime = r_prime.reshape((-1, 1)) * x / r.reshape((-1, 1))

    return x_prime

x = uniform_in_hypersphere(2, 2000)
#x = uniform_in_hypersphere_tax_duin(2, 2000)
print x

plt.figure(figsize=(6, 6))
plt.plot(x[:,0], x[:,1], ".")
plt.axis('equal')
plt.xlim((-1, 1))
plt.ylim((-1, 1))
plt.savefig("uniform_in_hypersphere.png")

