# ML-snippets
Snippets of machine learning code for educational porpoises

When I write or find a very short piece of code for some machine
learning task that looks simple enough to be educational I put it in
here and refer my students to it. Since it includes code written by
myself and code written by others, the license varies on a per-file
basis.

Contents:

* DBOCC: density-based one-class classification

One-class classification is also known as anomaly detection. It's
about training a classifier when you only have one set of data -- no
labels. We can do it be learning the density (or similar concepts) of
the data, and then applying a threshold -- new points are classed as
anomalies if they lie in regions where the training data is of low
density. This code implements three related approaches to OCC, and is
built in Python using Numpy, Scipy, and Scikit-learn components.

(Written by me.)
