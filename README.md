# ML-snippets
Snippets of machine learning code for educational porpoises

When I write or find a very short piece of code for some machine
learning task that looks simple enough to be educational I put it in
here and refer my students to it.

Hello!

Contents:

* [DBOCC](DBOCC): density-based one-class classification.

  One-class classification is also known as anomaly detection. It's
  about training a classifier when you only have one set of data -- no
  labels. We can do it be learning the density (or similar concepts)
  of the data, and then applying a threshold -- new points are classed
  as anomalies if they lie in regions where the training data is of
  low density. This code implements several related approaches to OCC,
  and is built in Python using Numpy, Scipy, and Scikit-learn
  components.

* [Liver](Liver): simple contingency table, predict majority, and
  logistic regression on the well-known BUPA Liver Disorders dataset.
  There has been a widespread misconception that the final variable
  indicates presence or absence of a liver disorder in the
  subjects. In fact, the final variable is just a train/test
  selector. Richard Forsyth and I wrote an article discussing the
  issue. This (very simple) code is part of that.

* [Uniform In Hypersphere](Uniform_in_Hypersphere): generate vectors
  uniformly distributed in a hypersphere of given dimension. Two
  methods are supplied, which are similar but not identical. One, from
  Tax and Duin 2001, corrects what I think is an error in the
  original.

* [RecSys](RecSys): a collaborative filtering recommender system,
  using singular value decomposition by manual gradient descent. Based
  on Funk's and Paterek's work towards the Netflix prize.
  
* [SVM](SVM): I think a lot of SVM tutorials give all the details
  about the maximum margin separating hyperplane, the quadratic
  programming and support vectors, and radial kernels, but don't give
  a good intuition on one important part of the big picture. When the
  kernel does its implicit mapping from the original feature space to
  a new feature space, what does that new feature space look like?
  What do the features in that new space mean? This short notebook
  tries to fill in the missing link.

* [Representations for AI](Representations_for_AI): This
  notebook/presentation is about learning representations (embeddings)
  for data using classic machine learning methods, and also using a
  modern method (a Siamese neural network with triplet loss). It is
  part of the Atlantec 2019 AI Tools and Techniques session.
  

* [Kernel Regression Association](Kernel_Regression_Association): This notebook
  presents a novel approach to measuring the strength of association between
  two real variables. Correlation measures strength of linear relationship,
  while rank-based correlation such as Kendall's tau measures strength of 
  non-linear monotonic relationship. Mutual information measures strength of
  non-linear, non-monotonic relationship, but depends on an arbitrary binning
  of the data into discrete bins. Kernel Regression Association aims to
  measure the strength of non-linear, non-monotonic relationship without
  binning, relying on the *training error* of kernel regression of one
  variable against the other (and vice versa).

