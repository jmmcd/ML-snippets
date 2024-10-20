import numpy as np
from scipy.spatial.distance import pdist, euclidean
from scipy.stats import pearsonr, kendalltau

def metric_distortion(X, Y, epsilon=10**-8, sample=None):
    """Suppose we have metric spaces X and Y and a mapping f: X -> Y.

    The metric distortion is a measure of how well f preserves
    structure.

    This function assumes that X and Y are datasets from those spaces,
    not necessarily the entire spaces. This is useful in practice.

    There are multiple possible definitions of metric distortion,
    nearly all [1] based on the ratio between d_X(x_i, x_j) and
    d_Y(y_i, y_j) for all pairs i, j, where y_i = f(x_i).

    The central definition of metric distortion [2]:

    A mapping f: X -> Y is an **embedding with distortion alpha**
    (alpha >= 1) if there exists a constant r > 0 st forall i, j:

    r d_X(x_i, x_j) <= d_Y(y_i, y_j) <= alpha r d_X(x_i, x_j).

    We take the minimum alpha which satisfies these inequalities.

    A value of alpha = 1 is a perfect preservation of structure and
    larger values are worse. Smaller values cannot occur.

    Notice this is symmetric: alpha(X, Y) = alpha(Y, X).

    The r is there to account for scaling, eg if all distances in Y
    are just 1000x the distances in X, it is not really a
    distortion. We would have r = 1000 and alpha = 1.

    To calculate r we take min(d_Y / d_X), satisfying the left-hand
    inequality. Then take alpha = max(d_Y / d_X) / r, satisfying the
    right-hand inequality.

    By default we use the distances between all pairs of points in X
    and Y. However if n is large, we can pass eg sample=0.1 or sample
    = 0.01 for a representative sub-sample. Note this is a sample of
    distances, not the distances for a sample of X and Y.

    The definition of metric distortion tends to emphasise the
    worst-case. Another option is to calculate the **correlation**
    between d_X and d_Y. R = 1 means perfect preservation of
    structure. Since the hard work is all done already, calculate R as
    well. But R is linear correlation. Maybe rank correlation is better?
    So calculate Kendall's tau also. And for both, we get the p-value as well.
    
    Thus, we return alpha, r, R, R_pvalue, tau, tau_pvalue (in that order).

    Our final test below seems to show that Pearson correlation is very robust
    to our sampling method, but the strict definition of metric distortion
    (alpha) is not.

    Measuring metric distortion is a very useful tool in machine
    learning when dealing with embeddings, eg Vankadara and von
    Luxburg [1] discuss several related measures of distortion.

    Metric distortion can also be useful for the case where we start
    with some other space Z or dataset sampled from Z, and X = f_X(Z)
    and Y = f_Y(Z). For example, if Z is a dataset of (image, caption)
    pairs, then we could have f_X = LLM embeddings from the caption,
    and f_Y = ResNet embeddings on the image. Then this function would
    tell us how closely related the LLM and ResNet embeddings are. If
    we assume the LLM is "good", then metric distortion is a measure
    of good the ResNet embeddings are. My diagram explaining that is at [3].
    
    [1] "Measures of distortion for machine learning", 32nd Conference
    on Neural Information Processing Systems (NeurIPS 2018), Montréal,
    Canada. https://proceedings.neurips.cc/paper_files/paper/2018/file/4c5bcfec8584af0d967f1ab10179ca4b-Paper.pdf

    [2] https://web.stanford.edu/class/cs369m/cs369mlecture1.pdf

    [3] https://www.overleaf.com/project/6511b16bfda63f200b20217b

    """
    d_X, d_Y = get_distances(X, Y, sample)

    # first calculate alpha (and r):
    
    # we have to avoid nan and inf, which could arise if any distance
    # is zero.
    nonzero = (d_X > epsilon) & (d_Y > epsilon)
    d_X_nz = d_X[nonzero]
    d_Y_nz = d_Y[nonzero]
    ratios = d_Y_nz / d_X_nz
    r = np.min(ratios)
    alpha = np.max(ratios) / r

    # from here on we revert to the original d_X and d_Y

    # next calculate R and p-value. 
    res = pearsonr(d_X, d_Y)
    R = res.statistic
    R_pvalue = res.pvalue

    # and calculate Kendall's tau and p-value
    res = kendalltau(d_X, d_Y)
    tau = res.statistic
    tau_pvalue = res.pvalue

    return alpha, r, R, R_pvalue, tau, tau_pvalue



def get_distances(X, Y, sample=None):
    # helper function to get paired distance values in X and in Y
    # either *all of them* if X and Y are small, else a sample
    # of pairs.
    if sample is None: # calculate all pairwise distances
        d_X = pdist(X)
        d_Y = pdist(Y)
        
    else: # only a sample of pairwise distances
        n = Y.shape[0]
        n_dists = n * (n - 1) / 2
        sample_size = int(n_dists * sample)
        d_X = []
        d_Y = []
        cache = set()
        while len(d_X) < sample_size:
            i, j = np.random.choice(n, 2, replace=False)

            # avoid doing same pair twice
            i, j = sorted((i, j))
            if (i, j) in cache: continue
            cache.add((i, j))
            
            dx = euclidean(X[i], X[j])
            dy = euclidean(Y[i], Y[j])
            d_X.append(dx)
            d_Y.append(dy)
        d_X = np.array(d_X)
        d_Y = np.array(d_Y)
    return d_X, d_Y   

def metric_distortion_test_report(X, Y, sample, plot_filename=None, print_dataset=False):
    if print_dataset:
        print(X)
        print(Y)

    print("no sampling")
    alpha, r, R, R_pvalue, tau, tau_pvalue = metric_distortion(X, Y)
    print(f"alpha: {alpha}, r: {r}, R: {R}, R p-value: {R_pvalue}, tau: {tau}, tau_pvalue: {tau_pvalue}")
    print("")
    print("sampling")
    alpha, r, R, R_pvalue, tau, tau_pvalue = metric_distortion(X, Y, sample=sample)
    print(f"alpha: {alpha}, r: {r}, R: {R}, R p-value: {R_pvalue}, tau: {tau}, tau_pvalue: {tau_pvalue}")
    print("")
    if plot_filename is not None:
        plot_distances(*get_distances(X, Y), alpha, r, R, tau, plot_filename)


    
def plot_distances(dx, dy, alpha, r, R, tau, filename):
    import matplotlib.pyplot as plt
    plt.scatter(dx, dy)
    plt.xlabel(r"$d_X$")
    plt.ylabel(r"$d_Y$")
    plt.title(f'alpha {alpha:.2f} r {r:.2f} R {R:.2f} tau {tau:.2f}')
    plt.savefig(filename)
    plt.close()

def test():
    # several tiny datasets: compare all against all
    # 6 points, 3 dimensions, but 3rd dim is 0
    X1 = np.array([[0, 0, 0], [0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0.5, 0.5, 0]])
    # same points, but discard 3rd dimension
    X2 = X1[:, :2]
    # same points scaled
    X3 = 1000 * X2
    # totally different points, in 1D
    X4 = np.array([[10, 9, 8, 7, 6, 5]]).T
    for i, x1 in enumerate((X1, X2, X3, X4)):
        for j, x2 in enumerate((X1, X2, X3, X4)):
            metric_distortion_test_report(x1, x2, sample=0.5, plot_filename=f"test_{i}_{j}.png", print_dataset=True)

    # another test
    n = 1000
    dx = 5
    dy = 3
    sigma = 0.01
    print(f"sampling X as {n} points in {dx} dimensions, then making Y by throwing away {dx - dy} dimensions and adding {sigma} noise")
    X = np.random.random((n, dx))
    Y = X[:, :dy] + sigma * np.random.random((n, dy))
    metric_distortion_test_report(X, Y, sample=0.01, plot_filename="second_test.png", print_dataset=False)

if __name__ == '__main__':
    test()