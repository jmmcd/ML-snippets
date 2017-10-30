import os
import datetime
import sys
import numpy as np

"""

Recommender system based on latent factors, to be uncovered by
singular value decomposition (SVD), with training by manual gradient
descent (GD).

James McDermott <jamesmichaelmcdermott@gmail.com>

Based mostly on Funk, Netflix Update: Try This at Home,
http://sifter.org/~simon/journal/20061211.html and on Paterek,
Improving regularized singular value decomposition for collaborative
filtering.

Several other recommenders out there keep things simpler by using a
library for SVD, but my goal is to teach (and learn about) the
gradient descent.

TODO

Clip predictions to the range [1, 5].

We have K, an offset for the whole datset, and C, an offset per user
(some users are more generous than others) and D, an offset per item
(some movies are better than others). Funk initialises them and then
GDs them. I have implemented this but at the moment it's causing ML to
diverge after a few iterations. On the fake data it actually helps, a
little. The alternative is to initialise them and then leave them
alone.

Also, Funk recommends to calculate their (initial) values as
posteriors after observing the data, rather than as straight-up
averages.

Implement a train-test split? Check
https://github.com/ahaldar/Movie-Recommender/blob/master/movie-reco.py
for this and for cosine similarity.

Momentum.

My implementation GDs all parameters at once, rather than working on
"one factor at a time" as suggested. I've tried the alternative
several times but not systematically, finding no improvement. Should
try this test properly.

Initialisation of U and V seems to matter. Funk uses a constant 0.1
for all. I tried a light Gaussian noise and it's far better ON THE
FAKE DATASET -- almost always. Need to test whether it helps on the ML
dataset. The disadvantage is it makes things stochastic.

"""


# reading/processing input data
###############################

def make_tiny_test_data():
    # I usually start with a hand-written dataset of this small size
    # to make things concrete for myself and make sure my code will be
    # dimensionally right.
    A = np.array([
        # user, item, rating
        [0, 0, 3],
        [1, 0, 4],
        [0, 1, 2],
        [1, 1, 3],
        [0, 2, 5],
        [1, 2, 4],
        [0, 3, 3]
        ])
    n = 2 # users
    m = 4 # items
    return A, n, m

def read_fake_ratings():
    # fake ratings generated with an assumption of 2 hidden factors,
    # see make_fake_ratings.py
    filename = "fake_ratings.csv"
    A = np.genfromtxt(filename, delimiter=",", skip_header=1)
    return make_consecutive(A)

def read_movielens():
    # A well-known test dataset of ratings from real users.  It comes
    # from:
    # http://files.grouplens.org/datasets/movielens/ml-latest-small-README.html
    # Thanks due to GroupLens research group, U of Minnesota
    filename = "movielens/ratings.csv"
    A = np.genfromtxt(filename, delimiter=",", skip_header=1)
    A = A[:, :3]
    return make_consecutive(A)

def make_consecutive(A):
    # We need consecutive user-ids and movie-ids because we're going
    # to refer to indices in matrices. The raw data mightn't have
    # that, eg we might have user 17 and user 19 but no user 18. This
    # function just renames users and movies to make all consecutive.
    users_to_IDs = {}  # map of renames
    movies_to_IDs = {} # map of renames
    B = np.zeros_like(A) # our output, same shape as input
    n_users_visited = 0 # this means "visited during our iteration"
    n_movies_visited = 0
    for idx, (user, movie, rating) in enumerate(A):
        user = int(user)     # A might be floating-point
        movie = int(movie)   # so we convert
        if user not in users_to_IDs:
            users_to_IDs[user] = n_users_visited
            n_users_visited += 1
        if movie not in movies_to_IDs:
            movies_to_IDs[movie] = n_movies_visited
            n_movies_visited += 1
        B[idx] = (users_to_IDs[user], movies_to_IDs[movie], rating)

    n = len(set(A[:, 0]))
    m = len(set(A[:, 1]))
    # Let's check we got things right
    assert n == n_users_visited
    assert m == n_movies_visited
    assert set(B[:, 0]) == set(range(n))
    assert set(B[:, 1]) == set(range(m))
    return B, n, m




# methods for display
############################################

def display(A, Ahat):
    Ahat_at_known = [Ahat[int(i), int(j)] for i, j, rating in A]
    c = np.corrcoef(ratings, Ahat_at_known)[0][1]
    examples = [(ratings[i], Ahat_at_known[i]) for i in range(5)]
    return c, examples

def calc_rmse(Ahat):
    # RMSE against known ratings only
    SSE = sum((rating - Ahat[int(i), int(j)])**2 for i, j, rating in A)
    MSE = SSE / len(A)
    return np.sqrt(MSE)

def cost(U, V, Ahat):
    error = calc_rmse(Ahat)
    regularisation = np.linalg.norm(U) + np.linalg.norm(V)
    return error + Lambda * regularisation



# core of the model
###################

def predict(U, V, C=None, D=None, M=None):
    # A rating is modelled as:
    #
    # R = M + C[i] + D[j] + sum U[i,k]V[k,j]
    #
    # M is a scalar: the mean of all ratings.
    #
    # C is a value per user: how generous is a user (positive or
    # negative, ie above or below M.)
    #
    # D is a value per movie: how good is a movie (positive or
    # negative, ie above or below M.)
    #
    # UV then models the movie-factor-user interaction, and can be
    # positive or negative.
    #
    # UV can also be positive or negative for any particular
    # item-movie, but will be centered at 0

    # @ means matrix multiplication, ie np.dot(U, V). It's new in
    # Python 3.x.
    Ahat = U @ V
    if C is not None:
        Ahat += C # C has shape (n, 1) so we can add
    if D is not None:
        Ahat += D # D has shape (1, m) so we can add
    if M is not None:
        Ahat += M # M a scalar
    return Ahat






# settings and setup
####################

timestamp = str(datetime.datetime.now()).replace(" ", "_").replace(":", "_")
dirname = "saves"
if not os.path.isdir(dirname):
    os.mkdir(dirname)


np.random.seed(0) # for replication

print_every = 5  # how often should we print?
maxits = 10000   # number of iterations allowed
tol = 0.01       # tolerance: stop when cost less than this
alpha = 0.001    # learning rate. Funk recommneds 0.001
Lambda = 0.02    # Regularisation strength for U and V. Funk
                 # recommends 0.02. (We use uppercase L because "lambda"
                 # is a Python reserved word.)
Lambda2 = 0.05   # Regularisation strength for C and D. Paterek recommends 0.05.
GD_on_CD = True  # Should we do gradient descent on C and D?


# Several datasets for testing. We customize some of the paramters
# per-dataset.

dataset_name = "fake"
if dataset_name == "movielens":
    # Achieves about RMSE 0.60, R 0.82. Very slow start (first 150 its)
    A, n, m = read_movielens()
    h = 10
    maxits = 500
    alpha = 0.001
    GD_on_CD = False # I haven't been able to make it work on this dataset
    print_every = 1
elif dataset_name == "fake":
    # Achieves about RMSE 0.22, R 0.97 with these settings.
    A, n, m = read_fake_ratings()
    h = 2
    maxits = 1000
    alpha = 0.01
    print_every = 10
elif dataset_name == "tiny":
    A, n, m = make_tiny_test_data()
    h = 2
    maxits = 1000
    alpha = 0.001
    print_every = 10


print(n, m, h)

ratings = A[:, 2]

def initialise(A):
    # Initialise all the variables: U, V, C, D, M

    # Initialise U and V with light Gaussian noise.
    # U = 0.1 * np.ones((n, h))
    # V = 0.1 * np.ones((h, m))
    U = np.random.normal(loc=0, scale=0.01, size=(n, h))
    V = np.random.normal(loc=0, scale=0.01, size=(h, m))

    # Initialise C, D and M based on the data.
    C = np.zeros((n, 1), dtype=float)
    C_count = np.zeros((n, 1))
    D = np.zeros((1, m), dtype=float)
    D_count = np.zeros((1, m))
    M = 0.0
    M_count = 0
    for i, j, rating in A:
        i = int(i)
        j = int(j)
        C[i, :] += rating
        C_count[i, :] += 1
        D[:, j] += rating
        D_count[:, j] += 1
        M += rating
        M_count += 1

    M = M / float(M_count) # mean of the data
    C = C / C_count - M    # per-user mean, less M
    D = D / D_count - M    # per-movie mean, less M
    return U, V, C, D, M

U, V, C, D, M = initialise(A)


# main loop carrying out gradient descent
#########################################

for it in range(maxits):

    Ahat = predict(U, V, C, D, M) # make a prediction

    # We copy, and make updates on the copy (it's slightly inaccurate otherwise)
    U_ = U.copy()
    V_ = V.copy()
    if GD_on_CD:
        C_ = C.copy()
        D_ = D.copy()

    if print_every is not None and it % print_every == 0:
        c = calc_rmse(Ahat)
        print(it, c, display(A, Ahat))
        if c < tol:
            break

    for i, j, rating in A:
        i = int(i)
        j = int(j)

        err = rating - Ahat[i, j] # How bad was our prediction? Can be negative!

        for k in range(h):
            # The most important lines: GD on U and V. Paterek Section
            # 3.2
            U_[i, k] += alpha * (err * V[k, j] - Lambda * U[i, k])
            V_[k, j] += alpha * (err * U[i, k] - Lambda * V[k, j])

        if GD_on_CD:
            # Paterek Section 3.3. These are done once per rating, not k times.
            C_[i, 0] += alpha * (err - Lambda2 * (C[i, 0] + D[0, j] - M))
            D_[0, j] += alpha * (err - Lambda2 * (C[i, 0] + D[0, j] - M))

    # Overwrite variables with updated versions.
    U = U_
    V = V_
    if GD_on_CD:
        C = C_
        D = D_

# Save everything.
basefilename = os.path.join(dirname, timestamp)
np.savetxt(basefilename + "_U.dat", U)
np.savetxt(basefilename + "_V.dat", V)
np.savetxt(basefilename + "_C.dat", C)
np.savetxt(basefilename + "_D.dat", D)
np.savetxt(basefilename + "_M.dat", np.array([M]))
