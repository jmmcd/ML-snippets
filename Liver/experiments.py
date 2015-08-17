import numpy as np

# download bupa.dat from https://archive.ics.uci.edu/ml/datasets/Liver+Disorders
bupa = np.genfromtxt("bupa.dat", delimiter=",")

# gives almost no correlation: confirms that selector is not related to drinks
np.corrcoef(bupa[:,5], bupa[:,6])

# create the variables
x6d = (bupa[:,5] > 5)
x7 = (bupa[:,6] == 2)


# do a contingency table manually!

np.sum(~x6d & ~x7)
# 100

np.sum(~x6d & x7)
# 157

np.sum(x6d & ~x7)
# 45

np.sum(x6d & x7)
# 43

len(x6d) - np.sum(x6d)
# 257

np.sum(x6d)
# 88

len(x7) - np.sum(x7)
# 145

np.sum(x7)
# 200


