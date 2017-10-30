import numpy as np
import random

n = 10
m = 20
p = 0.5
h = 2

U = 2*(np.random.random((n, h))-0.5)
V = 2*(np.random.random((h, m))-0.5)
UV = U @ V
UV = (1 + 4 * (UV.max() - UV) / (UV.max() - UV.min()))
UV = np.around(UV, 0)

for i in range(n):
    for j in range(m):
        if random.random() < p:
            print("%d,%d,%d" % (i, j, UV[i, j]))
