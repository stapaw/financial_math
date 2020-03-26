# Exercise 1
# source used:  http://www.math.uchicago.edu/~may/VIGRE/VIGRE2010/REUPapers/Mcquighan.pdf
#  http://statweb.stanford.edu/~kjross/Lec25_1121.pdf
import numpy as np
import math
import matplotlib.pyplot as plt


def generate_poisson_trajectory(lbda, T):
    ts = [0]
    t = 0
    while t <= T:
        u = np.random.uniform(0, 1)
        y = -math.log(1 - u) / lbda
        t = t + y
        if t <= T:
            ts.append(t)
    return ts


ts = generate_poisson_trajectory(0.5, 25)
plt.step(ts, [i for i in range(0, len(ts))])
plt.show()

lbda = 0.5
T = 25
results = []
for x in range(0, 10000):
    results.append(len(generate_poisson_trajectory(lbda, T)) - 1)

data = np.arange(0, 25, 0.1)
plt.hist(results, bins=data)
plt.show()
