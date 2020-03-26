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
# plt.show()

lbda = 0.5
T = 25
results = []
for x in range(0, 10000):
    results.append(len(generate_poisson_trajectory(lbda, T)) - 1)

data = np.arange(0, 25, 0.1)
plt.hist(results, bins=data)
# plt.show()

# ex. 2
t = [i for i in range(0, 26)]
w = [0 for i in range(0, 26)]
for i in t[:-1]:
    w[i + 1] = w[i] + math.sqrt(t[i + 1] - t[i]) * np.random.normal()

plt.step(t, w)
# plt.show()


# S - macierz kwadratowa dodatnio okreslona

def create_sigma(T):
    sigma = [[0.0] * T for i in range(0, T)]
    for i in range(0, T):
        for j in range(0, T):
            sigma[i][j] = min(i+1, j+1)
    return sigma

def cholesky(S):
    n = len(S)
    L = [[0.0] * n for i in range(0, n)]
    for i in range(0, n):
        for k in range(0, i + 1):
            tmp_sum = sum(L[i][j] * L[k][j] for j in range(0, k))
            if i == k:
                L[i][k] = math.sqrt(S[i][i] - tmp_sum)
            else:
                L[i][k] = (1.0 / L[k][k] * (S[i][k] - tmp_sum))
    return L

T=10
sigma = create_sigma(T-1)
L = np.array(cholesky(sigma))
X = np.array([np.random.normal() for i in range(0, T-1)])
w = [0] + list(L@X)
print(w)