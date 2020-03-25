from math import pi, e, cos, sqrt, factorial, floor, erf
import numpy as np
from matplotlib import pyplot as plt
from scipy.special import comb


def linear_generator(a1, a0, m, seed=0):
    x = (a1 * seed + a0) % m
    return x, x / (m - 1)


def JKISS(x, y, z, c):
    x = to_int32(to_int32(314527869 * x) + 1234567)
    y ^= to_int32(y << 5)
    y ^= y >> 7
    y ^= to_int32(y << 22)
    t = ((4294584393 * z) % (2 ** 64) + c) % (2 ** 64)
    c = t >> 32
    z = to_int32(t)
    return to_int32(x + y + z), x, y, z, c


def to_int32(number):
    return number % (2 ** 32)


def init():
    x = 123456789
    y = 987654321
    z = 43219876
    c = 6543217
    return x, y, z, c


# exercise 1
seed = 0
generated_linear = []
for i in range(0, 1000):
    seed, x = linear_generator(pi * 10 ** 9, e * 10 ** 9, 2 ** 35, seed)
    generated_linear.append(x)

bins = np.linspace(0, 1, 20)
plt.hist(generated_linear, bins=bins, alpha=0.5)


# plt.show()


# exercise 2
def f(x):
    return (1 + cos(x) * e ** ((-x ** 2) / 2)) / (1 + e ** (-1 / 2) * sqrt(2 * pi))


data = np.arange(-10, 10, 0.1)
f_data = np.array([f(x) for x in data])
plt.plot(data, f_data)


# plt.show()


def f_generator(a, b, d):
    x = np.random.uniform(a, b)
    y = np.random.uniform(0, d)
    if y > f(x):
        x = f_generator(a, b, d)
    return x


data = np.arange(-10, 10, 0.5)
f_data = np.array([f_generator(-10, 10, 0.8) for x in range(0, 10000)])
plt.hist(f_data, bins=data, alpha=0.5)


# plt.show()
# exercise 3

def poisson_pmf(lbda, k):
    return (lbda ** k * e ** (-lbda)) / factorial(k)


def poisson_cdf(lbda, k):
    result = 0
    if k >= 0:
        result = poisson_pmf(lbda, k) + poisson_cdf(lbda, k - 1)
    return result


def poisson_generator1(lbda):
    x = np.random.uniform(0, 1)
    result = 0
    while (poisson_cdf(lbda, result - 1) > x) or (poisson_cdf(lbda, result) <= x):
        result += 1
    return result


def poisson_generator2(lbda):
    k = 0
    p = 1
    L = e ** (-lbda)
    while p >= L:
        k = k+1
        p = p* np.random.uniform(0,1)
    return k-1

print(poisson_pmf(1, 1), poisson_pmf(1, 2), poisson_pmf(10, 10))
print(poisson_cdf(1, 1), poisson_cdf(1, 2))
data = np.arange(0, 20, 1)
f_data = np.array([poisson_generator1(1) for x in range(0, 10000)])
f_data2 = np.array([poisson_generator2(1) for x in range(0, 10000)])
plt.hist(f_data, bins=data, alpha=0.5)
plt.hist(f_data2, bins=data, alpha=0.5)
plt.show()


# exercise 4
def custom_cdf(x):
    binomial_coef = sum(comb(10, i) * (1 / 3) ** i * (2 / 3) ** (10 - i) for i in range(0, floor(x) + 1))
    normal_coef = 1 / 2 * (1 + erf(x / sqrt(2)))
    exp_coef = 1 - e ** (-x)
    return (binomial_coef + normal_coef + exp_coef) / 3


data = np.arange(0, 20, 0.1)
f_data = np.array([custom_cdf(x) for x in data])
plt.plot(f_data)
plt.show()

# exercise 5
x, y, z, c = init()
for i in range(0, 10):
    result, x, y, z, c = JKISS(x, y, z, c)
    print(result / 4294967295)
