# Metropolis Hastings

import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
import seaborn as sns


# 设定目标分布的密度函数p(x)的特征
mus = np.array([5, 5])
sigmas = np.array([[1, .9], [.9, 1]])


def circle(x, y):
    return (x - 1) ** 2 + (y - 2) ** 2 - 3 ** 2


def pgauss(x, y):
    return st.multivariate_normal.pdf([x, y], mean=mus, cov=sigmas)


def metropolis_hastings(p, m=2000, n=10000):
    x, y = 0., 0.
    samples = np.zeros((n, 2))

    for i in range(n):
        x_star, y_star = np.array([x, y]) + np.random.normal(size=2)
        if np.random.rand() < p(x_star, y_star) / p(x, y):
            x, y = x_star, y_star
        samples[i] = np.array([x, y])

    return samples[m:]


if __name__ == '__main__':
    # 收敛步数m，迭代步数n
    m, n = 2000, 10000
    samples = metropolis_hastings(circle, m, n)
    sns.jointplot(samples[:, 0], samples[:, 1])
    plt.show()

    samples = metropolis_hastings(pgauss, m, n)
    sns.jointplot(samples[:, 0], samples[:, 1])
    plt.show()
