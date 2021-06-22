# Gibbs Sampling

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def p_x_given_y(y, mus, sigmas):
    """第一个维度"""
    mu = mus[0] + sigmas[1, 0] / sigmas[0, 0] * (y - mus[1])
    sigma = sigmas[0, 0] - sigmas[1, 0] / sigmas[1, 1] * sigmas[1, 0]
    return np.random.normal(mu, sigma)


def p_y_given_x(x, mus, sigmas):
    """第二个维度"""
    mu = mus[1] + sigmas[0, 1] / sigmas[1, 1] * (x - mus[0])
    sigma = sigmas[1, 1] - sigmas[0, 1] / sigmas[0, 0] * sigmas[0, 1]
    return np.random.normal(mu, sigma)


def gibbs_sampling(mus, sigmas, m=2000, n=10000):
    """
    吉布斯采样
    m: 收敛步数
    n: 迭代步数
    """
    samples = np.zeros((n, 2))
    y = np.random.rand() * 10

    for i in range(n):
        x = p_x_given_y(y, mus, sigmas)
        y = p_y_given_x(x, mus, sigmas)
        samples[i, :] = [x, y]

    return samples[m:]


if __name__ == '__main__':
    # 这里每个样本的维度为2
    mus = np.array([5, 5])
    sigmas = np.array([[1, .9], [.9, 1]])

    # 收敛步数，迭代步数
    m, n = 2000, 10000
    # dim = (n-m, 2)
    samples = gibbs_sampling(mus, sigmas, m, n)

    sns.jointplot(samples[:, 0], samples[:, 1])
    plt.show()
