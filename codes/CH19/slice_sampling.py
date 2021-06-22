# Slice Sampling

import numpy as np
import scipy.stats as st
import seaborn as sns
import matplotlib.pyplot as plt


def p(x, mu, sigma):
    return st.norm.pdf(x, loc=mu, scale=sigma)


def p_inv(y, mu, sigma):
    x = np.sqrt(-2 * sigma ** 2 * np.log(y * sigma * np.sqrt(2 * np.pi)))
    return mu - x, mu + x


def slice_sampling(mu, sigma, iter=10000):
    samples = np.zeros(iter)
    x = 0

    for i in range(iter):
        u = np.random.uniform(0, p(x, mu, sigma))
        x_lo, x_hi = p_inv(u, mu, sigma)
        x = np.random.uniform(x_lo, x_hi)
        samples[i] = x

    return samples


if __name__ == '__main__':
    mu = 65
    sigma = 32

    samples = slice_sampling(mu, sigma, iter=10000)
    sns.set()
    sns.distplot(samples, kde=False, norm_hist=True)

    x = np.arange(-100, 250)
    plt.plot(x, p(x, mu, sigma))
    plt.show()
