# Rejection Sampling

import numpy as np
import scipy.stats as st
import seaborn as sns
import matplotlib.pyplot as plt


def p(x):
    """不可以直接抽样的概率密度函数"""
    return st.norm.pdf(x, loc=30, scale=10) + st.norm.pdf(x, loc=80, scale=20)


def q(x):
    """建议分布非概率密度函数"""
    return st.norm.pdf(x, loc=50, scale=30)


def rejection_sampling(c, iter=1000):
    """接受-拒绝采样"""
    samples = []

    for i in range(iter):
        z = np.random.normal(50, 30)
        u = np.random.uniform(0, c * q(z))
        # 按均匀分布在(0,1)范围内抽样
        # u = np.random.uniform(0, 1)

        if u <= p(z):
            samples.append(z)

    return np.array(samples)


if __name__ == '__main__':
    x = np.arange(-50, 151)
    # c * q(x) >= p(x)
    c = max(p(x) / q(x))

    plt.plot(x, p(x))
    plt.plot(x, c * q(x))
    plt.show()

    s = rejection_sampling(c, iter=1000)
    sns.set()
    # sns.distplot(s)
    sns.histplot(s)
    plt.show()
