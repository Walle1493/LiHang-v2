import time
import numpy as np
from p01_perceptron import Perceptron as P1
from p02_sklearn import Perceptron as P2

X = np.array([[3, 3], [4, 3], [1, 1]])
y = [1, 1, -1]


if __name__ == '__main__':
    p1 = P1()
    p2 = P2()
    p1.fit(X, y)
    p1.fit(X, y)