import time
import numpy as np
from basic import Perceptron as BP
from sklearn.linear_model import Perceptron as SKP


if __name__ == '__main__':
    X = np.array([[3, 3], [4, 3], [1, 1]])
    y = [1, 1, -1]

    start = time.time()
    bp = BP()
    bp.fit(X, y)
    end = time.time()
    print("Basic perceptron costs {} seconds.".format(end - start))

    start = time.time()
    skp = SKP()
    skp.fit(X, y)
    end = time.time()
    print("Sklearn perceptron costs {} seconds.".format(end - start))
