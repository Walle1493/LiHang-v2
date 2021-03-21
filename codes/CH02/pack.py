import numpy as np
from sklearn.linear_model import Perceptron


if __name__ == "__main__":
    X = np.array([[3, 3], [4, 3], [1, 1]])
    y = [1, 1, -1]

    perceptron = Perceptron(alpha=0.001)
    perceptron.fit(X, y)
