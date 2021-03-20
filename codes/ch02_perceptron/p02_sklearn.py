import numpy as np
from sklearn.linear_model import Perceptron


X = np.array([[3, 3], [4, 3], [1, 1]])
y = [1, 1, -1]

per = Perceptron(alpha=0.001)
per.fit(X, y)