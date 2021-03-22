import numpy as np


class Perceptron(object):
    def __init__(self, learning_rate=1):
        self.w = np.array([0, 0]).reshape((-1, 1))
        self.b = 0
        self.eta = learning_rate

    def sign(self, x):
        return -1 if x < 0 else 1

    def calculate(self, X):
        yH = np.matmul(X, self.w) + self.b
        return np.apply_along_axis(self.sign, 1, yH)

    def get_wrong(self, X, yH, Y):
        for x, yh, y in zip(X, yH, Y):
            if yh != y:
                return {'x': x, 'y': y}
        return None

    def fit(self, X, y):
        import pdb;pdb.set_trace()
        while True:
            yH = self.calculate(X)
            wrong = self.get_wrong(X, yH, y)
            print(f"Wrong Point {wrong}")
            if not wrong:
                break
            self.w = self.w + self.eta * (wrong['x'] * wrong['y']).reshape((-1, 1))
            self.b = self.b + self.eta * wrong['y']
            print(f"update w to {self.w} update b to {self.b}")


if __name__ == '__main__':
    X = np.array([[3, 3], [4, 3], [1, 1]])
    y = [1, 1, -1]

    # zhouxiabing
    # X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    # y = [-1, -1, -1, 1]

    perceptron = Perceptron()
    perceptron.fit(X, y)
