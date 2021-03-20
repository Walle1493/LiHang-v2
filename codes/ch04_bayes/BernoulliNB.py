import numpy as np
from sklearn.naive_bayes import BernoulliNB

# 'S': 0, 'M': 1, 'L': 2
X = np.array([[1, 0], [1, 1], [1, 1], [1, 0], [1, 0], [2, 0],
              [2, 1], [2, 1], [2, 2], [2, 2], [3, 2], [3, 1],
              [3, 1], [3, 2], [3, 2]])

Y = np.array([-1, -1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, -1])

x = [[2, 0]]

clf = BernoulliNB()
clf.fit(X, Y)
res = clf.predict(x)

print(res)
# answer: [-1]

acc = clf.score(X, Y)
print(acc)
# accuracy: 0.733
