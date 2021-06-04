from sklearn.tree import DecisionTreeClassifier


# 贷款申请样本数据表

"""
1. 年龄
青年：0， 中年：1， 老年：2
2. 有工作
否：0， 是：1
3. 有自己的房子
否：0， 是：1
4. 信贷情况
一般：0， 好：1， 非常好：2
"""

samples = [
    (0, 0, 0, 0),
    (0, 0, 0, 1),
    (0, 1, 0, 1),
    (0, 1, 1, 0),
    (0, 0, 0, 0),
    (1, 0, 0, 0),
    (1, 0, 0, 1),
    (1, 1, 1, 1),
    (1, 0, 1, 2),
    (1, 0, 1, 2),
    (2, 0, 1, 2),
    (2, 0, 1, 1),
    (2, 1, 0, 1),
    (2, 1, 0, 2),
    (2, 0, 0, 0),
]

"""
类别
否：0， 是：1
"""

labels = [0, 0, 1, 1, 0,
          0, 0, 1, 1, 1,
          1, 1, 1, 1, 0]


if __name__ == '__main__':
    A = [(0, 0, 1, 0)]
    B = [(1, 1, 0, 1)]
    C = [(2, 0, 1, 0)]
    D = [(0, 0, 0, 0)]

    clf = DecisionTreeClassifier()
    clf.fit(samples, labels)

    r1 = clf.predict(A)
    r2 = clf.predict(B)
    r3 = clf.predict(C)
    r4 = clf.predict(D)

    print(r1)   # [1]
    print(r2)   # [1]
    print(r3)   # [1]
    print(r4)   # [0]
