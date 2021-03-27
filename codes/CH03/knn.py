from collections import namedtuple
from pprint import pformat
import numpy as np
import argparse


class Node(namedtuple('Node', 'location left_child right_child')):
    def __repr__(self):
        return pformat(tuple(self))


class KNN():

    def __init__(self,
                 k=1,
                 p=2):
        """

        :param k: knn
        :param p: p-norm(距离选择中的范数)
        """
        self.k = k
        self.p = p
        self.kdtree = None

    @staticmethod
    def _fit(X, depth=0):
        # construct kd-tree by recursion
        try:
            k = X.shape[1]
        except IndexError as e:
            return None
        # select axis=(0,1,0,1,...)
        axis = depth % k
        X = X[X[:, axis].argsort()]
        median = X.shape[0] // 2

        try:
            X[median]
        except IndexError:
            return None

        node = Node(
            location=X[median],
            # 递归搜索左右子节点
            left_child=KNN._fit(X[:median], depth + 1),
            right_child=KNN._fit(X[median + 1:], depth + 1)
        )

        return node

    def _distance(self, x, y):
        """求L_p范数(L_2 Norm)"""
        return np.linalg.norm(x-y, ord=self.p)

    def _search(self, point, tree=None, depth=0, best=None):
        """search nearest neighbour by kd-tree"""
        if tree is None:
            return best

        k = point.shape[0]
        # update best
        if best is None or self._distance(point, tree.location) < self._distance(best, tree.location):
            next_best = tree.location
        else:
            next_best = best

        # update branch
        if point[depth%k] < tree.location[depth%k]:
            next_branch = tree.left_child
        else:
            next_branch = tree.right_child
        return self._search(point, tree=next_branch, depth=depth+1, best=next_best)

    def fit(self, X):
        self.kdtree = KNN._fit(X)
        return self.kdtree

    def predict(self, X):
        rst = self._search(X, self.kdtree)
        return rst

    def predict_proba(self, X):
        pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=1, help="search k nearest neighbour")
    parser.add_argument("--p", type=int, default=2, help="Lp norm for calculating the distance between 2 points")
    parser.add_argument("--path", type=str, default="", help="data/data_3-2.txt")
    args = parser.parse_args()

    if args.path:
        X = np.loadtxt(args.path)
    else:
        X = np.array([[1, 1], [5, 1], [4, 4]], dtype=np.float)
    
    target = np.array([4, 5], dtype=np.float)

    # knn model
    clf = KNN(k=args.k, p=args.p)
    # construct kd-tree
    clf.fit(X)
    # nearest neighbour
    rst = clf.predict(target)

    print("Constructing ke-tree:")
    print(clf.kdtree)
    print("The nearest neighbour of [4, 5]:")
    print(rst)
