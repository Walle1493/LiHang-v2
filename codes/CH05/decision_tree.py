from ast import parse
import numpy as np
import pandas as pd
import argparse
import logging


class DecisionTree(object):

    def __init__(self,
                 tol=10e-3,
                 criterion='gain',
                 min_samples_leaf=5):
        self.tree = dict()
        self.tol = tol
        self.criterion = criterion
        self.criteria = {"gain": self._gain,
                         "gain_ratio": self._gain_ratio}
        self.alpha = 0
        self.num_leaf = 0
        self.importance = None
        self.min_samples_leaf = min_samples_leaf

    def fit(self,
            X,
            y):
        return self._build_tree(X, y)

    def _search(self,
                X,
                parent=None):
        if parent is None:
            parent = self.tree
        key_x = list(parent.keys())[0]
        # is leaf
        if parent[key_x] is None:
            # {key_x: None} is leaf node
            return key_x
        else:
            key_child = X[key_x].values[0]
            # print("\n%s|%s|%s|%s\n" % (parent, key_x, key_child, parent[key_x][key_child].keys()))
            return self._search(X, parent=parent[key_x][key_child])

    def predict(self,
                X):
        return self._search(X)

    def _cal_loss(self, X, y):
        #
        pass

    def _pruning(self):

        pass

    @staticmethod
    def _cal_entropy(y):
        if y.shape[0] == 0:
            return 0
        unique, cnts = np.unique(y, return_counts=True)
        freq = cnts / y.shape[0]
        return -np.sum(freq * np.log2(freq))

    @staticmethod
    def _cal_conditioanl_entropy(X, y):
        if X.shape[0] == 0 or y.shape[0] == 0:
            return 0
        rst = 0
        items, cnts = np.unique(X, return_counts=True)
        for item, cnt in zip(items, cnts):
            ent = DecisionTree._cal_entropy(y[X == item])
            freq = cnt/X.shape[0]
            rst += freq*ent
        return rst

    @staticmethod
    def _gain(X, y):
        return DecisionTree._cal_entropy(y) - DecisionTree._cal_conditioanl_entropy(X, y)

    @staticmethod
    def _gain_ratio(X, y):
        return DecisionTree._gain(X, y) / DecisionTree._cal_entropy(X)

    @staticmethod
    def _cal_gini(X, y):
        pass

    def _min_samples_leaf_check(self,
                                X):
        items, cnts = np.unique(X, return_counts=True)
        return np.min(cnts) < self.min_samples_leaf

    def _build_tree(self,
                    X,
                    y):
        ck, cnts = np.unique(y, return_counts=True)
        # same y
        if ck.shape[0] == 1:
            self.num_leaf += 1
            return {ck[0]: None}
        elif X.shape[1] == 0:
            self.num_leaf += 1
            return {ck[np.argmax(cnts)]: None}
        else:
            rst = 0
            cols = X.columns.tolist()
            rst_col = cols[0]
            for col in cols:
                criterion = self.criteria[self.criterion](X[col], y)
                if criterion >= rst:
                    rst, rst_col = criterion, col
            if criterion < self.tol:
                self.num_leaf += 1
                return {ck[np.argmax(cnts)]: None}

            # min_leaf_node check
            if self._min_samples_leaf_check(X[rst_col]):
                self.num_leaf += 1
                return {ck[np.argmax(cnts)]: None}

            cols.remove(rst_col)
            rst = dict()
            X_sub = X[cols]
            for x in np.unique(X[rst_col]):
                mask = X[rst_col] == x
                rst.update({x: self._build_tree(X_sub[mask], y[mask])})
            self.tree = {rst_col: rst}
        return self.tree


def ID3(args, logger):
    raw_data = pd.read_csv(args.path)
    cols = raw_data.columns
    X = raw_data[cols[1:-1]]
    y = raw_data[cols[-1]]
    # default criterion: gain
    clf = DecisionTree()
    clf.fit(X, y)
    logger.info("ID3 Algorithm")
    logger.info("Criterion: gain")
    logger.info("Decision tree:\n%s", clf.tree)


def C4_5(args, logger):
    raw_data = pd.read_csv(args.path)
    cols = raw_data.columns
    X = raw_data[cols[1:-1]]
    y = raw_data[cols[-1]]
    # criterion: gain_ratio
    clf = DecisionTree(criterion="gain_ratio")
    clf.fit(X, y)
    logger.info("C4.5 Algorithm")
    logger.info("Criterion: gain_ratio")
    logger.info("Decision tree:\n%s", clf.tree)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="data/data_5-1.txt")
    parser.add_argument("--criterion", type=str, default="gain", choices=["gain", "gain_ratio"])
    args = parser.parse_args()

    if args.criterion == "gain":
        ID3(args, logger)
    elif args.criterion == "gain_ratio":
        C4_5(args, logger)
    else:
        # raise Exception("Wrong parameter \"criterion\"")
        pass
