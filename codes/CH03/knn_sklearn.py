import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import argparse


def main():
    # get arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_neighbors", type=int, default=5)
    parser.add_argument("--algorithm", type=str, default="kd_tree")
    parser.add_argument("--p", type=int, default=2, help="Power parameter for the Minkowski metric")
    parser.add_argument("--metric", type=str, default="minkowski", help="the distance metric to use for the tree.")
    args = parser.parse_args()

    # load data
    iris = load_iris()
    X_train, X_test, y_train, y_test = \
        train_test_split(iris.data, iris.target, test_size=0.2)
    
    # train model
    clf = KNeighborsClassifier(n_neighbors=args.n_neighbors, algorithm=args.algorithm, p=args.p, metric=args.metric)
    clf.fit(X_train, y_train)

    # predictions
    predictions = clf.predict(X_test)
    score = clf.score(X_test, y_test)

    # print results
    print("predictions:")
    print(predictions)
    print("labels:")
    print(y_test)
    print("score:")
    print(score)


if __name__ == "__main__":
    main()


# predictions:
# [2 2 2 2 0 1 0 0 0 1 1 2 1 0 2 0 2 1 2 2 2 2 2 0 1 0 2 2 0 2]
# labels:
# [2 2 2 2 0 1 0 0 0 1 1 2 1 0 2 0 2 1 2 2 2 1 2 0 1 0 2 2 0 2]
# score:
# 0.9666666666666667
