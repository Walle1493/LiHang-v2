import numpy as np
import pandas as pd
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/data_4-1.csv")
    parser.add_argument("--alpha", type=float, default=1.0, \
        help="Alpha for Laplace smoothing which 0 stands for Maximum Likelihood Estimate and 1 stands for Bayesian Estimate")
    parser.add_argument("--method", type=str, default="BernoulliNB", \
        help="Choose method from BernoulliNB, GaussianNB and MultinomialNB")
    args = parser.parse_args()

    # data
    X = np.array([[1, 0], [1, 1], [1, 1], [1, 0], [1, 0], [2, 0],
              [2, 1], [2, 1], [2, 2], [2, 2], [3, 2], [3, 1],
              [3, 1], [3, 2], [3, 2]])
    y = np.array([-1, -1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, -1])
    
    # train
    if args.method.lower() == "BernoulliNB".lower():
        nb = BernoulliNB(alpha=args.alpha)
    elif args.method.lower() == "GaussianNB".lower():
        nb = GaussianNB()
    elif args.method.lower() == "MultinomialNB".lower():
        nb = MultinomialNB(alpha=args.alpha)
    else:
        raise Exception("Method must be chosen from one of BernoulliNB, GaussianNB and MultinomialNB!")
    nb.fit(X, y)

    #predict and accuracy(training set)
    prediction = nb.predict([[2, 0], [3, 2]])
    print("<prediction>")
    print(prediction)
    accuracy = nb.score(X, y)
    print("<accuracy>")
    print(accuracy)
