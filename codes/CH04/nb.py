import numpy as np
import pandas as pd
import argparse


class NB():

    def __init__(self, lambda_):
        self.lambda_ = lambda_  # Laplace Smoothing
        self.classes_ = None    # Label classes (-1, 1)
        # Example: (0, 1, -1) stands for on the condition of label=-1 the probability of feature(0)=1
        self.prior_ = None  # Conditional Probability
        self.class_prior_ = None    # Prior Probability
        self.class_count_ = None    # amount of each class within label

    def fit(self, X, y):
        # array([-1,  1], dtype=int64)
        self.classes_ = np.unique(y)
        # to df
        X = pd.DataFrame(X)
        y = pd.DataFrame(y)

        # 1:9, -1:6
        self.class_count_ = y[y.columns[0]].value_counts()
        # 1:0.6, -1:0.4
        # (Maximum Likelihood Extimate)MLE：lambda_=0
        # self.class_prior_ = self.class_count_ / y.shape[0]

        # Bayesian Estimation：add lambda_
        # add laplace smoothing(self.lambda_), K=len(self.classes_)
        self.class_prior_ = (self.class_count_ + self.lambda_) / \
            (y.shape[0] + len(self.classes_) * self.lambda_)
        # prior
        self.prior_ = dict()
        # traverse all features X from (X(0), X(1)) -> idx
        for idx in X.columns:
            # traverse all labels y from (-1, 1) -> j
            for j in self.classes_:
                # for X(idx), cal the num of X(idx)[0], X(idx)[1], X(idx)[2]
                p_x_y = X[(y == j).values][idx].value_counts()
                for i in p_x_y.index:
                    # cal conditional probability
                    # add laplace smoothing(self.lambda_), S_j=len(p_x_y)
                    self.prior_[(idx, i, j)] = (p_x_y[i] + self.lambda_) / \
                        (self.class_count_[j] + len(p_x_y) * self.lambda_)

    def predict(self, X):
        rst = []
        for class_ in self.classes_:
            py = self.class_prior_[class_]
            pxy = 1
            # posterior probability = prior probability * conditional probability
            # for each class，cal posterior probability
            for idx, x in enumerate(X):
                pxy *= self.prior_[(idx, x, class_)]
            # put all posterior probability into result, and return the class of max probability
            rst.append(py*pxy)
        return self.classes_[np.argmax(rst)]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/data_4-1.csv")
    parser.add_argument("--lambda_", type=float, default=1.0, \
        help="Lambda for Laplace smoothing which 0 stands for Maximum Likelihood Estimate and 1 stands for Bayesian Estimate")
    args = parser.parse_args()

    # data
    data = pd.read_csv(args.data_path, header=None, sep=",")
    print("<data>")
    print(data)
    print()
    X = data[data.columns[0:2]]
    y = data[data.columns[2]]

    nb = NB(lambda_=args.lambda_)
    nb.fit(X, y)

    # probability
    print("<prior probabilty>")
    print(nb.class_prior_)
    print()
    print("<conditional probabilty>")
    print(nb.prior_)
    print()
    
    # predictions
    predictions = nb.predict([2, "S"])
    print("<predictions for (2, S)>")
    print(predictions)
    print()
