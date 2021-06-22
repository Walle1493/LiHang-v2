from ast import parse
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import time
import logging
import argparse


class LogisticRegression(object):

    def __init__(self,
                 learning_step = 0.0001,
                 epsilon=0.001,
                 n_iter=1500):
        self.learning_step = learning_step
        self.epsilon_ = epsilon
        self.n_iter_ = n_iter
        self.coef_ = np.array([])
        self.cols_ = []

    def fit(self, x_, y_):
        return self.gradient_descent(x_, y_, epsilon_=self.epsilon_, n_iter=self.n_iter_)

    def predict(self, x_):
        # print(self.cols_, self.coef_)
        rst = np.array([self.cols_[idx] for idx in [np.argmax(rst) for rst in sigmoid(np.dot(x_, self.coef_.T))]])
        return rst

    def gradient_descent(self, x_, y_, epsilon_=0.00001, n_iter=1500):
        n = x_.shape[len(x_.shape)-1]
        # f_his = []
        # one-hot encoding
        y_ = pd.get_dummies(y_)
        w_ = np.array([])
        # print(n, y_.shape, y_.columns)

        # OvR for multiclass N个分类器 ck vs rest
        for ck in np.arange(y_.shape[1]):
            wck_ = np.zeros(n)
            # k = 0
            for k in np.arange(n_iter):
                # f_xk = self.f(x_, y_.values[:, ck], wck_)
                g_k = self.g(x_, y_.values[:, ck], wck_)

                if np.average(g_k*g_k) < epsilon_:
                    w_ = wck_ if w_.size == 0 else np.vstack([w_, wck_])
                    break
                else:
                    p_k = -g_k
                lambda_k = 0.0000001  # TODO: 更新算法
                wck_ = wck_ + lambda_k*p_k
                # f_his.append(f_xk)
            if k == n_iter-1:
                w_ = wck_ if w_.size == 0 else np.vstack([w_, wck_])
            # print("progress: %d done" % ck)
        self.coef_ = w_
        self.cols_ = y_.columns.tolist()
        return self.coef_, self.cols_


def f(x_, y_, w_):
    # Logistic Regression Loss
    m = y_.size
    rst_ = -(1 / m) * np.sum(np.dot(x_, w_) * y_ - np.log(1 + np.exp(np.dot(x_, w_))))
    return rst_


def g(x_, y_, w_):
    m = y_.size
    # y is one-hot form
    # print(y_)
    # probe here and check
    # x_.shape, y_.shape, w_.shape
    # np.dot(x_, w_).shape
    # sigmoid(np.dot(x_, w_)).shape
    # np.dot(x_.T, y_ * sigmoid(np.dot(x_, w_))).shape
    rst_ = -(1 / m) * np.dot(x_.T, y_-sigmoid(np.dot(x_, w_)))
    return rst_


def sigmoid(x_):
    p = np.exp(x_)
    p = p / (1 + p)
    return p


def load_data(path_='data/train.csv'):
    """
    data size is 28x28, 784
    :param path_:
    :return:
    """
    raw_data = pd.read_csv(path_)
    y = raw_data["label"].values
    del raw_data["label"]
    X = raw_data.values
    return X, y


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument()
    # args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)
    
    logger.info("Start read data")
    time_1 = time.time()
    X, y = load_data()
    X = X[:300]
    y = y[:300]
    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.3, random_state=2021)
    time_2 = time.time()
    logger.info("read data cost %.4s second\n", time_2 - time_1)

    logger.info("Start training")
    clf = LogisticRegression()
    clf.f = f
    clf.g = g
    clf.fit(train_x, train_y)
    time_3 = time.time()
    logger.info("training cost %.4s second\n", time_3 - time_2)

    logger.info("Start predicting")
    pred = clf.predict(test_x)
    time_4 = time.time()
    logger.info("predicting cost %.4s second\n", time_4 - time_3)

    score = accuracy_score(test_y, pred)
    logger.info("The accuracy score is %.4s", score)
