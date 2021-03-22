import numpy as np
import pandas as pd


def load_data(path, test_ratio=0.25):
    iris = pd.read_csv(path, header=None)
    # shuffle data
    iris = iris.sample(frac=1)

    # split 25% of datasets to test_datasets
    test_size = int(iris.shape[0] * test_ratio)
    # train_ratio = iris.shape[0] - test_ratio

    X_train = np.array(iris.iloc[test_size:, :4])
    X_test = np.array(iris.iloc[:test_size, :4])
    y_train = np.array(iris.iloc[test_size:, -1])
    y_test = np.array(iris.iloc[:test_size, -1])

    return X_train, X_test, y_train, y_test
