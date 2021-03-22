import numpy as np
import pandas as pd
from sklearn.svm import SVC
from utils import load_data


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data(path="data/iris-virginica.txt")
    svm = SVC(max_iter=1000)
    svm.fit(X_train, y_train)
    score = svm.score(X_test, y_test)
    print(score)
