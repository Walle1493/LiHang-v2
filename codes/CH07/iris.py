import numpy as np
from sklearn.svm import LinearSVC, SVC
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import argparse
import logging


class SVM():
    def __init__(self, args):
        self.x_train, self.x_test, self.y_train, self.y_test = None, None, None, None
        self.model = None
        self.coef = None
        self.intercept = None
        self.predictions = None
        self.accuracy = None
        self.max_iter = args.max_iter
        self.svm_method = args.svm_method
    
    def load_dataset(self, dataset):
        try:
            self.x_train, self.x_test, self.y_train, self.y_test = \
                train_test_split(dataset["data"], dataset["target"])
        except:
            raise Exception("Something wrong with dataset.")

    def build(self):
        if self.svm_method == "LinearSVC":
            self.model = LinearSVC(max_iter=self.max_iter)
            return self.model
        elif self.svm_method == "SVC":
            self.model = SVC(max_iter=self.max_iter)
            return self.model
        else:
            raise Exception("SVM must be selected from \"LinearSVC\" or \"SVC\"")

    def train(self):
        self.model.fit(self.x_train, self.y_train)
        self.coef = self.model.coef_
        self.intercept = self.model.intercept_
        return self.model

    def predict(self):
        self.predictions = self.model.predict(self.x_test)
        self.accuracy = np.sum(self.predictions == self.y_test) / len(self.y_test)
        return self.predictions

    def __getattr__(self, item):
        return "Cannot find this attribute!"


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser()
    parser.add_argument("--max_iter", type=int, default=5000)
    parser.add_argument("--svm_method", type=str, default="LinearSVC", help="LinearSVC or SVC")
    args = parser.parse_args()

    iris = load_iris()
    svm = SVM(args)
    svm.load_dataset(dataset=iris)

    svm.build()
    logger.info(svm.model)

    svm.train()
    logger.info(svm.coef)
    logger.info(svm.intercept)

    predictions = svm.predict()
    logger.info(predictions)
    logger.info(svm.accuracy)


if __name__ == "__main__":
    main()
