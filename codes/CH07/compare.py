import time
from utils import load_data
from svm_smo import SVM
from sklearn.svm import SVC


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data(path="data/iris-virginica.txt")

    svm_start = time.time()
    svm = SVM(max_iter=1000)
    svm.fit(X_train, y_train)
    svm_acc = svm.score(X_test, y_test)
    svm_end = time.time()

    svc_start = time.time()
    svc = SVC(max_iter=1000)
    svc.fit(X_train, y_train)
    svc_acc = svc.score(X_test, y_test)
    svc_end = time.time()

    print("Accuracy comparison: {} and {}".format(svm_acc, svc_acc))
    print("Time consumption: {}s and {}s".format(svm_end-svm_start, svc_end-svc_start))
