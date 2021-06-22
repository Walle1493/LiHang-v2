import sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


if __name__ == "__main__":
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3)

    # build logistic model
    clf = LogisticRegression(max_iter=1000)
    # train
    clf.fit(X_train, y_train)
    # predict
    prediction = clf.predict(X_test)

    print("Prediction:")
    print(prediction)
    print("Labels:")
    print(y_test)

    # Probabilities
    proba = clf.predict_proba(X_test)

    # Accuracy
    score = clf.score(X_test, y_test)
    print("Accuracy:")
    print(score)

    # print(clf.classes_) # [0 1 2]
    # print(clf.coef_)
    # print(clf.intercept_)
    # print(clf.n_iter_)  # [90]
