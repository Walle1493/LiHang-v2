from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import load_iris


if __name__ == "__main__":
    iris = load_iris()
    X = iris.data
    y = iris.target

    clf = AdaBoostClassifier(n_estimators=100, random_state=0)
    clf.fit(X, y)

    inputs = [[5.4, 3.3, 1.7, 0.3], [5.8, 2.7, 3.2, 1.3], [5.8, 2.7, 5.1, 1.9]]
    preds = clf.predict(inputs)
    print("Inputs:")
    print(inputs)
    print("Predictions:")
    print(preds)

    score = clf.score(X, y)
    print("Score:")
    print(score)
    