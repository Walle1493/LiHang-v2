import numpy as np
from sklearn.mixture import GaussianMixture


if __name__ == "__main__":
    X = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])
    gmm = GaussianMixture(n_components=2)
    gmm.fit(X)
    
    print(gmm.weights_)
    print(gmm.means_)

    prediction = gmm.predict([[0, 0], [12, 3]])
    print(prediction)
