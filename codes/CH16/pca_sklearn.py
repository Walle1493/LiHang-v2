import numpy as np
from sklearn.decomposition import PCA


if __name__ == "__main__":
    R = np.array([[1, 0.44, 0.29, 0.33], [0.44, 1, 0.35, 0.32], \
        [0.29, 0.35, 1, 0.60], [0.33, 0.32, 0.60, 1]])
    # 去掉均值
    R = R - np.mean(R, axis=1).reshape(-1, 1)
    # assert (np.mean(x, axis=1) == np.zeros(R.shape[0])).all()
    assert ((np.mean(R, axis=1) - np.zeros(R.shape[0]))< 3e-5).all()

    # fit pca
    # for sklearn x.shape == (n_samples, n_features)
    pca = PCA(n_components=2)
    pca.fit(R.T)
    R_new = pca.fit_transform(R.T).T

    print("singular values:\n", pca.singular_values_)
    print("explained variance ratio:\n", pca.explained_variance_ratio_)
    print("transform:\n", R_new)
