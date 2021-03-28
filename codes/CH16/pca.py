import numpy as np


class PCA():
    def __init__(self, n_components=2):
        self.n_components_ = n_components
        self.explained_variance_ratio_ = None
        self.singular_values_ = None
        self.u = None
        self.vh = None
        self.components_ = None

    def __str__(self,):
        rst = "PCA algorithms:\n"
        rst += "n_components: " + str(self.n_components_)
        return rst

    def fit(self, x):
        # check n_components and min(n_samples, n_features)
        n = x.shape[0]
        assert n > 1
        # assert (np.mean(x, axis=1) == np.zeros(n)).all()
        assert ((np.mean(x, axis=1) - np.zeros(n)) < 3e-5).all()
        x_ = x.T/np.sqrt(n-1)
        # mxk kxk kxn: m features , k components, n samples
        u, s, vh = np.linalg.svd(x_, full_matrices=False)
        self.vh = vh
        self.u = u
        self.singular_values_ = s
        self.explained_variance_ratio_ = s**2/np.sum(s**2)
        # print("u:\n", self.u)
        # print("s:\n", self.singular_values_)
        # print("vh:\n", self.vh)

        # sign flip
        # sign of keep largest value is positive
        max_abs_cols = np.argmax(np.abs(u), axis=0)
        signs = np.sign(u[max_abs_cols, range(u.shape[1])])
        u *= signs
        vh *= signs[:, np.newaxis]
        # print(s)
        # print(u)
        # print(vh)

        # print("max abs cols:\n", max_abs_cols)
        # print("max abs cols sign:\n", signs[:, np.newaxis])
        self.u = u
        self.vh = vh

    def fit_transform(self, x):
        self.fit(x)
        self.components_ = np.dot(self.vh, x)
        return self.components_


if __name__ == "__main__":
    R = np.array([[1, 0.44, 0.29, 0.33], [0.44, 1, 0.35, 0.32], \
        [0.29, 0.35, 1, 0.60], [0.33, 0.32, 0.60, 1]])
    # 去掉均值
    R = R - np.mean(R, axis=1).reshape(-1, 1)
    # assert (np.mean(x, axis=1) == np.zeros(R.shape[0])).all()
    assert ((np.mean(R, axis=1) - np.zeros(R.shape[0]))< 3e-5).all()

    # fit pca
    pca = PCA(n_components=2)
    R_new = pca.fit_transform(R)
    print("singular values:\n", pca.singular_values_)
    print("explained variance ratio:\n", pca.explained_variance_ratio_)
    print("transform:\n", R_new)
