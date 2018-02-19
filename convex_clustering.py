# coding: utf-8
import numpy as np


class ConvexClustering:
    def __init__(self, n_dim=1, sigma=1):
        self.n_dim = n_dim
        self.sigma = sigma
        self.p_arr = None

    def _gauss_dist(self, x):
        return np.exp(- x @ x.reshape(-1, 1) / (2 * self.sigma**2)) / np.sqrt((2*np.pi*self.sigma**2)**self.n_dim)

    def _distance_matrix(self, X):
        dist_matrix = np.zeros((X.shape[0], X.shape[0]))
        for i in range(X.shape[0]):
            for j in range(i, X.shape[0]):
                x = X[i] - X[j]
                dist_matrix[i, j] = self._gauss_dist(x)

        return dist_matrix

    def likelyhood(self, x):
        lh = 0
        for p in self.p_arr[self.p_arr > 1e-4]:
            lh += p * self._gauss_dist(x)

    def fit(self, X):
        self.p_arr = np.random.rand(X.shape[0])
        self.p_arr /= self.p_arr.sum()
        current_llh = 0

        while True:
            prev_llh = current_llh
            F = self._distance_matrix(X)
            P = F @ self.p_arr.reshape(-1, 1)
            self.p_arr = np.sum((self.p_arr.reshape(-1, 1) * F) / P, axis=1) / X.shape[0]
            current_llh = np.sum(np.log(F @ self.p_arr.reshape(-1, 1)))

            print(np.abs(current_llh - prev_llh))
            if np.abs(current_llh - prev_llh) < 1e-3:
                print(1)
                break


def main():
    import matplotlib.pyplot as plt
    cc = ConvexClustering(n_dim=2, sigma=0.1)
    X1 = np.random.normal(3, 1, (100, 2))
    X2 = np.random.normal(-1, 1, (100, 2))
    X = np.vstack((X1, X2))

    cc.fit(X)
    Y = X[np.argsort(cc.p_arr)[:40]]
    # print(X.shape, X[cc.p_arr > 0.001].shape)
    xlist = np.linspace(-5, 7, 100)
    ylist = np.linspace(-5, 7, 100)
    x, y = np.meshgrid(xlist, ylist)
    X = np.vstack((x.ravel(), y.ravel())).T
    z = np.array([cc.likelyhood(x) for x in X]).reshape(x.shape)
    plt.subplot(2, 1, 1)
    plt.xlim(-5, 7)
    plt.ylim(-5, 7)
    plt.contour(x, y, z, 8)
    plt.scatter(X1[:, 0], X1[:, 1])
    plt.scatter(X2[:, 0], X2[:, 1])

    plt.subplot(2, 1, 2)
    plt.xlim(-5, 7)
    plt.ylim(-5, 7)
    plt.scatter(Y[:, 0], Y[:, 1])
    plt.scatter(Y[:, 0], Y[:, 1])
    plt.show()


if __name__ == '__main__':
    main()
