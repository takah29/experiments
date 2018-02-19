# coding: utf-8
import numpy as np
from numpy.linalg import inv, det


class GMM:
    def __init__(self, n_dim=1, n_components=1, p_arr=None):
        self.n_dim = n_dim
        self.n_components = n_components

        self.p_arr = np.random.rand(n_components) if p_arr is None else p_arr
        self.p_arr /= self.p_arr.sum()
        self.mu_arr = np.random.rand(n_components, n_dim)
        self.sigma_arr = np.random.rand(n_components, n_dim, n_dim)
        self.sigma_arr = self.sigma_arr.transpose(0, 2, 1) @ self.sigma_arr

    def gaussian_distribution(self, x, i):
        v = np.exp(-(x - self.mu_arr[i]) @ inv(self.sigma_arr[i]) @ (x - self.mu_arr[i]).reshape(-1, 1) / 2)[0]
        return v / np.sqrt((2 * np.pi)**self.n_dim * det(self.sigma_arr[i]))

    def likelyhood(self, x):
        normal_vec = [self.gaussian_distribution(x, i) for i in range(self.n_components)]
        return (self.p_arr @ normal_vec)

    def fit(self, X):
        P1 = np.zeros((self.n_components, X.shape[0]))

        sum_llh = 0
        while True:
            prev_llh = sum_llh
            sum_llh = 0
            for i in range(self.n_components):
                for k in range(X.shape[0]):
                    tmp_lh = self.likelyhood(X[k])
                    P1[i, k] = self.p_arr[i] * self.gaussian_distribution(X[k], i) / tmp_lh
                    sum_llh += np.log(tmp_lh)

            print('logP(w_i; x_k): {}'.format(sum_llh))
            if np.abs(prev_llh - sum_llh) < 1e-16:
                break

            sum_p_arr = np.sum(P1, axis=1)
            self.mu_arr = (P1 @ X) / sum_p_arr.reshape(-1, 1)

            tmp_sigma_list = []
            for i in range(self.n_components):
                S = np.zeros(self.sigma_arr.shape[1:])
                for k in range(X.shape[0]):
                    s = (X[k] - self.mu_arr[i])
                    S += P1[i, k] * (s.reshape(-1, 1) @ s.reshape(1, -1))

                tmp_sigma_list.append(S)

            self.sigma_arr = np.stack(tmp_sigma_list) / sum_p_arr.reshape(-1, 1, 1)
            self.p_arr = sum_p_arr / X.shape[0]


def main():
    import matplotlib.pyplot as plt
    gmm = GMM(n_dim=2, n_components=2, p_arr=np.array([0.2, 0.2]))
    X1 = np.random.normal(3, 1, (50, 2))
    X2 = np.random.normal(-1, 1, (50, 2))
    X = np.vstack((X1, X2))

    gmm.fit(X)

    xlist = np.linspace(-5, 7, 200)
    ylist = np.linspace(-5, 7, 200)
    x, y = np.meshgrid(xlist, ylist)
    X = np.vstack((x.ravel(), y.ravel())).T
    z = np.array([gmm.likelyhood(x) for x in X]).reshape(x.shape)
    plt.contourf(x, y, z, 8)
    plt.scatter(X1[:, 0], X1[:, 1])
    plt.scatter(X2[:, 0], X2[:, 1])
    plt.show()

    # print(gmm._gaussian_distribution(0, np.array([0]), np.array([[1]])))


if __name__ == '__main__':
    main()
