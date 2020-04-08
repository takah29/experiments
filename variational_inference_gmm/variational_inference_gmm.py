from math import gamma
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import det, inv
from scipy.special import digamma
from scipy.stats import invwishart
from sklearn.mixture import GaussianMixture


class VariationalInferenceGmm:
    def __init__(self, n_components=1):
        self.n_components = n_components
        self.pi = np.ones(n_components) / n_components

        self.MAXITER = 100

    def __initialize(self, X):
        self.N, self.n_dim = X.shape

        # Parameter of Normal dist p(x)
        self.m_ = np.random.rand(self.n_components, self.n_dim)
        self.m = self.m_.copy()
        self.beta_ = np.ones(self.n_components) * 0.1
        self.beta = self.beta_.copy()

        # Parameter of Wishart dist p(L)
        self.nu_ = np.ones(self.n_components) * (self.n_dim + 1.0)
        self.nu = self.nu_.copy()
        self.W_ = np.stack([np.identity(self.n_dim) for _ in range(self.n_components)])
        self.W = self.W_.copy()

        # Parameter of Dirichlet dist
        self.alpha_ = np.ones(self.n_components) * 100.0
        self.alpha = self.alpha_.copy()

        # Parameter of Categorical dist
        self.eta = np.random.multinomial(1, np.ones(self.n_components) / self.n_components, self.N)

    def fit(self, X):
        self.__initialize(X)
        for i in range(self.MAXITER):
            print(f"iteration {i + 1}")

            # update q(S)
            self.__update_S_approx_dist(X)

            # update q(mu, L)
            self.__update_mu_L_approx_dist(X)

            # update q(pi)
            self.__update_pi_approx_dist()

        return self.eta

    def score_samples(self, X):
        X_lh = []
        ex_S = []
        for x in X:
            x_lh, ex_s = self.__score_sample(x)
            X_lh.append(x_lh)
            ex_S.append(ex_s)
        return np.array(X_lh), np.vstack(ex_S)

    def __update_S_approx_dist(self, X):
        self.ex_L = np.zeros((self.n_components, self.n_dim, self.n_dim))
        self.ex_ln_det_L = np.zeros(self.n_components)
        self.ex_L_mu = np.zeros((self.n_components, self.n_dim))
        self.ex_muT_L_mu = np.zeros(self.n_components)
        self.ex_ln_pi = np.zeros(self.n_components)

        for k in range(self.n_components):
            self.ex_L[k] = self.nu[k] * self.W[k]
            self.ex_ln_det_L[k] = (
                digamma(self.nu[k] + 1 - np.arange(1, self.n_dim + 1) / 2).sum()
                + self.n_dim * np.log(2)
                + np.log(det(self.W[k]))
            )
            self.ex_L_mu[k] = self.nu[k] * self.m[k] @ self.W[k]
            self.ex_muT_L_mu[k] = (
                self.nu[k] * self.m[k] @ self.W[k] @ self.m[k][:, np.newaxis]
                + self.n_dim / self.beta[k]
            )
            self.ex_ln_pi[k] = digamma(self.alpha[k]) - digamma(self.alpha.sum())

        for n in range(self.N):
            for k in range(self.n_components):
                self.eta[n, k] = (
                    -0.5 * X[n] @ self.ex_L[k] @ X[n][:, np.newaxis]
                    + self.ex_L_mu[k] @ X[n][:, np.newaxis]
                    - 0.5 * self.ex_muT_L_mu[k]
                    + 0.5 * self.ex_ln_det_L[k]
                    + self.ex_ln_pi[k]
                )
        self.eta = np.exp(self.eta)
        self.eta = self.eta / self.eta.sum(axis=1)[:, np.newaxis]

    def __update_mu_L_approx_dist(self, X):
        ex_S = self.eta

        for k in range(self.n_components):
            self.beta[k] = ex_S[:, k].sum() + self.beta_[k]
            self.m[k] = (ex_S[:, k] @ X + self.beta_[k] * self.m_[k]) / self.beta[k]
            self.W[k] = inv(
                (ex_S[:, k] * X.T) @ X
                + self.beta_[k] * self.m_[k][:, np.newaxis] @ self.m_[k][np.newaxis, :]
                - self.beta[k] * self.m[k][:, np.newaxis] @ self.m[k][np.newaxis, :]
                + inv(self.W_[k])
            )
            self.nu[k] = ex_S[:, k].sum() + self.nu_[k]

    def __update_pi_approx_dist(self):
        ex_S = self.eta
        self.alpha = ex_S.sum(axis=0) + self.alpha_

    def __score_sample(self, x):
        # return p(x_* | s_*, X, S) and <s_*>_p(s_* | x_*, X, S)
        prop_ex_s = self.__propotional_ex_s(x)
        x_lh = prop_ex_s.sum()
        ex_s = prop_ex_s / x_lh
        return x_lh, ex_s

    def __propotional_ex_s(self, x):
        s = np.zeros(self.n_components)
        for k in range(self.n_components):
            mu = self.m[k]
            L = (1 - self.n_dim + self.nu[k]) * self.beta[k] / (1 + self.beta[k]) * self.W[k]
            nu = 1 - self.n_dim + self.nu[k]
            s[k] = (self.alpha[k] / self.alpha.sum()) * self.__multi_t_distribution(x, mu, L, nu)
        return s

    def __multi_t_distribution(self, x, mu, L, nu):
        return (
            gamma((nu + self.n_dim) / 2)
            / gamma(nu / 2)
            * det(L) ** 0.5
            / (np.pi * nu) ** (0.5 * self.n_dim)
            * (1 + (x - mu) @ L @ (x - mu)[:, np.newaxis] / nu) ** (-(nu + self.n_dim) / 2)
        )


if __name__ == "__main__":
    N = 100
    np.random.seed(12)
    inv_wi = invwishart(4, 12 * np.eye(2))
    data = []
    for mu in ((-5, -5), (5, -2.5), (0, 5)):
        data.append(np.random.multivariate_normal(mu, inv_wi.rvs(), N))

    X = np.vstack(data)
    n_components = 3
    colors = ("red", "lime", "blue")
    # Variational Inference GMM
    vbgmm = VariationalInferenceGmm(n_components)
    ex_S = vbgmm.fit(X)
    S = np.argmax(ex_S, axis=1)
    cmap_S = np.array(colors)[S]
    res = 200
    ran = 15
    x = np.linspace(-ran, ran, res)
    y = np.linspace(ran, -ran, res)
    xx, yy = np.meshgrid(x, y)
    Y = np.vstack((xx.ravel(), yy.ravel())).T

    X_lh, ex_S2 = vbgmm.score_samples(Y)
    ex_S2 = ex_S2.reshape(res, res, n_components)
    X_lh = X_lh.reshape(res, res)

    vmin, vmax = 0, 0.01
    plt.figure(figsize=(8, 8), dpi=200)

    # plot 1
    plt.subplot(2, 2, 1)
    plt.title(r"VIGMM $p(x_{*} | X, S)$")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.imshow(
        X_lh,
        interpolation="bilinear",
        cmap="jet",
        extent=[-ran, ran, -ran, ran],
        vmin=vmin,
        vmax=vmax,
    )
    plt.scatter(
        X[:, 0], X[:, 1], c="white", s=10, marker="o", edgecolors="black", lw=0.4, alpha=0.5
    )

    # plot 2
    plt.subplot(2, 2, 2)
    plt.title(r"VIGMM $\langle s_* \rangle_{p(s_* | x_*, X, S)}$")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.imshow(ex_S2, interpolation="bilinear", extent=[-ran, ran, -ran, ran], alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=cmap_S, s=10, marker="o", edgecolors="black", lw=0.4)

    # Normal GMM
    gmm = GaussianMixture(n_components)
    gmm.fit(X)
    Z1 = np.exp(gmm.score_samples(Y)).reshape(res, res)
    Z2 = gmm.predict(Y)
    S3 = np.zeros((res * res, n_components))
    for i in range(res * res):
        S3[i][Z2[i]] = 1
    S3 = S3.reshape(res, res, n_components)
    cmap_Z2 = np.array(colors)[Z2]
    Z3 = gmm.predict(X)
    cmap_Z3 = np.array(colors)[Z3]

    # plot 3
    plt.subplot(2, 2, 3)
    plt.title(r"GMM $p(x_{*} | X, S)$")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.imshow(
        Z1,
        interpolation="bilinear",
        cmap="jet",
        extent=[-ran, ran, -ran, ran],
        vmin=vmin,
        vmax=vmax,
    )
    plt.scatter(
        X[:, 0], X[:, 1], c="white", s=10, marker="o", edgecolors="black", lw=0.4, alpha=0.5
    )

    # plot 4
    plt.subplot(2, 2, 4)
    plt.title(r"GMM $p(s_* | x_*, X, S)$")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.imshow(S3, interpolation="bilinear", extent=[-ran, ran, -ran, ran], alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=cmap_Z3, s=10, marker="o", edgecolors="black", lw=0.4)

    plt.tight_layout()
    plt.savefig("result.png")
    # plt.show()
