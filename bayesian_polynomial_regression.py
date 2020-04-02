import numpy as np
import matplotlib.pyplot as plt


class BayesianPolynomialRegressor:
    def __init__(self, M, precision):
        self.M = M
        self.l = precision
        self.L = np.identity(M)
        self.m = np.zeros(M)

        self.L_hat_inv = None
        self.m_hat = None

    def fit(self, X, Y):
        X_ = np.vstack([self.__vec(x) for x in X])
        L_hat = self.l * X_.T @ X_ + self.L
        self.L_hat_inv = np.linalg.inv(L_hat)
        self.m_hat = (
            self.l * Y[np.newaxis, :] @ X_ + self.m[np.newaxis, :] @ self.L
        ) @ self.L_hat_inv
        return self

    def samples(self, X):
        return np.array([self.__sample(x) for x in X])

    def predict(self, X):
        return np.vstack([np.array(self.__pred(x)) for x in X]).T

    def __sample(self, x):
        return np.random.normal(*self.__pred(x))

    def __pred(self, x):
        vec_x = self.__vec(x)
        mu_star = np.dot(self.m_hat, vec_x)[0]
        l_star_inv = 1 / self.l + vec_x @ self.L_hat_inv @ vec_x.T
        return mu_star, l_star_inv

    def __vec(self, x):
        return np.array([x ** i for i in range(self.M)])


def plot_setting(title_s):
    plt.title(title_s)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.xlim(-2.5, 12.5)
    plt.ylim(-4, 4)


if __name__ == "__main__":
    np.random.seed(5)

    # number of samples
    n = 10

    # observation data
    precision = 10
    X = 10 * np.random.rand(n)
    Y = np.sin(X) + np.random.normal(0, 1 / precision, n)

    # bayesian polynomial regression
    M = 8  # order + 1
    reg = BayesianPolynomialRegressor(M, precision)
    reg.fit(X, Y)

    Z = np.linspace(-5, 15, 1000)
    W1 = reg.samples(Z)
    W2 = reg.predict(Z)

    plt.figure(figsize=(10, 4))
    # plot 1
    plt.subplot(1, 2, 1)
    plot_setting("Bayesian Polynomial Regression")

    plt.plot(Z, np.sin(Z), c="gray", lw=0.5)
    plt.scatter(X, Y, c="blue", s=10)
    plt.scatter(Z, W1, c="green", s=0.2)

    plt.plot(Z, W2[0], c="red", lw=1)
    plt.plot(Z, W2[0] + W2[1], c="green", lw=0.5, linestyle="dashed")
    plt.plot(Z, W2[0] - W2[1], c="green", lw=0.5, linestyle="dashed")
    plt.fill_between(Z, W2[0] + W2[1], W2[0] - W2[1], facecolor="green", alpha=0.3)

    # polynomial regression
    X_ = np.vstack(np.array([X ** i for i in range(M)]))
    w = Y @ X_.T @ np.linalg.inv(X_ @ X_.T)

    Z_ = np.vstack(np.array([Z ** i for i in range(M)]))
    W3 = w @ Z_

    # plot 2
    plt.subplot(1, 2, 2)
    plot_setting("Polynomial Regression")

    plt.plot(Z, np.sin(Z), c="gray", lw=0.5)
    plt.scatter(X, Y, c="blue", s=10)

    plt.plot(Z, W3, c="red", lw=1)

    plt.tight_layout()
    plt.show()
