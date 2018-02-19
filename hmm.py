# -*- coding: utf-8 -*-
import numpy as np
from itertools import product

import matplotlib.pyplot as plt


class HiddenMarkovModel:
    def __init__(self, n_states, n_outputs):
        self.n_states = n_states
        self.n_outputs = n_outputs

        self.trans_matrix = np.random.rand(n_states, n_states)
        self.trans_matrix /= np.sum(self.trans_matrix, axis=1)[:, np.newaxis]

        self.emission_prob = np.random.rand(n_states, n_outputs)
        self.emission_prob /= np.sum(self.emission_prob, axis=1)[:, np.newaxis]

        self.start_prob = np.random.rand(n_states)
        self.start_prob /= np.sum(self.start_prob)

    def set_params(self, start_prob=None, trans_matrix=None, emission_prob=None):
        if start_prob is not None:
            self.start_prob = start_prob

        if trans_matrix is not None:
            self.trans_matrix = trans_matrix

        if emission_prob is not None:
            self.emission_prob = emission_prob

    def __str_to_array(self, s):
        return np.array([int(c) for c in s])

    def __generate_sample_from_state(self, state):
        return (np.cumsum(self.emission_prob[state, :]) > np.random.rand()).argmax()

    def calc_prob_states_bruteforce(self, x):
        ''' bruteforce algorithm '''
        x_ = x
        if isinstance(x_, str):
            x_ = self.__str_to_array(x_)

        p_x = 0
        max_p_xs = 0
        max_s = ''
        for s in product(range(self.n_states), repeat=len(x_)):
            p_xs = self.start_prob[s[0]] * self.emission_prob[s[0], x_[0]]

            for i in range(1, len(x_)):
                p_xs *= self.trans_matrix[s[i-1], s[i]] * self.emission_prob[s[i], x_[i]]
            if p_xs > max_p_xs:
                max_s = ''.join([str(x) for x in s])
                max_p_xs = p_xs
            p_x += p_xs

        return p_x, max_s

    def calc_prob_forward(self, x):
        ''' forward algorithm '''
        x_ = x
        if isinstance(x_, str):
            x_ = self.__str_to_array(x_)

        alpha = self.__forward_algorithm(x_)
        p_x = np.sum(alpha[-1, :])

        return p_x

    def calc_prob_backward(self, x):
        ''' backward algorithm '''
        x_ = x
        if isinstance(x_, str):
            x_ = self.__str_to_array(x_)

        beta = self.__backward_algorithm(x_)
        p_x = np.sum(self.start_prob * self.emission_prob[:, x_[0]] * beta[0, :])

        return p_x

    def __forward_algorithm(self, x_):
        alpha = np.zeros((x_.shape[0], self.n_states))
        alpha[0, :] = self.start_prob * self.emission_prob[:, x_[0]]

        for i in range(1, x_.shape[0]):
            alpha[i, :] = alpha[i - 1, :] @ self.trans_matrix * self.emission_prob[:, x_[i]]

        return alpha

    def __backward_algorithm(self, x_):
        beta = np.zeros((x_.shape[0], self.n_states))
        beta[-1, :] = 1

        for i in range(x_.shape[0] - 2, -1, -1):
            beta[i, :] = (self.emission_prob[:, x_[i + 1]] * beta[i + 1, :]) @ self.trans_matrix.T

        return beta

    def predict_sequence(self, x):
        ''' vitabi algorithm '''
        x_ = x
        if isinstance(x_, str):
            x_ = self.__str_to_array(x_)

        psi = np.zeros((x_.shape[0], self.n_states))
        psi[0, :] = self.start_prob * self.emission_prob[:, x_[0]]

        Psi = np.zeros(psi.shape, dtype=np.int32)

        for i in range(1, x_.shape[0]):
            tmp_mat = np.tile(psi[i - 1, :], (self.n_states, 1)) * self.trans_matrix.T
            psi[i, :] = np.max(tmp_mat, axis=1) * self.emission_prob[:, x_[i]]
            Psi[i, :] = np.argmax(tmp_mat, axis=1)

        p_xs_star = np.max(psi[-1, :])
        s = np.zeros(x_.shape[0], dtype=np.int32)
        s[-1] = np.argmax(psi[-1, :])
        for i in range(x_.shape[0] - 2, -1, -1):
            s[i] = Psi[i + 1, s[i + 1]]

        return s, p_xs_star

    def sample(self, n_samples):
        state_sequence = []
        sample_sequence = []
        current_state = (np.cumsum(self.start_prob) > np.random.rand()).argmax()
        state_sequence.append(current_state)
        sample_sequence.append(self.__generate_sample_from_state(current_state))

        for i in range(1, n_samples):
            current_state = (np.cumsum(self.trans_matrix[current_state, :]) > np.random.rand()).argmax()
            state_sequence.append(current_state)
            sample_sequence.append(self.__generate_sample_from_state(current_state))

        return np.array(sample_sequence), np.array(state_sequence)

    def fit(self, x, n):
        x_ = x
        if isinstance(x_, str):
            x_ = self.__str_to_array(x_)

        f = lambda i, j: (self.trans_matrix[i, j] * np.sum(
            alpha[:-1, i] * beta[1:, j] * self.emission_prob[j, x_[1:]])) / (alpha[:-1, i] @ beta[:-1, i])

        delta = np.eye(self.n_outputs)
        g = lambda j, k: np.sum(delta[x_, k] * alpha[:, j] * beta[:, j]) / (alpha[:, j] @ beta[:, j])

        h = lambda i: (alpha[0, i] * beta[0, i]) / np.sum(alpha[-1, :])

        count = 0
        lh = []
        while True:
            count += 1
            prev_ll = 0
            alpha = self.__forward_algorithm(x_)
            beta = self.__backward_algorithm(x_)

            for i in range(self.trans_matrix.shape[0]):
                for j in range(self.trans_matrix.shape[1]):
                    self.trans_matrix[i, j] = f(i, j)

            for j in range(self.emission_prob.shape[0]):
                for k in range(self.emission_prob.shape[1]):
                    self.emission_prob[j, k] = g(j, k)

            for i in range(self.n_states):
                self.start_prob[i] = h(i)

            tmp = np.abs(prev_ll - np.log(self.calc_prob_forward(x_)))
            if tmp < 0.0001 or count == n:
                break
            else:
                prev_ll = np.log(self.calc_prob_forward(x_))
                print(tmp, np.log(self.calc_prob_forward(x_)))
                lh.append(prev_ll)

        return lh


def main():
    n_states = 3
    n_outputs = 2

    hmm = HiddenMarkovModel(n_states=n_states, n_outputs=n_outputs)

    trans_mat = np.array([[0.1, 0.7, 0.2], [0.2, 0.1, 0.7], [0.7, 0.2, 0.1]])
    emission_prob = np.array([[0.9, 0.1], [0.6, 0.4], [0.1, 0.9]])
    start_prob = np.array([1/3, 1/3, 1/3])

    hmm.set_params(start_prob, trans_mat, emission_prob)

    obs = '000111101011111'
    print(hmm.calc_prob_states_bruteforce(obs))
    print(hmm.calc_prob_forward(obs))
    print(hmm.calc_prob_backward(obs))
    # print('predict:', hmm.predict_sequence(obs))
    # print('sample_sequence:', hmm.sample(50))
    lh = hmm.fit(obs, 20)
    plt.plot(lh)
    plt.show()


if __name__ == '__main__':
    main()
