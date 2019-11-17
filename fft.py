import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft
from time import perf_counter


def signal_gen(time, fs, hz_list=[1]):
    data_size = fs * time
    x = np.linspace(0, time, data_size)
    y = np.zeros(data_size)
    for k in hz_list:
        y += np.sin(2 * np.pi * k * x)
    return y


def ij_matrix(n):
    result = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            result[i, j] = result[j, i] = i * j

    return result


def dft_matrix(n):
    # Nで除算しない
    omega_n = np.exp(1j * 2 * np.pi / n)
    return omega_n ** -ij_matrix(n)


def my_dft(x: np.ndarray):
    n = x.size
    return x @ dft_matrix(n).T


def my_fft(x: np.ndarray):
    n = x.size
    if n == 1:
        return x[0]
    else:
        x1 = x[::2]
        x2 = x[1::2]
        y1 = my_fft(x1)
        y2 = np.exp(-1j * 2 * np.pi / n) ** np.arange(n // 2) * my_fft(x2)
        z1 = y1 + y2
        z2 = y1 - y2
        return np.hstack((z1, z2))


if __name__ == "__main__":
    fs = 2 ** 11
    nq = fs // 2
    time = 1
    s = signal_gen(time, fs, [1, 10, 100, 200, 300, 400])

    # scipy FFT
    start = perf_counter()
    S = fft(s)
    print(f"scipy fft: {perf_counter() - start}")

    # DFT
    start = perf_counter()
    T = my_dft(s)
    print(f"dft: {perf_counter() - start}")
    assert (np.abs(S - T) < 1e-6).all()

    # FFT
    start = perf_counter()
    U = my_fft(s)
    print(f"my fft: {perf_counter() - start}")
    assert (np.abs(S - U) < 1e-6).all()

    # graph plot
    plt.subplot(4, 1, 1)
    plt.title("waveform")
    x = np.linspace(0, fs, time * fs)
    plt.grid()
    plt.plot(x, s, c="blue")

    plt.subplot(4, 1, 2)
    plt.title("scipy FFT")
    x = np.linspace(0, nq, S.size)
    plt.grid()
    plt.plot(x, np.abs(S), c="red")

    plt.subplot(4, 1, 3)
    plt.title("my DFT")
    x = np.linspace(0, nq, T.size)
    plt.grid()
    plt.plot(x, np.abs(T), c="green")

    plt.subplot(4, 1, 4)
    plt.title("my FFT")
    x = np.linspace(0, nq, U.size)
    plt.grid()

    plt.plot(x, np.abs(U), c="orange")

    plt.xlabel("Hz")
    plt.tight_layout()
    plt.show()
