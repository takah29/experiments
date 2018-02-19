# coding: utf-8
import sys

import numpy as np
from scipy.io.wavfile import read
from scipy.signal import spectrogram, hanning
from scipy.linalg import norm
import matplotlib.pyplot as plt


def nmf(X, n_base=10, n_iters=10):
    H = np.random.rand(X.shape[0], n_base)
    U = np.random.rand(n_base, X.shape[1])
    for i in range(n_iters):
        H *= ((X @ U.T) / (H @ U @ U.T))
        U *= ((H.T @ X) / (H.T @ H @ U))
        print(i + 1, norm(X - H @ U))
    return H, U


def main():
    argvs = sys.argv
    argc = len(argvs)

    if argc < 2:
        sys.exit('Usage: python {} [wav file]'.format(argvs[0]))

    fft_width = 1024
    fft_olap = fft_width * 3 // 4

    wave_filepath = argvs[1]
    wav = read(wave_filepath)

    spec = spectrogram(wav[1], wav[0], window=hanning(fft_width),
                       noverlap=fft_olap, mode='magnitude')

    plt.subplot(2, 1, 1)
    plt.title('original', fontsize=10)
    plt.pcolormesh(spec[1], spec[0], np.log10(spec[2] + 1e-2))

    H, U = nmf(spec[2], 50, 100)
    Y = H @ U

    plt.subplot(2, 1, 2)
    plt.title('reconstructed', fontsize=10)
    plt.pcolormesh(spec[1], spec[0], np.log10(Y + 1e-2))

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
