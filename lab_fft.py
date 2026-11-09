"""PASTE YOUR LAB FFT HERE.
Contract: fft(x) takes a 1-D array of length N (power of 2) and returns the N complex DFT values.
The code below is only a PLACEHOLDER radix-2 FFT so the scripts run before you paste yours.
Delete it and put your own function in, keeping the name `fft`."""
import numpy as np


def fft(x):
    x = np.asarray(x, dtype=complex)
    N = len(x)
    if N == 1:
        return x
    even, odd = fft(x[0::2]), fft(x[1::2])
    tw = np.exp(-2j*np.pi*np.arange(N//2)/N) * odd
    return np.concatenate([even + tw, even - tw])
