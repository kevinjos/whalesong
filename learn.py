#!/usr/bin/env python

import numpy as np


def fft(data, chans=1, rate=44100, size=1):
    """
    returns : [[[fft_s0_c0],[fft_s1_c0],...,[fft_sn_c0]], ..., [[fft_s0_cn],[fft_s1_cn],...,[fft_sn_cn]]]
    shape : (chans, samples, 1 + rate*size/2)
    """
    result = np.empty([chans, len(data) // (rate * size), 1 + rate * size // 2], dtype='float64')
    for chan in range(chans):
        for sample, (start, stop) in enumerate(zip(range(0, len(data), rate * size), range(rate * size, len(data), rate * size))):
            f_data = np.fft.fft(data[start:stop, chan])
            result[chan, sample] = np.abs(f_data[:1 + len(f_data) // 2])
    return result
