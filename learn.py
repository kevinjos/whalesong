#!/usr/bin/env python

import numpy as np
from scipy.io import wavfile


def fft(data, chans=1, rate=44100, size=1):
    """
    returns : [[[fft_s0_c0],[fft_s1_c0],...,[fft_sn_c0]], ..., [[fft_s0_cn],[fft_s1_cn],...,[fft_sn_cn]]]
    shape : (chans, samples, 1 + rate*size/2)
    """
    result = np.empty([chans, len(data) // (rate * size), 1 + rate*size//2], dtype='float64')
    for chan in range(chans):
        for sample, (start, stop) in enumerate(zip(range(0, len(data), rate*size), range(rate*size, len(data), rate*size))):
            f_data = np.fft.fft(data[start:stop, chan])
            result[chan, sample] = np.abs(f_data[:1+len(f_data)//2])
    return result


def find_true_start_(data, thresh=5):
    for idx, x in enumerate(data):
        if np.abs(x) > thresh:
            return idx


def find_true_start_multichan_(data, thresh=5):
    for idx, x in enumerate(data):
        if np.abs(max(x)) > thresh:
            return idx


def truncate_data(data, chans, rate):
    to_trunc = len(data) % rate
    data = data[to_trunc:]
    if chans == 1:
        data = data[find_true_start_(data):]
    else:
        data = data[find_true_start_multichan_(data):]
    return data


def plot_spectrogram(data):
    pass


root = "./train/whale/wav"
fn = "080630_224300-224900_000710_os_many-S2s-call-response-maybe.wav"
rate, data = wavfile.read("%s/%s" % (root, fn))
chans = data.shape[1]
data = truncate_data(data, chans, rate)
f_data = fft(data, chans=chans, rate=rate)

