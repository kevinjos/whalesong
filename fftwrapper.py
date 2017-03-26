import numpy as np


class FFT(object):
    def __init__(self, data, samprate, winsize, lowcut=None, highcut=None):
        self.data = data
        self.ndata = len(data)
        self.samprate = samprate
        self.winsize = winsize
        freqbins = np.fft.fftfreq(winsize, d=1.0 / samprate)
        self.maxfi = freqbins.argmax() if not highcut else (freqbins >= highcut).argmax()
        self.minfi = freqbins.argmin() if not lowcut else (freqbins >= lowcut).argmax()
        self.freqbins = freqbins[self.minfi:self.maxfi + 1]

    def run(self, norm=True, amponly=True):
        for start, stop in zip(range(0, self.ndata, self.winsize),
                               range(self.winsize, self.ndata + 1, self.winsize)):
            ampl, phase = self._fft(self.data[start:stop], amponly)
            if not norm:
                yield (ampl, phase)
            else:
                yield (self._norm(ampl), phase)

    def _fft(self, d, amponly):
        d = np.fft.fft(d)
        d = d[self.minfi:self.maxfi + 1]
        if not amponly:
            ampl, phase = np.power(np.abs(d), 2), np.angle(d)
        else:
            ampl, phase = np.power(np.abs(d), 2), None
        return (ampl, phase)

    def _norm(self, ampl):
        return ampl / sum(ampl)
