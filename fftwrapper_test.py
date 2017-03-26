from fftwrapper import FFT
import numpy as np


nsamp = 8
samprate = np.power(2, 8)
d1hz = [np.sin(x) for x in np.linspace(0, 2 * np.pi * nsamp, num=samprate * nsamp)]
d2hz = [np.sin(2 * x) for x in np.linspace(0, 2 * np.pi * nsamp, num=samprate * nsamp)]


def test_basic_d1hz():
    winsize = np.power(2, 8)
    nout = winsize // 2
    majfreqbin, majfreqbini = 1, 1
    f = FFT(d1hz, samprate, winsize)
    assert f.freqbins[majfreqbini] == majfreqbin, "i=[%s], freqbin=[%s] expected=[%s]" % (majfreqbini, f.freqbins, majfreqbin)
    for i, (ampl, phase) in enumerate(f.run()):
        assert len(ampl) == len(phase) == nout == len(f.freqbins), "[%s] == [%s] == [%s] == [%s]" % (len(ampl), len(phase), nout, len(f.freqbins))
        assert ampl.argmax() == majfreqbini
    assert nsamp == i + 1, "nsamp [%s] != iterations [%s]" % (nsamp, i + 1)


def test_basic_d2hz():
    winsize = np.power(2, 8)
    nout = winsize // 2
    majfreqbin, majfreqbini = 2, 2
    f = FFT(d2hz, samprate, winsize)
    assert f.freqbins[majfreqbini] == majfreqbin, "i=[%s], freqbin=[%s] expected=[%s]" % (majfreqbini, f.freqbins, majfreqbin)
    for i, (ampl, phase) in enumerate(f.run()):
        assert len(ampl) == len(phase) == nout == len(f.freqbins), "[%s] == [%s] == [%s] == [%s]" % (len(ampl), len(phase), nout, len(f.freqbins))
        assert ampl.argmax() == majfreqbini
    assert nsamp == i + 1, "nsamp [%s] != iterations [%s]" % (nsamp, i + 1)


def test_smallwin_d1hz():
    winsize = np.power(2, 7)
    nout = winsize // 2
    majfreqbin, majfreqbini = 0, 0
    f = FFT(d1hz, samprate, winsize)
    assert f.freqbins[majfreqbini] == majfreqbin, "i=[%s], freqbin=[%s] expected=[%s]" % (majfreqbini, f.freqbins, majfreqbin)
    for i, (ampl, phase) in enumerate(f.run()):
        assert len(ampl) == len(phase) == nout == len(f.freqbins), "[%s] == [%s] == [%s] == [%s]" % (len(ampl), len(phase), nout, len(f.freqbins))
        assert ampl.argmax() == majfreqbini
    assert nsamp * 2 == i + 1, "nsamp [%s] != iterations [%s]" % (nsamp, i + 1)


def test_smallwin_d2hz():
    winsize = np.power(2, 7)
    nout = winsize // 2
    majfreqbin, majfreqbini = 2, 1
    f = FFT(d2hz, samprate, winsize)
    assert f.freqbins[majfreqbini] == majfreqbin, "i=[%s], freqbin=[%s] expected=[%s]" % (majfreqbini, f.freqbins, majfreqbin)
    for i, (ampl, phase) in enumerate(f.run()):
        assert len(ampl) == len(phase) == nout == len(f.freqbins), "[%s] == [%s] == [%s] == [%s]" % (len(ampl), len(phase), nout, len(f.freqbins))
        assert ampl.argmax() == majfreqbini
    assert nsamp * 2 == i + 1, "nsamp [%s] != iterations [%s]" % (nsamp, i + 1)


def test_largewin_d1hz():
    winsize = np.power(2, 9)
    nout = winsize // 2
    majfreqbin, majfreqbini = 1, 2
    f = FFT(d1hz, samprate, winsize)
    assert f.freqbins[majfreqbini] == majfreqbin, "i=[%s], freqbin=[%s] expected=[%s]" % (majfreqbini, f.freqbins, majfreqbin)
    for i, (ampl, phase) in enumerate(f.run()):
        assert len(ampl) == len(phase) == nout == len(f.freqbins), "[%s] == [%s] == [%s] == [%s]" % (len(ampl), len(phase), nout, len(f.freqbins))
        assert ampl.argmax() == majfreqbini
    assert nsamp / 2 == i + 1, "nsamp [%s] != iterations [%s]" % (nsamp, i + 1)


def test_largewin_d2hz():
    winsize = np.power(2, 9)
    nout = winsize // 2
    majfreqbin, majfreqbini = 2, 4
    f = FFT(d2hz, samprate, winsize)
    assert f.freqbins[majfreqbini] == majfreqbin, "i=[%s], freqbin=[%s] expected=[%s]" % (majfreqbini, f.freqbins, majfreqbin)
    for i, (ampl, phase) in enumerate(f.run()):
        assert len(ampl) == len(phase) == nout == len(f.freqbins), "[%s] == [%s] == [%s] == [%s]" % (len(ampl), len(phase), nout, len(f.freqbins))
        assert ampl.argmax() == majfreqbini
    assert nsamp / 2 == i + 1, "nsamp [%s] != iterations [%s]" % (nsamp, i + 1)


def test_basic_highcut_d1hz():
    winsize = np.power(2, 8)
    majfreqbin, majfreqbini = 1, 1
    highcut = 10
    nout = 11
    f = FFT(d1hz, samprate, winsize, highcut=highcut)
    assert f.freqbins[majfreqbini] == majfreqbin, "i=[%s], freqbin=[%s] expected=[%s]" % (majfreqbini, f.freqbins, majfreqbin)
    for i, (ampl, phase) in enumerate(f.run()):
        assert len(ampl) == len(phase) == nout == len(f.freqbins), "[%s] == [%s] == [%s] == [%s]" % (len(ampl), len(phase), nout, len(f.freqbins))
        assert ampl.argmax() == majfreqbini
    assert nsamp == i + 1, "nsamp [%s] != iterations [%s]" % (nsamp, i + 1)


def test_smallwin_highcut_d1hz():
    winsize = np.power(2, 7)
    majfreqbin, majfreqbini = 0, 0
    highcut = 10
    nout = 6
    f = FFT(d1hz, samprate, winsize, highcut=highcut)
    assert f.freqbins[majfreqbini] == majfreqbin, "i=[%s], freqbin=[%s] expected=[%s]" % (majfreqbini, f.freqbins, majfreqbin)
    for i, (ampl, phase) in enumerate(f.run()):
        assert len(ampl) == len(phase) == nout == len(f.freqbins), "[%s] == [%s] == [%s] == [%s]" % (len(ampl), len(phase), nout, len(f.freqbins))
        assert ampl.argmax() == majfreqbini
    assert nsamp * 2 == i + 1, "nsamp [%s] != iterations [%s]" % (nsamp, i + 1)


def test_largewin_highcut_d1hz():
    winsize = np.power(2, 9)
    majfreqbin, majfreqbini = 2, 4
    highcut = 10
    nout = 21
    f = FFT(d2hz, samprate, winsize, highcut=highcut)
    assert f.freqbins[majfreqbini] == majfreqbin, "i=[%s], freqbin=[%s] expected=[%s]" % (majfreqbini, f.freqbins, majfreqbin)
    for i, (ampl, phase) in enumerate(f.run()):
        assert len(ampl) == len(phase) == nout == len(f.freqbins), "[%s] == [%s] == [%s] == [%s]" % (len(ampl), len(phase), nout, len(f.freqbins))
        assert ampl.argmax() == majfreqbini, "[%s] != [%s]"
    assert nsamp / 2 == i + 1, "nsamp [%s] != iterations [%s]" % (nsamp, i + 1)
