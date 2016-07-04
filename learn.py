#!/usr/bin/env python

import numpy as np
from core import WhaleSongDB, ModelService, TrainService, RecService, Paths, read_wav, num_chans
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import pandas
import sklearn_pandas
from sklearn2pmml import sklearn2pmml


def fft_gen(data, rate, chans, window_size):
    """
    returns : [[[fft_s0_c0],[fft_s1_c0],...,[fft_sn_c0]], ..., [[fft_s0_cn],[fft_s1_cn],...,[fft_sn_cn]]]
    shape : (chans, samples, 1 + rate*size/2)
    """
    if chans == 1:
        return _fft(data, rate, chans, window_size)
    ampl = np.empty([chans, len(data) // window_size, 1 + window_size // 2], dtype='float')
    phase = np.empty([chans, len(data) // window_size, 1 + window_size // 2], dtype='float')
    for chan in range(chans):
        for sample, (start, stop) in enumerate(zip(range(0, len(data), window_size), range(window_size, len(data) + 1, window_size))):
            f_data = np.fft.fft(data[start:stop, chan])
            f_data = f_data[:1 + len(f_data) // 2]
            ampl[chan, sample] = np.abs(f_data)
            phase[chan, sample] = np.angle(f_data)
    return (ampl, phase)


def _fft(data, rate, chans, window_size):
    """
    returns : [[[fft_s0_c0],[fft_s1_c0],...,[fft_sn_c0]], ..., [[fft_s0_cn],[fft_s1_cn],...,[fft_sn_cn]]]
    shape : (chans, samples, 1 + rate*size/2)
    """
    ampl = np.empty([len(data) // window_size, 1 + window_size // 2], dtype='float')
    phase = np.empty([len(data) // window_size, 1 + window_size // 2], dtype='float')
    for sample, (start, stop) in enumerate(zip(range(0, len(data), window_size), range(window_size, len(data) + 1, window_size))):
        f_data = np.fft.fft(data[start:stop])
        f_data = f_data[:1 + len(f_data) // 2]
        ampl[sample] = np.abs(f_data)
        phase[sample] = np.angle(f_data)
    return (ampl, phase)


def avg_chans(data):
    return data[0, :, :]


def main():
    WINDOW_SIZE = 1024

    rec_ids = trainService.get_all_rec_ids()
    fns_rec_ids = recService.get_fn_rec_id_by_rec_ids(ids=rec_ids)
    model_id = modelService.get_id_by_model_name(name="human")

    X, Y = np.empty((0, WINDOW_SIZE * 2), dtype="float32"), np.empty((0), dtype="int")
    for fn, rec_id in fns_rec_ids:
        rate, data = read_wav(Paths.TRAIN_DATA + fn)
        chans = num_chans(data)
        ampl, phase = fft_gen(data, rate, chans, WINDOW_SIZE)
        ampl, phase = avg_chans(ampl), avg_chans(phase)
        fft_data = np.concatinate([ampl, phase])
        X = np.concatenate([X, fft_data])
        classifications = trainService.get_classification_by_model_id_rec_id(model_id=model_id, rec_id=rec_id)
        Y = np.concatenate([Y, classifications])
    # colnames = [str(x) + "hz" for x in range(FFT_THRESH)]
    colnames = [str(x) + "hz" for x in range(WINDOW_SIZE)]

    df = pandas.concat((pandas.DataFrame(X, columns=colnames), pandas.DataFrame(Y, columns=["label"])), axis=1)
    df_mapper = sklearn_pandas.DataFrameMapper([(colnames, StandardScaler()), (['label'], None), ])
    X = df_mapper.fit_transform(df)
    # X, Y = X[:, :FFT_THRESH], X[:, FFT_THRESH:]
    clf = MLPClassifier(algorithm='adam',
                        alpha=1e-5,
                        hidden_layer_sizes=(1100, 550, ),
                        activation='tanh',
                        learning_rate='adaptive',
                        max_iter=100,
                        verbose=True,
                        early_stopping=True)
    clf.fit(X, Y)
    # sklearn2pmml(clf, df_mapper, Paths.MODEL_DATA + "test.xml", with_repr=True)
    sklearn2pmml(clf, None, Paths.MODEL_DATA + "test.xml", with_repr=True)


if __name__ == '__main__':
    passwd = "whalesong"
    with WhaleSongDB(passwd) as db:
        modelService = ModelService(db.conn)
        trainService = TrainService(db.conn)
        recService = RecService(db.conn)
        main()
