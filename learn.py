#!/usr/bin/env python

import numpy as np
from core import WhaleSongDB, ModelService, TrainService, RecService, Paths, read_wav, num_chans
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import pandas
import sklearn_pandas
from sklearn2pmml import sklearn2pmml


def fft(data, chans=1, rate=44100, size=1):
    """
    returns : [[[fft_s0_c0],[fft_s1_c0],...,[fft_sn_c0]], ..., [[fft_s0_cn],[fft_s1_cn],...,[fft_sn_cn]]]
    shape : (chans, samples, 1 + rate*size/2)
    """
    result = np.empty([chans, len(data) // (rate * size), 1 + rate * size // 2], dtype='float64')
    for chan in range(chans):
        for sample, (start, stop) in enumerate(zip(range(0, len(data), rate * size), range(rate * size, len(data) + 1, rate * size))):
            f_data = np.fft.fft(data[start:stop, chan])
            result[chan, sample] = np.abs(f_data[:1 + len(f_data) // 2])
    return result


def low_pass(data, freq_thresh, size=1):
    return data[:, :, :freq_thresh * size]


def avg_chans(data):
    return data[0, :, :]


def main():
    FFT_THRESH = 4400
    rec_ids = trainService.get_all_rec_ids()
    fns_rec_ids = recService.get_fn_rec_id_by_rec_ids(ids=rec_ids)
    model_id = modelService.get_id_by_model_name(name="human")
    X, Y = np.empty((0, FFT_THRESH), dtype="float32"), np.empty((0), dtype="int")
    for fn, rec_id in fns_rec_ids:
        rate, data = read_wav(Paths.TRAIN_DATA + fn)
        chans = num_chans(data)
        fft_data = fft(data, chans=chans, rate=rate)
        fft_data = low_pass(fft_data, FFT_THRESH)
        fft_data = avg_chans(fft_data)
        X = np.concatenate([X, fft_data])
        classifications = trainService.get_classification_by_model_id_rec_id(model_id=model_id, rec_id=rec_id)
        Y = np.concatenate([Y, classifications])
    colnames = [str(x) + "hz" for x in range(FFT_THRESH)]
    df = pandas.concat((pandas.DataFrame(X, columns=colnames), pandas.DataFrame(Y, columns=["label"])), axis=1)
    df_mapper = sklearn_pandas.DataFrameMapper([(colnames, StandardScaler()), (['label'], None), ])
    X = df_mapper.fit_transform(df)
    X, Y = X[:, :FFT_THRESH], X[:, FFT_THRESH:]
    clf = MLPClassifier(algorithm='adam',
                        alpha=1e-5,
                        hidden_layer_sizes=(1100, 550, ),
                        activation='tanh',
                        learning_rate='adaptive',
                        max_iter=100,
                        verbose=True,
                        early_stopping=True)
    clf.fit(X, Y)
    sklearn2pmml(clf, df_mapper, Paths.MODEL_DATA + "test.xml", with_repr=True)


if __name__ == '__main__':
    with WhaleSongDB() as db:
        modelService = ModelService(db.conn)
        trainService = TrainService(db.conn)
        recService = RecService(db.conn)
        main()
