#!/usr/bin/env python

from fftwrapper import FFT
import numpy as np
from core import WhaleSongDB, ModelService, TrainService, RecService, Paths, read_wav
from sklearn.cross_validation import train_test_split

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D
import matplotlib.pyplot as plt


def makey(cls, winsize, samprate, y):
    """
    cls is [(start_ms_0, stop_ms_0, class_0), ..., (start_ms_n, stop_ms_n, class_n)]
    """
    mspersamp = 1000.0 * winsize / samprate
    i, j = 0, 0
    for start, stop, obs in cls:
        i, j = j, int(stop / mspersamp)
        y[i:j] = obs
    return y


def loadrawfft(winsize=2048, lowcut=2000, highcut=4000, amponly=True):
    rec_ids = trainService.get_all_rec_ids()
    fns_rec_ids = recService.get_fn_rec_id_by_rec_ids(ids=rec_ids)
    model_id = modelService.get_id_by_model_name(name="human")

    X, Y = [], []
    for fn, rec_id in fns_rec_ids:
        samprate, data = read_wav(Paths.TRAIN_DATA + fn)
        fft_data = FFT(data, samprate, winsize, lowcut=lowcut, highcut=highcut)
        for i, (ampl, phase) in enumerate(fft_data.run()):
            if not amponly:
                X.append(np.concatenate((ampl, phase)))
            else:
                X.append(ampl)
        cls = trainService.get_classification_by_model_id_rec_id(model_id=model_id, rec_id=rec_id)
        Y.extend(makey(cls, winsize, samprate, np.zeros(i + 1)))
        assert len(Y) == len(X), "sample and class date should be the same length"
    return (np.array(X), np.array(Y)), fft_data


def rollwindow(X, Y, by=1):
    Yout = np.zeros(Y.shape[0] - by)
    Xout = np.zeros((X.shape[0] - by, by, X.shape[1]))
    l = range(0, X.shape[0] - by)
    h = [x + by for x in l]
    for i, (si, ei) in enumerate(zip(l, h)):
        # Yout[i] = max(Y[si:ei])  # If there's a positive classification in the window at all then 1 else 0
        Yout[i] = 1.0 if sum(Y[si:ei]) / by > 0.2 else 0.0  # If there's 20% or more positive classifications in the window then 1 else 0
        Xout[i] = X[si:ei]
    return (Xout, Yout)


def main():
    (X, Y), fft_data = loadrawfft()
    X, Y = rollwindow(X, Y, by=80)
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.2, random_state=42)

    input_shape = (X.shape[1], X.shape[2])

    batch_size = 16
    epochs = 16

    model = Sequential()
    model.add(Conv1D(128, 3, activation='relu', input_shape=input_shape))
    model.add(Conv1D(128, 3, activation='relu'))
    model.add(MaxPooling1D(3))
    model.add(Conv1D(256, 3, activation='relu'))
    model.add(Conv1D(256, 3, activation='relu'))
    model.add(GlobalAveragePooling1D())
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    model.fit(Xtrain, Ytrain, batch_size=batch_size, epochs=epochs)

    Yp = model.predict(Xtest)
    for i, (ya, yp) in enumerate(zip(Ytest, Yp)):
        yp_ = 1 if yp > .5 else 0
        if yp_ != ya:
            print("for sample=[%s]\nprediction=[%s]\nactual=[%s]\n" % (i, yp, ya))

            d = Xtest[i].T
            fig, (ax1, ax2) = plt.subplots(1, 2, sharey=False)
            ax1.imshow(d, aspect='auto', origin='lower', interpolation='none',
                       extent=[0,
                               d.shape[0],
                               min(fft_data.freqbins),
                               max(fft_data.freqbins)])
            ax1.format_coord = fft_data.freqbins
            ax2.hist(d.flatten(), bins=100)
            plt.show()
            raw_input()
            plt.close('all')

    score = model.evaluate(Xtest, Ytest, batch_size=16)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    """
    clf = MLPClassifier(solver='adam',
                        alpha=1e-5,
                        hidden_layer_sizes=(100, 10),
                        activation='relu',
                        max_iter=100,
                        verbose=True,
                        tol=1E-7,
                        early_stopping=True)
    clf.fit(Xa, Y)

    model = Sequential()
    model.add(Conv1D(64, 3, activation='relu', input_shape=(seq_length, 100)))
    model.add(Conv1D(64, 3, activation='relu'))
    model.add(MaxPooling1D(3))
    model.add(Conv1D(128, 3, activation='relu'))
    model.add(Conv1D(128, 3, activation='relu'))
    model.add(GlobalAveragePooling1D())
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    model.fit(Xtrain, Ytrain, batch_size=batch_size, epochs=epochs)
    score = model.evaluate(x_test, y_test, batch_size=16)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    model = Sequential()
    model.add(Dense(512, input_dim=Xa.shape[1], activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    model.fit(Xtrain, Ytrain, batch_size=batch_size, epochs=epochs,
              verbose=1, validation_data=(Xtest, Ytest))
    """


if __name__ == '__main__':
    passwd = "whalesong"
    with WhaleSongDB(passwd) as db:
        modelService = ModelService(db.conn)
        trainService = TrainService(db.conn)
        recService = RecService(db.conn)
        main()
else:
    passwd = "whalesong"
    db = WhaleSongDB(passwd)
    modelService = ModelService(db.conn)
    trainService = TrainService(db.conn)
    recService = RecService(db.conn)

"""
d = Xa[:1000,:].T
fig, ax = plt.subplots()
ax.imshow(d, aspect='auto', origin='lower', interpolation='none',
          extent=[0, max(map(lambda x: x[1], cls)), min(fft_data.freqbins), max(fft_data.freqbins)])
ax.format_coord = fft_data.freqbins
plt.show()

l = range(0, Xa.shape[0], samprate/winsize)
h = [x + 5*samprate/winsize for x in l]
for si, ei in zip(l, h):
    d = Xa[si:ei,:].T
    d = np.log10(d)
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=False)
    ax1.imshow(d, aspect='auto', origin='lower', interpolation='none',
              extent=[0, max(map(lambda x: x[1], cls)), min(fft_data.freqbins), max(fft_data.freqbins)])
    ax1.format_coord = fft_data.freqbins
    ax2.hist(d.flatten(), bins=100)
    plt.show()
    raw_input()
    plt.close('all')
"""
