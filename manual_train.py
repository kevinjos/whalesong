from core import Paths, read_wav, num_chans, WhaleSongDB, ModelService, RecService, TrainService
import matplotlib.pyplot as plt
from pylab import rcParams
rcParams['figure.figsize'] = 40, 15


def plot_spectrogram(chans, rate, data):
    if chans == 2:
        ax1 = plt.subplot(211)
        plt.specgram(data[:, 0], NFFT=rate, Fs=rate)
        plt.subplot(212, sharex=ax1)
        plt.specgram(data[:, 1], NFFT=rate, Fs=rate)
    else:
        plt.specgram(data, NFFT=rate, Fs=rate)
    plt.show(block=False)


def load_train_file(fn):
    rate, data = read_wav(Paths.TRAIN_DATA + fn)
    chans = num_chans(data)
    return chans, rate, data


def classify_samples(fn, chans, rate, data):
    model_id = modelService.get_id_by_model_name(name="human")
    rec_id = recService.get_id_by_file_name(fn=fn)
    rec_duration = recService.get_duration(rec_id=rec_id)
    sample_duration = 1
    prev_pos_sample_number = 0
    ok = True

    def true_negatives(s, e):
        for i in range(s, e):
            trainService.insert_classification(rec_id=rec_id,
                                               model_id=model_id,
                                               sample_number=i,
                                               sample_duration=sample_duration,
                                               predicted_class=0,
                                               actual_class=0)
    while ok:
        sample_number = input("Sample with whale vocalization: ")
        if sample_number == '':
            break
        sample_number = sample_number.split("-")
        sample_number = [int(x) for x in sample_number]
        if max(sample_number) * sample_duration > rec_duration:
            print("Sample number=[%s] is outside of recording duration for filename=[%s] duration=[%s]" % (sample_number, fn, rec_duration))
            continue
        true_negatives(prev_pos_sample_number, min(sample_number))
        for sn in range(min(sample_number), max(sample_number) + 1):
            trainService.insert_classification(rec_id=rec_id,
                                               model_id=model_id,
                                               sample_number=sn,
                                               sample_duration=sample_duration,
                                               predicted_class=1,
                                               actual_class=1)
        prev_pos_sample_number = max(sample_number) + 1
    completed_file = True if input("Remaining samples without whale vocalizations? [y/n]: ") == "y" else False
    if completed_file:
        true_negatives(prev_pos_sample_number + 1, rec_duration + 1)


def main():
    print("loaded manual classification module")
    ok = True
    while ok:
        fn = input("Filename for analysis: ")
        try:
            chans, rate, data = load_train_file(fn)
        except Exception as e:
            print("Failed to read filename=[%s] with exception=[%s]" % (fn, e))
            continue
        plot_spectrogram(chans, rate, data)
        classify_samples(fn, chans, rate, data)
        ok = True if input("Classify another file? [y/n]: ") == "y" else False

if __name__ == '__main__':
    with WhaleSongDB() as db:
        modelService = ModelService(db.conn)
        trainService = TrainService(db.conn)
        recService = RecService(db.conn)
        main()
