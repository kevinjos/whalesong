from core import Paths, read_wav, num_chans, WhaleSongDB, ModelService, RecService, TrainService
import argparse


def load_train_file(fn):
    rate, data = read_wav(Paths.TRAIN_DATA + fn)
    chans = num_chans(data)
    return chans, rate, data


def classify_samples(fn, chans, rate, data):
    model_id = modelService.get_id_by_model_name(name="human")
    rec_id = recService.get_id_by_file_name(fn=fn)
    rec_duration = recService.get_duration(rec_id=rec_id)
    ok = True

    prev_stoptime = -1
    contin = True if input("Continuing classification? [y/n]: ") == "y" else False
    if contin:
        prev_stoptime = trainService.get_last_stoptime_by_fn(fn=fn)

    def true_negatives(starttime, stoptime):
        trainService.insert_classification(rec_id=rec_id,
                                           model_id=model_id,
                                           starttime_ms=starttime,
                                           stoptime_ms=stoptime,
                                           predicted_class=0,
                                           actual_class=0)
    while ok:
        starttime = input("Whale vocalization start time (ms): ")
        if starttime == '':
            break
        else:
            starttime = int(starttime)
        stoptime = input("Whale vocalization stop time (ms): ")
        stoptime = int(stoptime)
        true_negatives(prev_stoptime + 1, starttime - 1)
        prev_stoptime = stoptime
        trainService.insert_classification(rec_id=rec_id,
                                           model_id=model_id,
                                           starttime_ms=starttime,
                                           stoptime_ms=stoptime,
                                           predicted_class=1,
                                           actual_class=1)
    completed_file = True if input("Remaining samples without whale vocalizations? [y/n]: ") == "y" else False
    if completed_file:
        true_negatives(prev_stoptime + 1, rec_duration)


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
        classify_samples(fn, chans, rate, data)
        ok = True if input("Classify another file? [y/n]: ") == "y" else False

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Migrate some data.')
    parser.add_argument('--passwd', help='dbuser password', default="")
    args = parser.parse_args()
    with WhaleSongDB(args.passwd) as db:
        modelService = ModelService(db.conn)
        trainService = TrainService(db.conn)
        recService = RecService(db.conn)
        main()
