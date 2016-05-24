from core import Paths, read_wav, write_wav, truncate_data, num_chans
import os


def main():
    for fin in os.listdir(Paths.TRAIN_DATA):
        try:
            rate, data = read_wav(Paths.TRAIN_DATA + fin)
            chans = num_chans(data)
            data = truncate_data(data, chans, rate)
            write_wav(Paths.TRAIN_DATA + fin, rate, data)
        except Exception as e:
            print("Failed to clean file=[%s] with exception=[%s]" % (fin, e))


if __name__ == '__main__':
    main()
