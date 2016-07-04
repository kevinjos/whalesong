#!/usr/bin/env python
import asyncio
from collections import deque
import datetime as dt
import logging
from multiprocessing import Pool
import numpy as np
import os
from scipy.io import wavfile
import subprocess
from subprocess import PIPE, DEVNULL
from core import Location


SECS = "1200"
LOG = logging.getLogger("aquire_and_align")
LOG.setLevel(logging.INFO)

MPG123STDERR = b"High Performance"


class Recording(object):

    FN = "data/{date}_{starttime}_{loc}_{secs}.{ft}"
    TRUEAUDIOTHRESHOLD = 5

    def __init__(self, samplerate, chans, location):
        self.samplerate = samplerate
        self.chans = chans
        self.location = location
        self.url = Location.get_url(location)
        self.stop_file = Location.get_stop_file(location)
        self.wav_data = deque([], 2)

    def set_file_name(self):
        date = dt.date.today().strftime("%y%m%d")
        starttime = dt.datetime.today().strftime("%H%M%S")
        fn = self.FN.format(loc=self.location, date=date, starttime=starttime, secs=SECS, ft="{ft}")
        self.fn_mp3 = fn.format(ft="mp3")
        self.fn_wav = fn.format(ft="wav")
        self.fn_cue = fn.format(ft="cue")

    def set_wav_data(self):
        rate, data = wavfile.read(self.fn_wav)
        if rate != self.samplerate:
            self.samplerate = rate
            LOG.error("wav file sample rate [%s] != predefined sample rate [%s] for [%s]" % (rate, self.samplerate, self.location))
        LOG.debug("Data len=[%s] BEFORE truncate" % len(data))
        data = self.truncate_data_(data)
        LOG.debug("Data len=[%s] AFTER truncate" % len(data))
        self.wav_data.append(data)

    def truncate_data_(self, data):
        if self.chans == 1:
            true_start_idx = self.find_true_start_(data)
            data = data[true_start_idx:]
            to_trunc = len(data) % self.samplerate
            data = data[to_trunc:]
        else:
            true_start_idx = self.find_true_start_multichan_(data)
            data = data[true_start_idx:]
            to_trunc = len(data) % self.samplerate
            data = data[to_trunc:]
        return data

    def find_true_start_(self, data):
        for idx, x in enumerate(data):
            if np.abs(x) > self.TRUEAUDIOTHRESHOLD:
                return idx

    def find_true_start_multichan_(self, data):
        for idx, x in enumerate(data):
            if np.abs(max(x)) > self.TRUEAUDIOTHRESHOLD:
                return idx

    def get_cur_wav(self):
        res = None
        if len(self.wav_data) == 1:
            res = self.wav_data[0]
        else:
            res = self.wav_data[1]
        return res

    def get_prev_wav(self):
        res = None
        if len(self.wav_data) == 2:
            res = self.wav_data[0]
        return res


class SeqAlign(object):

    MIN_ALIGN = 44100

    def __init__(self, Recording):
        self.seq_0 = Recording.get_prev_wav()
        self.seq_1 = Recording.get_cur_wav()
        self.fn = Recording.fn_wav
        self.samplerate = Recording.samplerate
        self.channels = Recording.chans

    def find_start_align(self, data_0_last_60, data_1_first_60):
        frame_1 = data_1_first_60[:self.MIN_ALIGN]
        for idx, x in enumerate(data_0_last_60):
            frame_0 = data_0_last_60[idx:idx + self.MIN_ALIGN]
            assert len(frame_0) == len(frame_1)
            if np.all(frame_1 == frame_0):
                LOG.info("Alignment beginning in file=[%s] at data_0=[%s] data_1=[%s]" % (self.fn, idx, 0))
                return (idx, 0)
            elif idx + self.MIN_ALIGN - 1 >= len(data_0_last_60):
                break
        LOG.info("Alignment not found for file=[%s]" % self.fn)

    def find_end_align(self, data_0_last_aligned, data_1_first_aligned):
        n_data_0 = len(data_0_last_aligned)
        if not np.all(data_0_last_aligned == data_1_first_aligned[:n_data_0]):
            LOG.error("Write more code to handle case where next stream rip does not have all of first stream rip aligned segment")
        return -1, n_data_0

    def align(self):
        if self.seq_0 is None:
            LOG.info("Skiping alignment for file=[%s]: No alignment required" % (self.fn))
            return -1
        LOG.info("Beginning alignment for file=[%s]" % (self.fn))

        data_0_last_60 = self.seq_0[(-1 * (60 * self.samplerate)):]
        data_1_first_60 = self.seq_1[:(60 * self.samplerate)]

        idx_0_start_align, idx_1_start_align = self.find_start_align(data_0_last_60, data_1_first_60)

        data_0_last_aligned = data_0_last_60[idx_0_start_align:]
        data_1_first_aligned = data_1_first_60[idx_1_start_align:]

        idx_0_end_align, idx_1_end_align = self.find_end_align(data_0_last_aligned, data_1_first_aligned)

        assert idx_0_end_align == -1
        assert idx_1_start_align == 0

        data_0_aligned = data_0_last_60[idx_0_start_align:]
        data_1_aligned = data_1_first_60[:idx_1_end_align]

        assert np.all(data_0_aligned == data_1_aligned)
        self.seq_1 = self.seq_1[idx_1_end_align + 1:]
        return 0

    def write(self):
        wavfile.write(self.fn, self.samplerate, self.seq_1)
        LOG.debug("Wrote aligned file=[%s]" % self.fn)


def do_align(A):
    retcode = A.align()
    if retcode == 0:
        A.write()


def capture_stream(recording):
    LOG.info("Capturing stream at [%s]" % recording.location)
    p = subprocess.Popen(["streamripper", recording.url, "-i", "--xs-none", "-l", SECS, "-a", recording.fn_mp3], stdout=DEVNULL, stderr=PIPE)
    _, stderr = p.communicate()
    if stderr != b'':
        LOG.error("[%s] [streamripper]: %s" % (recording.location, stderr.decode()))
        return 1
    p = subprocess.Popen(["mpg123", "-w", recording.fn_wav, recording.fn_mp3], stdout=DEVNULL, stderr=PIPE)
    _, stderr = p.communicate()
    if stderr != b'' and stderr.find(MPG123STDERR) == -1:
        LOG.error("[%s] [mpg123]: %s" % (recording.location, stderr.decode()))
        return 2
    p = subprocess.Popen(["rm", recording.fn_mp3, recording.fn_cue], stdout=DEVNULL, stderr=PIPE)
    _, stderr = p.communicate()
    if stderr != b'':
        LOG.error("[%s] [rm]: %s" % (recording.location, stderr.decode()))
        return 3
    return 0


async def release_lock():
    await asyncio.sleep(1)


async def stream_handle(recording, pool):
    LOG.info("Initializing stream handler for [%s]" % recording.location)
    ok = True
    while ok:
        if os.path.isfile(recording.stop_file):
            LOG.info("Found stopfile=[%s], stopping" % recording.stop_file)
            subprocess.call(["rm", recording.stop_file], stdout=DEVNULL, stderr=DEVNULL)
            ok = False
            continue
        recording.set_file_name()
        res = pool.apply_async(capture_stream, (recording, ))
        while not res.ready():
            await release_lock()
        retcode = res.get()
        if retcode > 0:
            continue
        recording.set_wav_data()
        LOG.info("Aligning current and previous wav files for [%s]" % recording.location)
        A = SeqAlign(recording)
        res = pool.apply_async(do_align, (A, ))
    LOG.info("Stopping stream handler for [%s]" % recording.location)


def init():
    formatter = logging.Formatter('[%(asctime)s] - [%(name)s] - [%(levelname)s] - %(message)s')
    fh = logging.FileHandler("log/aquire_and_align.log")
    fh.setFormatter(formatter)
    # sh = logging.StreamHandler()
    # sh.setFormatter(formatter)
    # LOG.addHandler(sh)
    LOG.addHandler(fh)


def main():
    init()
    OrcaSoundLab = Recording(44100, 1, Location.OSL)
    LimeKiln = Recording(22050, 1, Location.LK)
    PortTownsend = Recording(22050, 2, Location.PTE)
    King = Recording(44100, 1, Location.KING)

    loop = asyncio.get_event_loop()
    with Pool(processes=6) as pool:
        tasks = [
            asyncio.ensure_future(stream_handle(PortTownsend, pool)),
            asyncio.ensure_future(stream_handle(OrcaSoundLab, pool)),
            asyncio.ensure_future(stream_handle(King, pool)),
            asyncio.ensure_future(stream_handle(LimeKiln, pool))]
        loop.run_until_complete(asyncio.wait(tasks))
    loop.close()


if __name__ == '__main__':
    main()
