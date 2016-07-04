import psycopg2
from collections import OrderedDict
import itertools
import re
import numpy as np
from scipy.io import wavfile


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


def num_chans(data):
    s = data.shape
    if len(data.shape) == 1:
        return 1
    else:
        return s[1]


def read_wav(fn):
    rate, data = wavfile.read(fn)
    return rate, data


def write_wav(fn, rate, data):
    wavfile.write(fn, rate, data)


def transactional(f):
    def wrapper(*args, **kwargs):
        with args[0].conn:
            cur = args[0].conn.cursor()
            kwargs['cur'] = cur
            res = f(*args, **kwargs)
            cur.close()
        return res
    return wrapper


class Paths(object):
    TRAIN_DATA = "./train/whale/wav/"
    MODEL_DATA = "./models/"


class Location(object):
    OSL = "orcasoundlab"
    PTE = "porttownsend"
    LK = "limekiln"
    KING = "king"
    UNKNOWN = "unknown"
    URL = {OSL: "http://208.80.53.110:17636",
           PTE: "http://sc3.spacialnet.com:16948",
           LK: "http://ice.stream101.com:8047",
           KING: "http://echo.pacificwild.org:8000/King"}
    STOP = {OSL: "oslstop",
            PTE: "ptestop",
            LK: "lkstop",
            KING: "kingstop"}

    @classmethod
    def get_url(self, location):
        return self.URL[location]

    @classmethod
    def get_stop_file(self, location):
        return self.STOP[location]

    @classmethod
    def get_location_from_fn(self, fn):
        delims = "|".join(["_", "-", "\."])
        fn = re.split(delims, fn)
        if "os" in fn:
            return self.OSL
        elif "lk" in fn:
            return self.LK
        elif "pt" in fn:
            return self.PTE
        return self.UNKNOWN


class Table(object):
    SCHEMA = {}
    NAME = ""

    def get_schema(self):
        return self.SCHEMA

    def get_name(self):
        return self.NAME


class Train(Table):
    SCHEMA = OrderedDict([("id", "serial"),
                          ("created", "timestamp DEFAULT now()"),
                          ("modified", "timestamp DEFAULT now()"),
                          ("rec_id", "int"),
                          ("model_id", "int"),
                          ("starttime_ms", "int"),
                          ("stoptime_ms", "int"),
                          ("predicted_class", "int"),
                          ("actual_class", "int")])
    NAME = "train"


class Model(Table):
    SCHEMA = OrderedDict([("id", "serial"),
                          ("created", "timestamp DEFAULT now()"),
                          ("modified", "timestamp DEFAULT now()"),
                          ("filename", "varchar")])
    NAME = "model"


class Rec(Table):
    SCHEMA = OrderedDict([("id", "serial"),
                          ("created", "timestamp DEFAULT now()"),
                          ("modified", "timestamp DEFAULT now()"),
                          ("filename", "varchar"),
                          ("start_time", "timestamp"),
                          ("location", "varchar"),
                          ("duration", "int")])
    NAME = "rec"


class WhaleSongDB(object):
    HOST = "localhost"
    PORT = "5432"
    USER = "whalesong"
    DBNM = "whalesong"
    TABLES = [Train(), Model(), Rec()]

    def __init__(self, passwd):
        self.passwd = passwd
        self.__enter__()

    def __enter__(self):
        self.conn = psycopg2.connect(database=self.DBNM,
                                     user=self.USER,
                                     host=self.HOST,
                                     port=self.PORT,
                                     password=self.passwd)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.conn.close()

    def set_table(self, table):
        if table == "train":
            self.TABLES = [Train()]
        elif table == "rec":
            self.TABLES = [Rec()]
        elif table == "model":
            self.TABLES = [Model()]


class TrainService(Train):
    def __repr__(self):
        return "TrainService"

    def __init__(self, conn):
        self.conn = conn

    @transactional
    def insert_classification(self, cur=None, rec_id=None, model_id=None,
            starttime_ms=None, stoptime_ms=None, predicted_class=None,
            actual_class=None):
        cur.execute("""
        INSERT INTO train (rec_id, model_id, starttime_ms, stoptime_ms,
                           predicted_class, actual_class)
        VALUES (%s, %s, %s, %s, %s, %s);
        """, (rec_id, model_id, starttime_ms, stoptime_ms, predicted_class,
              actual_class,))

    @transactional
    def get_all_rec_ids(self, cur=None):
        cur.execute("""
        SELECT distinct rec_id FROM train;
        """)
        res = cur.fetchall()
        res = list(itertools.chain(*res))
        return res

    @transactional
    def get_classification_by_model_id_rec_id(self, cur=None, model_id=None, rec_id=None):
        cur.execute("""
        SELECT actual_class FROM train WHERE model_id = %s AND rec_id = %s
        """, (model_id, rec_id,))
        res = cur.fetchall()
        res = list(itertools.chain(*res))
        return res

    @transactional
    def get_last_stoptime_by_fn(self, cur=None, fn=None):
        cur.execute("""
        SELECT max(train.stoptime_ms) FROM train, rec WHERE train.rec_id = rec.id and rec.filename = %s
        """, (fn,))
        res = cur.fetchall()
        res = res[0][0]
        return res


class ModelService(Model):
    def __repr__(self):
        return "ModelService"

    def __init__(self, conn):
        self.conn = conn

    @transactional
    def get_id_by_model_name(self, cur=None, name=None):
        cur.execute("SELECT id FROM model WHERE filename = %s;", (name,))
        res = cur.fetchone()
        return res[0]

    @transactional
    def insert_model(self, cur=None, filename=None):
        cur.execute("""
        INSERT INTO model (filename)
        VALUES (%s);
        """, (filename,))


class RecService(Rec):
    def __repr__(self):
        return "RecService"

    def __init__(self, conn):
        self.conn = conn

    @transactional
    def get_id_by_file_name(self, cur=None, fn=None):
        cur.execute("SELECT id FROM rec WHERE filename = %s", (fn,))
        res = cur.fetchone()
        return res[0]

    @transactional
    def get_fn_rec_id_by_rec_ids(self, cur=None, ids=None):
        cur.execute("SELECT filename, id FROM rec WHERE id = ANY(%s);", (ids,))
        res = cur.fetchall()
        return res

    @transactional
    def get_duration(self, cur=None, rec_id=None):
        cur.execute("SELECT duration FROM rec WHERE id = %s", (rec_id,))
        res = cur.fetchone()
        return res[0]
