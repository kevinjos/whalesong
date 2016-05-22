import psycopg2
from collections import OrderedDict
import re


class Location(object):
    OSL = "orcasoundlab"
    PTE = "porttownsend"
    LK = "limekiln"
    UNKNOWN = "unknown"
    URL = {OSL: "http://208.80.53.110:17636",
           PTE: "http://sc3.spacialnet.com:16948",
           LK: "http://ice.stream101.com:8047"}
    STOP = {OSL: "oslstop",
            PTE: "ptestop",
            LK: "lkstop"}

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
                          ("sample_number", "int"),
                          ("sample_duration", "int"),
                          ("predicted_class", "int"),
                          ("actual_class", "int")])
    NAME = "train"


class Model(Table):
    SCHEMA = OrderedDict([("id", "serial"),
                          ("created", "timestamp DEFAULT now()"),
                          ("modified", "timestamp DEFAULT now()"),
                          ("pmml", "xml")])
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


def withcursor(f):
    def wrapper(*args):
        cur = args[0].conn.cursor()
        f(args[0], cur)
        cur.close()
    return wrapper


class WhaleSongDB(object):
    HOST = "localhost"
    PORT = "5432"
    USER = "whalesong"
    DBNM = "whalesong"
    TABLES = [Train(), Model(), Rec()]

    def __enter__(self):
        self.conn = psycopg2.connect(database=self.DBNM,
                                     user=self.USER,
                                     host=self.HOST,
                                     port=self.PORT)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.conn.close()
