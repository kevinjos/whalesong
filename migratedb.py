import argparse
import datetime as dt
import os
from scipy.io import wavfile

from core import transactional, WhaleSongDB, Location, Paths


def get_date_time_from_fn(fn):
    date_time = None
    try:
        date_time = dt.datetime.strptime(fn[:11], "%y%m%d_%H%M")
    except ValueError:
        try:
            date_time = dt.datetime.strptime(fn[:11], "%y%m%d-%H%M")
        except ValueError:
            try:
                date_time = dt.datetime.strptime(fn[:6], "%y%m%d")
            except ValueError:
                date_time = None
    return date_time


def get_duration(fn):
    dur = -1
    try:
        rate, data = wavfile.read(Paths.TRAIN_DATA + fn)
        dur = len(data) // rate
    except Exception:
        print("Failed to read wav for fn=[%s]" % fn)
    return dur


class Migrate(WhaleSongDB):
    @transactional
    def truncate(self, cur):
        for table in self.TABLES:
            cur.execute("DROP TABLE %s" % table.get_name())

    @transactional
    def create_tables(self, cur):
        for table in self.TABLES:
            cts = "CREATE TABLE %s (\n" % table.get_name()
            for k, v in table.get_schema().items():
                cts += "%s %s,\n" % (k, v)
            cidx = cts.rfind(",")
            cts = cts[:cidx] + cts[cidx + 1:]
            cts += ");"
            cur.execute(cts)

    @transactional
    def create_triggers(self, cur=None):
        cur.execute("""
        CREATE OR REPLACE FUNCTION update_modified_column()
        RETURNS TRIGGER AS $$
        BEGIN
            NEW.modified = now();
                RETURN NEW;
                END;
                $$ language 'plpgsql';
        """)
        for table in self.TABLES:
            cur.execute("""
            CREATE TRIGGER update_customer_modtime
            BEFORE UPDATE ON %s
            FOR EACH ROW EXECUTE PROCEDURE update_modified_column();
            """ % table.get_name())

    @transactional
    def populate_rec(self, cur=None):
        for fn in os.listdir(Paths.TRAIN_DATA):
            cur.execute("""
            INSERT INTO rec (filename, start_time, location, duration)
            VALUES (%s, %s, %s, %s);
            """, (fn, get_date_time_from_fn(fn), Location.get_location_from_fn(fn), get_duration(fn)))

    @transactional
    def populate_model(self, cur=None):
        cur.execute("""
        INSERT INTO model (filename)
        VALUES (%s);
        """, ('human',))


def main():
    with Migrate() as db:
        if args.all:
            db.truncate()
            db.create_tables()
            db.populate_rec()
            db.populate_model()
            db.create_triggers()
            return
        if args.rec:
            db.set_table("rec")
            db.truncate()
            db.create_tables()
            db.populate_rec()
            db.create_triggers()
        if args.model:
            db.set_table("model")
            db.truncate()
            db.create_tables()
            db.populate_model()
            db.create_triggers()
        if args.train:
            db.set_table("train")
            db.truncate()
            db.create_tables()
            db.create_triggers()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Migrate some data.')
    parser.add_argument('--rec', help='migrate rec table', action='store_true', default=False)
    parser.add_argument('--model', help='migrate model table', action='store_true', default=False)
    parser.add_argument('--train', help='migrate train table', action='store_true', default=False)
    parser.add_argument('--all', help='migrate all tables', action='store_true', default=False)
    args = parser.parse_args()
    main()
