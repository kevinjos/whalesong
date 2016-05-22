import datetime as dt
import psycopg2
import os
from scipy.io import wavfile

from core import withcursor, WhaleSongDB, Location

TRAINDATAPATH = "./train/whale/wav/"


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
        rate, data = wavfile.read(TRAINDATAPATH + fn)
        dur = len(data) // rate
    except Exception:
        print("Failed to read wav for fn=[%s]" % fn)
    return dur


class Migrate(WhaleSongDB):
    @withcursor
    def truncate(self, cur):
        with self.conn:
            for table in self.TABLES:
                cur.execute("DROP TABLE %s" % table.get_name())

    @withcursor
    def create_tables(self, cur):
        with self.conn:
            for table in self.TABLES:
                cts = "CREATE TABLE %s (\n" % table.get_name()
                for k, v in table.get_schema().items():
                    cts += "%s %s,\n" % (k, v)
                cidx = cts.rfind(",")
                cts = cts[:cidx] + cts[cidx + 1:]
                cts += ");"
                cur.execute(cts)

    @withcursor
    def create_triggers(self, cur):
        with self.conn:
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

    @withcursor
    def populate_rec(self, cur):
        """
        filename, start_time, location, duration
        """
        with self.conn:
            for fn in os.listdir(TRAINDATAPATH):
                cur.execute("""
                INSERT INTO rec (filename, start_time, location, duration)
                VALUES (%s, %s, %s, %s)
                """, (fn, get_date_time_from_fn(fn), Location.get_location_from_fn(fn), get_duration(fn)))


def main():
    with Migrate() as db:
        try:
            db.truncate()
        except psycopg2.ProgrammingError:
            print("DB tables don't exist")
        db.create_tables()
        db.create_triggers()
        db.populate_rec()

if __name__ == '__main__':
    main()
