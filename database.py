import sqlite3


class Database:

    def __init__(self):
        self._connection = sqlite3.connect('main.db')

        with self._connection as cur:
            cur.execute('''
                CREATE TABLE IF NOT EXISTS driver
                (
                    id integer primary key,
                    timestamp timestamp default current_timestamp not null,
                    speed float not null,
                    path path not null,
                    checked boolean default false not null,
                    plate varchar(10) default null
                )
           ''')

    @property
    def connection(self):
        return self._connection


db = Database()
