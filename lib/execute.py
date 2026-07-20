import sqlite3

from pathlib import Path

ROOT_DIRECTORY = Path(__file__, "../..").resolve()

DATABASE_FILE = Path(ROOT_DIRECTORY, "data/records.db").resolve()

INITIALIZE_DATABASE = """
    PRAGMA journal_mode = WAL;
    PRAGMA synchronous = NORMAL;
    PRAGMA cache_size = -65536;
"""


def execute(proxy, path=None, connection=None, **kwargs):
    close = False if connection is None else True

    if connection is None:
        connection = sqlite3.connect(DATABASE_FILE if path is None else path)

        connection.executescript(INITIALIZE_DATABASE)

    try:
        proxy(connection, **kwargs)

    finally:
        if close:
            connection.close()
