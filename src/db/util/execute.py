import sqlite3

from pathlib import Path

ROOT_DIRECTORY = Path(__file__, "../../..").resolve()
DB_FILE = ROOT_DIRECTORY.joinpath("/data/database.db")


def execute(proxy, **kwargs):
    connection = sqlite3.connect(DB_FILE)

    try:
        proxy(connection, **kwargs)
    finally:
        connection.close()
