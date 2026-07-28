import sqlite3

from pathlib import Path
from sqlite3 import Connection
from typing import Callable, Any

ROOT_DIRECTORY = Path(__file__, "../../..").resolve()

DATABASE_FILE = ROOT_DIRECTORY.joinpath("data/base.db").resolve()

INITIALIZE_DATABASE = """
    PRAGMA journal_mode = WAL;
    PRAGMA synchronous = NORMAL;
    PRAGMA cache_size = -65536;
"""


def execute(proxy: Callable[[Connection], None], path: Path | None = None, connection: Connection | None = None, **kwargs: Any):
    close = False if connection is None else True

    if connection is None:
        path = DATABASE_FILE if path is None else path
        path.parent.mkdir(parents=True, exist_ok=True)

        connection = sqlite3.connect(path)

        connection.executescript(INITIALIZE_DATABASE)

    try:
        proxy(connection, **kwargs)

    finally:
        if close:
            connection.close()
