import json
import sqlite3

from sqlite3 import Connection
from pathlib import Path

ROOT_DIRECTORY = Path(__file__, "../../../..").resolve()
SOURCE_FILE = Path(ROOT_DIRECTORY, "data/source/go-basic.json").resolve()
DATABASE_FILE = Path(ROOT_DIRECTORY, "data/records.db").resolve()

BATCH_SIZE = 2000

INITIALIZE_DATABASE = """
    PRAGMA journal_mode = WAL;
    PRAGMA synchronous = NORMAL;
    PRAGMA cache_size = -65536;
"""

CREATE_TABLE = """
    CREATE TABLE IF NOT EXISTS ontology (
        goid INTEGER PRIMARY KEY,
        label TEXT NOT NULL
    );
"""

INSERT_MANY = """
    INSERT INTO ontology (goid, label)
    VALUES (?, ?)
    ON CONFLICT (goid) DO UPDATE
    SET label = excluded.label
"""


def read(path: Path):
    with open(path) as file:
        data = json.load(file)

        for graph in data.get("graphs"):
            for node in graph.get("nodes"):
                goid = node.get("id")

                if goid is None:
                    continue

                try:
                    goid = int(goid[-7:])
                except:
                    continue

                label = node.get("lbl")

                if label is None:
                    continue

                yield (goid, label)


def main():
    connection: Connection = sqlite3.connect(DATABASE_FILE)

    connection.executescript(INITIALIZE_DATABASE)
    connection.executescript(CREATE_TABLE)

    try:
        batch: list[tuple[int, str]] = []

        for row in read(SOURCE_FILE):
            batch.append(row)

            if len(batch) >= BATCH_SIZE:
                connection.executemany(INSERT_MANY, batch)
                batch.clear()

        if len(batch) > 0:
            connection.executemany(INSERT_MANY, batch)

        connection.commit()

    finally:
        connection.close()


if __name__ == "__main__":
    main()
