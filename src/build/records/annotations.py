import gzip
import sqlite3

from sqlite3 import Connection
from pathlib import Path

ROOT_DIRECTORY = Path(__file__, "../../../..").resolve()
SOURCE_FILE = Path(ROOT_DIRECTORY, "data/source/goa_uniprot_all.gaf.gz").resolve()
DATABASE_FILE = Path(ROOT_DIRECTORY, "data/records.db").resolve()

BATCH_SIZE=20000

INITIALIZE_DATABASE = """
    PRAGMA journal_mode = WAL;
    PRAGMA synchronous = NORMAL;
    PRAGMA cache_size = -65536;
"""

CREATE_TABLE = """
    CREATE TABLE IF NOT EXISTS annotations (
        goid INTEGER NOT NULL,
        pmid INTEGER NOT NULL,
        date INTEGER NOT NULL,
        PRIMARY KEY (goid, pmid)
    );
"""

INSERT_MANY = """
    INSERT INTO annotations (goid, pmid, date)
    VALUES (?, ?, ?)
    ON CONFLICT (goid, pmid) DO UPDATE
    SET date = MIN(date, excluded.date)
"""

def read(path: Path):
    with gzip.open(path, "rt") as file:
        for line in file:
            if line.startswith("!"):
                continue

            values = line.rstrip().split("\t")

            goid = values[4]

            if not "GO:" in goid:
                continue

            pmid = values[5]

            if not pmid.startswith("PMID:"):
                continue

            date = values[13]

            yield (int(goid[-7:]), int(pmid[5:]), int(date))

def main():
    connection: Connection = sqlite3.connect(DATABASE_FILE)

    connection.executescript(INITIALIZE_DATABASE)
    connection.executescript(CREATE_TABLE)

    try:
        batch: list[tuple[int, int, int]] = []

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