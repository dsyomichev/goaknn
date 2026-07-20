import sqlite3
import sys

from pathlib import Path

ROOT_DIRECTORY = Path(__file__, "../../..").resolve()

sys.path.insert(0, str(ROOT_DIRECTORY))

from src.preprocess import preprocess

DATABASE_FILE = Path(ROOT_DIRECTORY, "data/records.db").resolve()

BATCH_SIZE = 2000

INITIALIZE_DATABASE = """
    PRAGMA journal_mode = WAL;
    PRAGMA synchronous  = NORMAL;
    PRAGMA cache_size   = -65536;
"""

CREATE_TABLE = """
    CREATE TABLE IF NOT EXISTS frequencies (
        term TEXT NOT NULL,
        pmid INTEGER NOT NULL,
        frequency INTEGER NOT NULL,
        PRIMARY KEY (term, pmid)
    );
"""

SELECT_INCOMPLETE = """
    SELECT pmid, body FROM documents
    WHERE body IS NOT NULL
"""

DELETE_STALE = """
    DELETE FROM frequencies
    WHERE pmid NOT IN (SELECT DISTINCT pmid FROM documents)
"""

INSERT_MANY = """
    INSERT INTO frequencies (term, pmid, frequency)
    VALUES (?, ?, ?)
    ON CONFLICT (term, pmid) DO UPDATE SET
        frequency = excluded.frequency
"""


def index(pmid, body):
    tokens = preprocess(body)

    frequencies = {}
    for token in tokens:
        frequencies[token] = frequencies.get(token, 0) + 1

    return [(term, pmid, frequency) for term, frequency in frequencies.items()]


def main(connection=None):
    close = False if connection is None else True

    if connection is None:
        connection = sqlite3.connect(DATABASE_FILE)

        connection.executescript(INITIALIZE_DATABASE)
        connection.executescript(CREATE_TABLE)

    try:
        incomplete = connection.execute(SELECT_INCOMPLETE)

        aggregate = []
        count = 0

        for pmid, body in incomplete:

            frequencies = index(pmid, body)
            aggregate.extend(frequencies)

            count += 1

            if count >= BATCH_SIZE:
                connection.executemany(INSERT_MANY, frequencies)

                frequencies.clear()

        if count >= 0:
            connection.executemany(INSERT_MANY, frequencies)

        connection.execute(DELETE_STALE)

        connection.commit()

    finally:
        if close:
            connection.close()


if __name__ == "__main__":
    main(None)
