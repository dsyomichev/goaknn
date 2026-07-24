import sys

from pathlib import Path
from sqlite3 import Connection

# Script Configuration

ROOT_DIRECTORY = Path(__file__, "../..").resolve()

BATCH_SIZE = 20000

# Import Local Modules

sys.path.insert(0, str(ROOT_DIRECTORY))

from lib.execute import execute
from lib.preprocess import tokenize, index

# SQL Queries

CREATE_FREQUENCIES_TABLE = """
    CREATE TABLE IF NOT EXISTS frequencies (
        token TEXT NOT NULL,
        pmid INTEGER NOT NULL,
        frequency INTEGER NOT NULL,
        PRIMARY KEY (token, pmid)
    );
"""

SELECT_AVAILABLE_DOCUMENTS = """
    SELECT pmid, body FROM documents
    WHERE body IS NOT NULL
"""

DELETE_STALE_FREQUENCIES = """
    DELETE FROM frequencies
    WHERE pmid NOT IN (SELECT DISTINCT pmid FROM documents)
"""

INSERT_FREQUENCY = """
    INSERT INTO frequencies (token, pmid, frequency)
    VALUES (?, ?, ?)
    ON CONFLICT (token, pmid) DO UPDATE SET
        frequency = excluded.frequency
"""


def compute(connection: Connection):

    connection.executescript(CREATE_FREQUENCIES_TABLE)

    available = connection.execute(SELECT_AVAILABLE_DOCUMENTS)

    aggregate: list[tuple[str, int, int]] = []

    for pmid, body in available:

        tokens = tokenize(body)
        aggregate.extend([(token, pmid, frequency) for token, frequency in index(tokens)])

        if len(aggregate) >= BATCH_SIZE:
            connection.executemany(INSERT_FREQUENCY, aggregate)

            connection.commit()
            aggregate.clear()

    if aggregate:
        connection.executemany(INSERT_FREQUENCY, aggregate)

    connection.execute(DELETE_STALE_FREQUENCIES)

    connection.commit()


if __name__ == "__main__":
    execute(compute)
