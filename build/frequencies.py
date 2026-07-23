import sys

from pathlib import Path

# Script Configuration

ROOT_DIRECTORY = Path(__file__, "../..").resolve()

BATCH_SIZE = 20000

# Import Local Modules

sys.path.insert(0, str(ROOT_DIRECTORY))

from lib.execute import execute
from lib.preprocess import tokenize, index

# SQL Queries

CREATE_TABLE = """
    CREATE TABLE IF NOT EXISTS frequencies (
        token TEXT NOT NULL,
        pmid INTEGER NOT NULL,
        frequency INTEGER NOT NULL,
        PRIMARY KEY (token, pmid)
    );
"""

SELECT_AVAILABLE = """
    SELECT pmid, body FROM documents
    WHERE body IS NOT NULL
"""

DELETE_STALE = """
    DELETE FROM frequencies
    WHERE pmid NOT IN (SELECT DISTINCT pmid FROM documents)
"""

INSERT_MANY = """
    INSERT INTO frequencies (token, pmid, frequency)
    VALUES (?, ?, ?)
    ON CONFLICT (token, pmid) DO UPDATE SET
        frequency = excluded.frequency
"""


def compute(connection):

    connection.executescript(CREATE_TABLE)

    available = connection.execute(SELECT_AVAILABLE)

    aggregate = []

    for pmid, body in available:

        tokens = tokenize(body)
        aggregate.extend([(token, pmid, frequency) for token, frequency in index(tokens)])

        if len(aggregate) >= BATCH_SIZE:
            connection.executemany(INSERT_MANY, aggregate)

            connection.commit()
            aggregate.clear()

    if aggregate:
        connection.executemany(INSERT_MANY, aggregate)

    connection.execute(DELETE_STALE)

    connection.commit()


if __name__ == "__main__":
    execute(compute)
