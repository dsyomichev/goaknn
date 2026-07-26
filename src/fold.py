import random

from sqlite3 import Connection

# Script Configuration

BATCH_SIZE = 20000
RANDOM_SEED = 0

# SQL Queries

DROP_FOLD_TABLE = """
    DROP TABLE IF EXISTS folds
"""

CREATE_FOLD_TABLE = """
    CREATE TABLE IF NOT EXISTS folds (
            pmid INTEGER PRIMARY KEY,
            fold INTEGER
        )
"""

SELECT_DISTINCT_DOCUMENTS = """
    SELECT DISTINCT pmid FROM documents
    WHERE body IS NOT NULL
    ORDER BY pmid
"""

INSERT_FOLD_ASSIGNMENT = """
    INSERT INTO folds (pmid, fold)
    VALUES (?, ?)
    ON CONFLICT (pmid)
    DO UPDATE SET
        fold = excluded.fold
"""

DROP_CORPUS_TABLE = """
    DROP TABLE IF EXISTS corpus
"""

CREATE_CORPUS_TABLE = """
    CREATE TABLE corpus AS
    SELECT token, pmid, frequency FROM frequencies
        WHERE token IN (
            SELECT token FROM frequencies
            WHERE pmid IN (
                SELECT pmid FROM folds
                WHERE fold != ?
            )
            GROUP BY token
            HAVING COUNT(DISTINCT pmid) > 5
        )
"""

CREATE_CORPUS_INDEX = """
    CREATE INDEX token_index ON corpus (token);
    CREATE INDEX pmid_index ON corpus (pmid);
"""


def assign(connection: Connection, k: int = 1):
    random.seed(RANDOM_SEED)

    connection.execute(DROP_FOLD_TABLE)
    connection.execute(CREATE_FOLD_TABLE)

    available = [e[0] for e in connection.execute(SELECT_DISTINCT_DOCUMENTS).fetchall()]

    random.shuffle(available)

    size = len(available) // k

    for i in range(k):
        split = available[i * size : (i + 1) * size]

        batch: list[tuple[int, int]] = []

        for pmid in split:
            batch.append((pmid, i))

            if len(batch) >= BATCH_SIZE:
                connection.executemany(INSERT_FOLD_ASSIGNMENT, batch)

                connection.commit()
                batch.clear()

        if len(batch) > 0:
            connection.executemany(INSERT_FOLD_ASSIGNMENT, batch)

        connection.commit()


def filter(connection: Connection, ignore: int = -1):
    connection.execute(DROP_CORPUS_TABLE)
    connection.execute(CREATE_CORPUS_TABLE, [ignore])
    connection.executescript(CREATE_CORPUS_INDEX)
