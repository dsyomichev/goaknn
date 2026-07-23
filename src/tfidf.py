import sys
import random

from pathlib import Path

# Script Configuration

ROOT_DIRECTORY = Path(__file__, "../..").resolve()

BATCH_SIZE = 20000
RANDOM_SEED = 0

# Import Local Modules

sys.path.insert(0, str(ROOT_DIRECTORY))

from lib.execute import execute

# SQL Queries

CREATE_FOLD_TABLE = """
    CREATE TABLE IF NOT EXISTS folds (
            pmid INTEGER PRIMARY KEY,
            fold INTEGER
        );
"""

SELECT_DISTINCT_DOCUMENTS = """
    SELECT DISTINCT pmid FROM documents
    WHERE body IS NOT NULL
    ORDER BY pmid
"""

DELETE_FOLD_ASSIGNMENT = """
    DELETE FROM folds
    WHERE pmid NOT IN (SELECT pmid from documents)
"""

INSERT_FOLD_ASSIGNMENT = """
    INSERT INTO folds (pmid, fold)
    VALUES (?, ?)
    ON CONFLICT (pmid)
    DO UPDATE SET
        fold = excluded.fold
"""

SELECT_DISTINCT_FOLD_VOCABULARY = """
    SELECT DISTINCT token FROM frequencies
    WHERE pmid IN (
        SELECT pmid FROM folds
        WHERE fold = ?
    )
"""


def assign(connection, k=1):
    random.seed(RANDOM_SEED)

    connection.executescript(CREATE_FOLD_TABLE)

    available = [int(e[0]) for e in connection.execute(SELECT_DISTINCT_DOCUMENTS).fetchall()]

    random.shuffle(available)

    size = len(available) // k

    for i in range(k):
        split = available[i * size : (i + 1) * size]

        batch = []

        for pmid in split:
            batch.append((pmid, i))

            if len(batch) >= BATCH_SIZE:
                connection.executemany(INSERT_FOLD_ASSIGNMENT, batch)

                connection.commit()
                batch.clear()

        if len(batch) > 0:
            connection.executemany(INSERT_FOLD_ASSIGNMENT, batch)

        connection.execute(DELETE_FOLD_ASSIGNMENT)

        connection.commit()


def vectorize(connection, k=1):
    pass
