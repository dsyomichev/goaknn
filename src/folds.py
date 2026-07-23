import sys
import random

from pathlib import Path

# Script Configuration

ROOT_DIRECTORY = Path(__file__, "../..").resolve()

BATCH_SIZE = 20000

# Import Local Modules

sys.path.insert(0, str(ROOT_DIRECTORY))

from lib.execute import execute

# SQL Queries

CREATE_TABLE = """
    CREATE TABLE IF NOT EXISTS folds (
            pmid INTEGER PRIMARY KEY,
            fold INTEGER
        );
"""

SELECT_DISTINCT = """
    SELECT DISTINCT pmid FROM documents
    WHERE body IS NOT NULL
    ORDER BY pmid
"""

DELETE_MANY = """
    DELETE FROM folds
    WHERE pmid NOT IN (SELECT pmid from documents)
"""

INSERT_MANY = """
    INSERT INTO folds (pmid, fold)
    VALUES (?, ?)
    ON CONFLICT (pmid)
    DO UPDATE SET
        fold = excluded.fold
"""


def fold(connection, k=1, seed=0):
    random.seed(seed)

    connection.executescript(CREATE_TABLE)

    available = [int(e[0]) for e in connection.execute(SELECT_DISTINCT).fetchall()]

    random.shuffle(available)

    size = len(available) // k

    for i in range(k):
        split = available[i * size : (i + 1) * size]

        batch = []

        for pmid in split:
            batch.append((pmid, i))

            if len(batch) >= BATCH_SIZE:
                connection.executemany(INSERT_MANY, batch)

                connection.commit()
                batch.clear()

        if len(batch) > 0:
            connection.executemany(INSERT_MANY, batch)

        connection.execute(DELETE_MANY)

        connection.commit()


if __name__ == "__main__":
    execute(fold)
