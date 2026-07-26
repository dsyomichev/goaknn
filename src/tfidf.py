import pickle

from sqlite3 import Connection

# Script Configuration

BATCH_SIZE = 20000
RANDOM_SEED = 0

# SQL Queries

SELECT_DOCUMENT_TOKEN_AGGREGATE = """
    SELECT SUM(frequency) FROM corpus
    WHERE pmid = ?
"""

SELECT_DOCUMENT_TOKEN_COUNT = """
    SELECT frequency FROM corpus
    WHERE pmid = ? AND token = ?
"""

SELECT_CORPUS_SIZE = """
    SELECT COUNT(DISTINCT pmid) FROM corpus
"""

SELECT_CORPUS_COUNT_WITH_TOKEN = """
    SELECT COUNT(DISTINCT pmid) FROM corpus
    WHERE token = ?
"""

SELECT_CORPUS_TOKENS = """
    SELECT DISTINCT(token) FROM corpus
    ORDER BY token ASC
"""

DROP_IDFS_TABLE = """
    DROP TABLE IF EXISTS idfs
"""

CREATE_IDFS_TABLE = """
    CREATE TABLE IF NOT EXISTS idfs (
        token TEXT PRIMARY KEY,
        idf FLOAT
    )
"""

INSERT_TOKEN_IDF = """
    INSERT INTO idfs (token, idf)
    VALUES (?, ?)
    ON CONFLICT (token)
    DO UPDATE SET
        idf = EXCLUDED.idf
"""

SELECT_CORPUS_PMIDS = """
    SELECT DISTINCT(pmid) FROM corpus
"""

DROP_VECTORIZATION_TABLE = """
    DROP TABLE IF EXISTS samples
"""

CREATE_VECTORIZATION_TABLE = """
    CREATE TABLE IF NOT EXISTS vectorizations (
        pmid INTEGER PRIMARY KEY,
        vector TEXT
    )
"""

INSERT_DOCUMENT_VECTORIZATION = """
    INSERT INTO vectorizations (pmid, vector)
    VALUES (?, ?)
    ON CONFLICT (pmid)
    DO UPDATE SET
        vector = EXCLUDED.vector
"""


def precompute(connection: Connection):

    connection.execute(DROP_IDFS_TABLE)
    connection.execute(CREATE_IDFS_TABLE)

    dictionary = connection.execute(SELECT_CORPUS_TOKENS)
    size = connection.execute(SELECT_CORPUS_SIZE).fetchone()[0]
    batch: list[tuple[str, float]] = []

    for (token,) in dictionary:
        count = connection.execute(SELECT_CORPUS_COUNT_WITH_TOKEN, [token]).fetchone()[0]
        idf = 0 if count == 0 or count is None else size / count

        batch.append((token, idf))

        if len(batch) > BATCH_SIZE:
            connection.executemany(INSERT_TOKEN_IDF, batch)

            connection.commit()
            batch.clear()

    if len(batch) > 0:
        connection.executemany(INSERT_TOKEN_IDF, batch)

        connection.commit()
        batch.clear()


def vectorize(connection: Connection):
    connection.execute(DROP_VECTORIZATION_TABLE)
    connection.execute(CREATE_VECTORIZATION_TABLE)

    dictionary = connection.execute(SELECT_CORPUS_TOKENS)
    pmids = connection.execute(SELECT_CORPUS_PMIDS)

    batch: list[tuple[int, bytes]] = []

    for (pmid,) in pmids:

        vector: list[float] = []
        aggregate = connection.execute(SELECT_DOCUMENT_TOKEN_AGGREGATE, [pmid]).fetchone()[0]

        for (token,) in dictionary:
            count = connection.execute(SELECT_DOCUMENT_TOKEN_COUNT, [pmid, token]).fetchone()

            tf = 0 if count == 0 or count is None else count[0] / aggregate

            vector.append(tf)

        serialized = pickle.dumps(vector)

        batch.append((pmid, serialized))

        if len(batch) > BATCH_SIZE:
            connection.executemany(INSERT_DOCUMENT_VECTORIZATION, batch)

            connection.commit()
            batch.clear()

    if len(batch) > 0:
        connection.executemany(INSERT_DOCUMENT_VECTORIZATION, batch)

        connection.commit()
        batch.clear()
