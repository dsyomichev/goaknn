import numpy

from sqlite3 import Connection
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import sparray, coo_array  # type: ignore
from numpy import ndarray, float64
from itertools import groupby

# Script Configuration

BATCH_SIZE = 20000

# SQL Queries

SELECT_CITATIONS = """
    SELECT id, content FROM corpus
    WHERE content IS NOT NULL
    ORDER BY id ASC
"""

DROP_VOCABULARY_TABLE = """
    DROP TABLE IF EXISTS vocabulary
"""

CREATE_VOCABULARY_TABLE = """
    CREATE TABLE IF NOT EXISTS vocabulary (
        token TEXT NOT NULL,
        position INTEGER NOT NULL,
        value FLOAT NOT NULL,
        PRIMARY KEY (token, position)
    )
"""

INSERT_TOKEN = """
    INSERT INTO vocabulary (token, position, value)
    VALUES (?, ?, ?)
    ON CONFLICT (token, position)
    DO UPDATE SET
        value = excluded.value
"""

SELECT_TOKENS = """
    SELECT token, position, value FROM vocabulary
    ORDER BY position ASC
"""

SELECT_TOKEN_COUNT = """
    SELECT COUNT(*) FROM vocabulary
"""

DROP_VECTORIZATIONS_TABLE = """
    DROP TABLE IF EXISTS vectorizations
"""

CREATE_VECTORIZATIONS_TABLE = """
    CREATE TABLE IF NOT EXISTS vectorizations (
        id INTEGER NOT NULL,
        position INTEGER NOT NULL,
        value FLOAT NOT NULL,
        PRIMARY KEY (id, position),
        FOREIGN KEY (id) REFERENCES corpus (id),
        FOREIGN KEY (position) REFERENCES vocabulary (position)

    )
"""

INSERT_VECTORIZATION = """
    INSERT INTO vectorizations (id, position, value)
    VALUES (?, ?, ?)
    ON CONFLICT (id, position)
    DO UPDATE SET
        value = excluded.value
"""

SELECT_VECTORIZATIONS = """
    SELECT id, position, value FROM vectorizations
    ORDER BY id ASC
"""


class DatabaseClient:

    def __init__(self, connection: Connection):
        self.connection = connection

    def get_citations(self):
        citations = self.connection.execute(SELECT_CITATIONS)

        X: list[str] = []
        y: list[int] = []

        for id, body in citations:
            X.append(body)
            y.append(id)

        return numpy.array(X), numpy.array(y)

    def get_vectorizer(self, vectorizer: TfidfVectorizer):
        vocabulary: dict[str, int] = {}
        idf: list[float64] = []

        for token, position, value in self.connection.execute(SELECT_TOKENS):
            vocabulary[token] = position
            idf.append(numpy.float64(value))

        vectorizer.vocabulary_ = vocabulary
        vectorizer.fixed_vocabulary_ = False
        vectorizer.idf_ = numpy.array(idf)

    def set_vectorizer(self, vectorizer: TfidfVectorizer):
        self.connection.execute(DROP_VOCABULARY_TABLE)
        self.connection.execute(CREATE_VOCABULARY_TABLE)

        batch: list[tuple[str, int, float]] = []

        for token, position in vectorizer.vocabulary_.items():  # type: ignore
            batch.append((token, position, vectorizer.idf_[position].item()))  # type: ignore

            if len(batch) > BATCH_SIZE:
                self.connection.executemany(INSERT_TOKEN, batch)

                self.connection.commit()
                batch.clear()

        if batch:
            self.connection.executemany(INSERT_TOKEN, batch)

            self.connection.commit()
            batch.clear()

    def get_vectorizations(self):
        row: list[int] = []
        col: list[int] = []
        data: list[float] = []

        y: list[int] = []

        cursor = self.connection.execute(SELECT_VECTORIZATIONS)

        for i, (id, group) in enumerate(groupby(cursor, key=lambda v: v[0])):
            for _, position, value in group:
                row.append(i)
                col.append(position)
                data.append(value)

            y.append(id)

        count = self.connection.execute(SELECT_TOKEN_COUNT).fetchone()[0]

        X_: sparray = coo_array((data, (row, col)), shape=(len(y), count)).tocsr()

        return X_, numpy.array(y)

    def set_vectorizations(self, X_: sparray, y: ndarray):
        self.connection.execute(DROP_VECTORIZATIONS_TABLE)
        self.connection.execute(CREATE_VECTORIZATIONS_TABLE)

        batch: list[tuple[int, int, float]] = []

        translation = coo_array(X_)

        for i, j, value in zip(translation.row, translation.col, translation.data):  # type: ignore
            batch.append((int(y[i]), int(j), float(value)))  # type: ignore

            if len(batch) >= BATCH_SIZE:
                self.connection.executemany(INSERT_VECTORIZATION, batch)

                self.connection.commit()
                batch.clear()

        if batch:
            self.connection.executemany(INSERT_VECTORIZATION, batch)

            self.connection.commit()
