import numpy

from sqlite3 import Connection
from scipy.sparse import sparray, coo_array  # type: ignore
from numpy import float64, int32
from itertools import groupby
from numpy.typing import NDArray
from numpy.dtypes import StringDType

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

SELECT_ANNOTATION = """
    SELECT DISTINCT term_id FROM annotations
    WHERE citation_id = ?
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

        return numpy.array(X, dtype=StringDType()), numpy.array(y, dtype=int32)

    def get_annotations(self, y: NDArray[int32]):
        annotations: list[NDArray[int32]] = []

        for id in y:
            terms = self.connection.execute(SELECT_ANNOTATION, [id.item()])
            annotations.append(numpy.array([term_id for (term_id,) in terms], dtype=int32))

        return annotations

    def get_vocabulary(self):
        vocabulary: dict[str, int] = {}
        idf: list[float64] = []

        for token, position, value in self.connection.execute(SELECT_TOKENS):
            vocabulary[token] = position
            idf.append(float64(value))

        return vocabulary, numpy.array(idf, dtype=float64)

    def set_vocabulary(self, vocabulary: dict[str, int], idf: NDArray[float64]):
        self.connection.execute(DROP_VOCABULARY_TABLE)
        self.connection.execute(CREATE_VOCABULARY_TABLE)

        batch: list[tuple[str, int, float]] = []

        for token, position in vocabulary.items():  # type: ignore
            batch.append((token, position, idf[position].item()))  # type: ignore

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

        for i, (id, group) in enumerate(groupby(cursor, key=lambda item: item[0])):
            for _, position, value in group:
                row.append(i)
                col.append(position)
                data.append(value)

            y.append(id)

        count = self.connection.execute(SELECT_TOKEN_COUNT).fetchone()[0]

        return coo_array((data, (row, col)), dtype=float64, shape=(len(y), count)).tocsr(), numpy.array(y, dtype=int32)

    def set_vectorizations(self, X_: sparray, y: NDArray[int32]):
        self.connection.execute(DROP_VECTORIZATIONS_TABLE)
        self.connection.execute(CREATE_VECTORIZATIONS_TABLE)

        batch: list[tuple[int, int, float]] = []

        translation = coo_array(X_)

        for i, j, value in zip(translation.row, translation.col, translation.data):  # type: ignore
            batch.append((y[i].item(), j.item(), value.item()))  # type: ignore

            if len(batch) >= BATCH_SIZE:
                self.connection.executemany(INSERT_VECTORIZATION, batch)

                self.connection.commit()
                batch.clear()

        if batch:
            self.connection.executemany(INSERT_VECTORIZATION, batch)

            self.connection.commit()
