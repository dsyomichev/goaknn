from numpy import array, pad, iinfo, float64, int_
from numpy.ma import masked_equal
from itertools import groupby
from scipy.sparse import coo_array  # type: ignore

from sqlite3 import Connection
from scipy.sparse import sparray  # type: ignore
from numpy import ndarray, dtype
from numpy.ma import masked_array
from collections.abc import Iterable

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

SELECT_ANNOTATIONS = """
    SELECT citation_id, term_id FROM annotations
    ORDER BY citation_id ASC
"""


class DatabaseClient:

    def __init__(
        self,
        connection: Connection,
    ) -> None:
        self.connection = connection

    def get_citations(
        self,
    ) -> tuple[Iterable[str], ndarray[tuple[int], dtype[int_]]]:
        citations = self.connection.execute(SELECT_CITATIONS)

        X: list[str] = []
        y: list[int] = []

        for id, body in citations:
            X.append(body)
            y.append(id)

        return X, array(y, dtype=int_)

    def get_annotations(
        self,
    ) -> tuple[
        ndarray[tuple[int], dtype[int_]], masked_array[tuple[int, int], dtype[int_]]
    ]:
        y: list[int] = []
        Z: list[list[int]] = []  # type: ignore

        cursor = self.connection.execute(SELECT_ANNOTATIONS)

        width = 0

        for citation_id, group in groupby(cursor, key=lambda item: item[0]):
            z: list[int] = []

            for _, term_id in group:
                z.append(term_id)

            y.append(citation_id)
            Z.append(z)  # type: ignore

            width = len(z) if len(z) > width else width

        Z: list[ndarray[tuple[int], dtype[int_]]] = [pad(z, (0, width - len(z)), constant_values=iinfo(int_).min) for z in Z]  # type: ignore

        return array(y, dtype=int_), masked_equal(Z, iinfo(int_).min)

    def get_vocabulary(
        self,
    ) -> tuple[dict[str, int], ndarray[tuple[int], dtype[float64]]]:
        vocabulary: dict[str, int] = {}
        idf: list[float] = []

        for token, position, value in self.connection.execute(SELECT_TOKENS):
            vocabulary[token] = position
            idf.append(value)

        return vocabulary, array(idf, dtype=float64)

    def set_vocabulary(
        self,
        vocabulary: dict[str, int],
        idf: ndarray[tuple[int], dtype[float64]],
    ) -> None:
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

    def get_vectorizations(
        self,
    ) -> tuple[
        sparray,
        ndarray[tuple[int], dtype[int_]],
    ]:
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

        return coo_array(
            (data, (row, col)), dtype=float64, shape=(len(y), count)
        ).tocsr(), array(y, dtype=int_)

    def set_vectorizations(
        self,
        X_: sparray,
        y: ndarray[tuple[int], dtype[int_]],
    ) -> None:
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
