import numpy

from sqlite3 import Connection
from sklearn.feature_extraction.text import TfidfVectorizer

# Script Configuration

BATCH_SIZE = 20000

# SQL Queries

SELECT_DOCUMENTS = """
    SELECT pmid, body FROM documents
    WHERE body IS NOT NULL
    ORDER BY pmid
"""

DROP_VOCABULARY_TABLE = """
    DROP TABLE IF EXISTS vocabulary
"""

CREATE_VOCABULARY_TABLE = """
    CREATE TABLE IF NOT EXISTS vocabulary (
        token TEXT PRIMARY KEY,
        position INTEGER NOT NULL,
        value FLOAT NOT NULL
    )
"""

INSERT_TOKEN = """
    INSERT INTO vocabulary (token, position, value)
    VALUES (?, ?, ?)
    ON CONFLICT (token)
    DO UPDATE SET
        position = excluded.position,
        value = excluded.value
"""

SELECT_TOKENS = """
    SELECT token, position, value FROM vocabulary
    ORDER BY position
"""


class DatabaseClient:

    def __init__(self, connection: Connection):
        self.connection = connection

    def get_documents(self):
        documents = self.connection.execute(SELECT_DOCUMENTS)

        X: list[str] = []
        y: list[int] = []

        for pmid, body in documents:
            X.append(body)
            y.append(pmid)

        return numpy.array(X), numpy.array(y)

    def get_vectorizer(self, vectorizer: TfidfVectorizer):
        vocabulary: dict[str, int] = {}
        idf: list[numpy.float64] = []

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

        if len(batch) > 0:
            self.connection.executemany(INSERT_TOKEN, batch)

            self.connection.commit()
            batch.clear()
