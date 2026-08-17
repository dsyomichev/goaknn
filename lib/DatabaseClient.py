import numpy
import itertools

from numpy import float64, int_
from scipy.sparse import coo_array

BATCH_SIZE = 20000

SELECT_CITATIONS = """
    SELECT id, content FROM corpus
    WHERE content IS NOT NULL
    ORDER BY id ASC
"""

SELECT_ANNOTATIONS = """
    SELECT citation_id, term_id FROM annotations
    WHERE citation_id in (
        SELECT id FROM corpus
        WHERE content IS NOT NULL
    )
    ORDER BY citation_id ASC
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
    ON CONFLICT (token, position) DO UPDATE
    SET value = excluded.value
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
        PRIMARY KEY (id, position)
    )
"""

INSERT_VECTORIZATION = """
    INSERT INTO vectorizations (id, position, value)
    VALUES (?, ?, ?)
    ON CONFLICT (id, position) DO UPDATE
    SET value = excluded.value
"""

SELECT_VECTORIZATIONS = """
    SELECT id, position, value FROM vectorizations
    ORDER BY id ASC
"""


class DatabaseClient:

    def __init__(self, connection):
        self.connection = connection

    def get_citations(self):
        citations = self.connection.execute(SELECT_CITATIONS)

        contents, ids = [], []

        for id, content in citations:
            contents.append(content)
            ids.append(id)

        return contents, numpy.array(ids, dtype=int_)

    def get_annotations(self):
        citation_ids, term_ids = [], []

        annotations = self.connection.execute(SELECT_ANNOTATIONS)

        for citation_id, group in itertools.groupby(annotations, key=lambda item: item[0]):
            annotation = []

            for _, term_id in group:
                annotation.append(term_id)

            citation_ids.append(citation_id)
            term_ids.append(numpy.array(annotation, dtype=int_))

        return numpy.array(citation_ids, dtype=int_), term_ids

    def get_vocabulary(self):
        vocabulary = {}
        idf = []

        for token, position, value in self.connection.execute(SELECT_TOKENS):
            vocabulary[token] = position
            idf.append(value)

        return vocabulary, numpy.array(idf, dtype=float64)

    def set_vocabulary(self, vocabulary, idf):
        self.connection.execute(DROP_VOCABULARY_TABLE)
        self.connection.execute(CREATE_VOCABULARY_TABLE)

        batch = []

        for token, position in vocabulary.items():
            batch.append((token, position, float(idf[position])))

            if len(batch) > BATCH_SIZE:
                self.connection.executemany(INSERT_TOKEN, batch)
                self.connection.commit()

                batch.clear()

        if batch:
            self.connection.executemany(INSERT_TOKEN, batch)
            self.connection.commit()

    def get_vectorizations(self):
        row = []
        col = []
        val = []
        citation_ids = []

        vectorizations = self.connection.execute(SELECT_VECTORIZATIONS)

        for i, (id, group) in enumerate(itertools.groupby(vectorizations, key=lambda item: item[0])):
            for _, position, value in group:
                row.append(i)
                col.append(position)
                val.append(value)

            citation_ids.append(id)

        count = self.connection.execute(SELECT_TOKEN_COUNT).fetchone()[0]
        citation_vectors = coo_array((val, (row, col)), dtype=float64, shape=(len(citation_ids), count)).tocsr()
        return citation_vectors, numpy.array(citation_ids, dtype=int_)

    def set_vectorizations(self, citation_vectors, citation_ids):
        self.connection.execute(DROP_VECTORIZATIONS_TABLE)
        self.connection.execute(CREATE_VECTORIZATIONS_TABLE)

        ijv = coo_array(citation_vectors)

        empty_idxs = numpy.setdiff1d(numpy.arange(citation_ids.shape[0]), ijv.row)
        nonzero = [(int(citation_ids[i]), int(j), float(v)) for i, j, v in zip(ijv.row, ijv.col, ijv.data)]
        missing = [(int(citation_ids[i]), 0, 0.0) for i in empty_idxs]

        batch = []

        for id, position, value in [*nonzero, *missing]:
            batch.append((id, position, value))

            if len(batch) >= BATCH_SIZE:
                self.connection.executemany(INSERT_VECTORIZATION, batch)
                self.connection.commit()

                batch.clear()

        if batch:
            self.connection.executemany(INSERT_VECTORIZATION, batch)
            self.connection.commit()
