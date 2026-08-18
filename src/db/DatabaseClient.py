import numpy
import itertools
import pickle

from scipy.sparse import coo_array

BATCH_SIZE = 20000

SELECT_CITATIONS = """
    SELECT id, content FROM corpus
    ORDER BY id ASC
"""

SELECT_ANNOTATIONS = """
    SELECT citation_id, term_id FROM annotations
    WHERE citation_id in (SELECT id FROM {})
    ORDER BY citation_id ASC
"""

SELECT_TERMS = """
    SELECT id, label FROM ontology
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
    ON CONFLICT (token, position) DO UPDATE
    SET value = excluded.value
"""

SELECT_TOKENS = """
    SELECT token, position, value FROM vocabulary
    ORDER BY position ASC
"""

SELECT_TOKEN_COUNT = """
    SELECT COUNT(token) FROM vocabulary
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

DROP_EMBEDDINGS_TABLE = """
    DROP TABLE IF EXISTS embeddings
"""

CREATE_EMBEDDINGS_TABLE = """
    CREATE TABLE IF NOT EXISTS embeddings (
        id INTEGER PRIMARY KEY
        data BLOB NOT NULL
    )
"""

INSERT_EMBEDDING = """
    INSERT INTO embeddings (id, data)
    VALUES (?, ?)
    ON CONFLICT (id) DO UPDATE
    SET data = excluded.data
"""

SELECT_EMBEDDINGS = """
    SELECT id, data FROM embeddings
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

        return contents, numpy.array(ids)

    def get_annotations(self, subset):
        citation_ids, term_ids = [], []

        annotations = self.connection.execute(SELECT_ANNOTATIONS.format(subset))

        for citation_id, group in itertools.groupby(annotations, key=lambda item: item[0]):
            annotation = []

            for _, term_id in group:
                annotation.append(term_id)

            citation_ids.append(citation_id)
            term_ids.append(numpy.array(annotation))

        return numpy.array(citation_ids), term_ids

    def get_terms(self):
        mapping = {}

        terms = self.connection.execute(SELECT_TERMS)

        for id, label in terms:
            mapping[id] = label

        return mapping

    def delete_vocabulary(self):
        self.connection.execute(DROP_VOCABULARY_TABLE)

    def get_vocabulary(self):
        vocabulary = {}
        idf = []

        for token, position, value in self.connection.execute(SELECT_TOKENS):
            vocabulary[token] = position
            idf.append(value)

        return vocabulary, numpy.array(idf)

    def set_vocabulary(self, vocabulary, idf):
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

    def delete_vectorizations(self):
        self.connection.execute(DROP_VECTORIZATIONS_TABLE)

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
        citation_vectors = coo_array((val, (row, col)), shape=(len(citation_ids), count)).tocsr()

        return citation_vectors, numpy.array(citation_ids)

    def set_vectorizations(self, citation_vectors, citation_ids):
        self.connection.execute(CREATE_VECTORIZATIONS_TABLE)

        ijv = coo_array(citation_vectors)

        batch = []

        for i, j, v in zip(ijv.row, ijv.col, ijv.data):
            batch.append((int(citation_ids[i]), int(j), float(v)))

            if len(batch) >= BATCH_SIZE:
                self.connection.executemany(INSERT_VECTORIZATION, batch)
                self.connection.commit()

                batch.clear()

        if batch:
            self.connection.executemany(INSERT_VECTORIZATION, batch)
            self.connection.commit()

    def delete_embeddings(self):
        self.connection.execute(DROP_EMBEDDINGS_TABLE)

    def get_embeddings(self):
        citation_ids, citation_embeds = [], []

        embeddings = self.connection.execute(SELECT_EMBEDDINGS)

        for id, data in embeddings:
            citation_ids.append(id)
            citation_embeds.append(pickle.loads(data))

        return numpy.array(citation_embeds), numpy.array(citation_ids)

    def set_embeddings(self, citation_embeds, citation_ids):
        self.connection.execute(CREATE_EMBEDDINGS_TABLE)

        batch = []

        for id, embed in zip(citation_ids, citation_embeds):
            batch.append((id, pickle.dumps(embed)))

            if len(batch) >= BATCH_SIZE:
                self.connection.executemany(INSERT_EMBEDDING, batch)
                self.connection.commit()

                batch.clear()

        if batch:
            self.connection.executemany(INSERT_EMBEDDING, batch)
            self.connection.commit()
