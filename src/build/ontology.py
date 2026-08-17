import json
import re

from pathlib import Path

from src.db.util.execute import execute

ROOT_DIRECTORY = Path(__file__, "../..").resolve()
SOURCE_FILE = ROOT_DIRECTORY.joinpath("data/source/go-basic.json").resolve()

BATCH_SIZE = 2000

EXTRACT_GOID = r"http:\/\/purl\.obolibrary\.org\/obo\/GO_(\d{7})"

CREATE_ONTOLOGY_TABLE = """
    CREATE TABLE IF NOT EXISTS ontology (
        id INTEGER PRIMARY KEY,
        label TEXT NOT NULL
    )
"""

INSERT_TERM = """
    INSERT INTO ontology (id, label)
    VALUES (?, ?)
    ON CONFLICT (id) DO UPDATE
    SET label = excluded.label
"""


def read(path):
    with open(path) as file:
        data = json.load(file)

        for graph in data.get("graphs"):
            for node in graph.get("nodes"):
                label = node.get("lbl")
                symbol = node.get("id")

                if label is None or symbol is None:
                    continue

                result = re.match(EXTRACT_GOID, symbol)

                if result is None or result.group(1) is None:
                    continue

                try:
                    id = int(result.group(1))
                except:
                    continue

                yield id, label


def main(connection):
    connection.execute(CREATE_ONTOLOGY_TABLE)

    batch = []

    for row in read(SOURCE_FILE):
        batch.append(row)

        if len(batch) >= BATCH_SIZE:
            connection.executemany(INSERT_TERM, batch)
            connection.commit()

            batch.clear()

    if len(batch) > 0:
        connection.executemany(INSERT_TERM, batch)
        connection.commit()


if __name__ == "__main__":
    execute(main)
