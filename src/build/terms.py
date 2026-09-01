import json
import re

from pathlib import Path

from src.db import execute

ROOT_DIR = Path(__file__, "../../..").resolve()
SOURCE_FILE = ROOT_DIR.joinpath("data/source/go-basic.json").resolve()

BATCH_SIZE = 2000

EXTRACT_GOID = r"http:\/\/purl\.obolibrary\.org\/obo\/GO_(\d{7})"

DROP_TERMS_TABLE = """
    DROP TABLE IF EXISTS terms
"""

CREATE_TERMS_TABLE = """
    CREATE TABLE IF NOT EXISTS terms (
        id INTEGER PRIMARY KEY,
        label TEXT NOT NULL
    )
"""

INSERT_TERM = """
    INSERT INTO terms (id, label)
    VALUES (?, ?)
    ON CONFLICT (id) DO UPDATE
    SET label = excluded.label
"""


def read(path):
    with open(path) as file:
        data = json.load(file)

        for graph in data.get("graphs"):
            for node in graph.get("nodes"):
                id = node.get("id", "")
                label = node.get("lbl", "")

                if not id or not label:
                    continue

                result = re.match(EXTRACT_GOID, id)

                if not result:
                    continue

                id = int(result.group(1))

                # Only yield if id and label were both extracted
                yield id, label


def main(connection):
    connection.execute(DROP_TERMS_TABLE)
    connection.execute(CREATE_TERMS_TABLE)

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
