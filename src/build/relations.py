import json
import re

from pathlib import Path

from src.db.execute import execute

ROOT_DIR = Path(__file__, "../../..").resolve()
SOURCE_FILE = ROOT_DIR.joinpath("data/source/go-basic.json").resolve()

BATCH_SIZE = 2000

EXTRACT_GOID = r"http:\/\/purl\.obolibrary\.org\/obo\/GO_(\d{7})"

DROP_RELATIONS_TABLE = """
    DROP TABLE IF EXISTS relations
"""

CREATE_RELATIONS_TABLE = """
    CREATE TABLE IF NOT EXISTS relations (
        child_id INTEGER NOT NULL,
        parent_id INTEGER NOT NULL,
        predicate TEXT NOT NULL,
        PRIMARY KEY (child_id, parent_id, predicate)
    )
"""

INSERT_RELATION = """
    INSERT INTO relations (child_id, parent_id, predicate)
    VALUES (?, ?, ?)
    ON CONFLICT (child_id, parent_id, predicate) DO NOTHING
"""

{
    "sub": "http://purl.obolibrary.org/obo/GO_0000018",
    "pred": "http://purl.obolibrary.org/obo/RO_0002211",
    "obj": "http://purl.obolibrary.org/obo/GO_0006310",
}


def read(path):
    with open(path) as file:
        data = json.load(file)

        properties = {}

        for graph in data.get("graphs"):
            for node in graph.get("nodes"):
                type = node.get("type", "")

                if type != "PROPERTY":
                    continue

                id = node.get("id", "")
                label = node.get("lbl", "")

                properties[id] = label

            for edge in graph.get("edges"):
                child_id = edge.get("sub", "")
                parent_id = edge.get("obj", "")
                predicate = edge.get("pred", "")

                if not child_id or not parent_id or not predicate:
                    continue

                result = re.match(EXTRACT_GOID, child_id)

                if not result:
                    continue

                child_id = int(result.group(1))

                result = re.match(EXTRACT_GOID, parent_id)

                if not result:
                    continue

                parent_id = int(result.group(1))

                yield child_id, parent_id, properties[predicate] if predicate != "is_a" else "is_a"


def main(connection):
    connection.execute(DROP_RELATIONS_TABLE)
    connection.execute(CREATE_RELATIONS_TABLE)

    batch = []

    for row in read(SOURCE_FILE):

        batch.append(row)

        if len(batch) >= BATCH_SIZE:
            connection.executemany(INSERT_RELATION, batch)
            connection.commit()

            batch.clear()

    if len(batch) > 0:
        connection.executemany(INSERT_RELATION, batch)
        connection.commit()


if __name__ == "__main__":
    execute(main)
