import json
import re
import requests
import sys

from pathlib import Path

# Script Configuration

ROOT_DIRECTORY = Path(__file__, "../../..").resolve()

CACHE_DIRECTORY = ROOT_DIRECTORY.joinpath("data/cache").resolve()

SOURCE_URL = "https://current.geneontology.org/ontology/go-basic.json"
SOURCE_FILE = ROOT_DIRECTORY.joinpath("data/cache/go-basic.json").resolve()

BATCH_SIZE = 2000
CHUNK_SIZE = 65536

# Import Local Modules

sys.path.insert(0, str(ROOT_DIRECTORY))

from lib.execute import execute

# SQL Queries

CREATE_TABLE = """
    CREATE TABLE IF NOT EXISTS terms (
        goid INTEGER PRIMARY KEY,
        label TEXT NOT NULL
    );
"""

INSERT_MANY = """
    INSERT INTO terms (goid, label)
    VALUES (?, ?)
    ON CONFLICT (goid)
    DO UPDATE SET
        label = EXCLUDED.label
"""

SELECT_DISTINCT = """
    SELECT DISTINCT goid FROM terms
"""

DELETE_MANY = """
    DELETE FROM terms
    WHERE goid = ?
"""

# Regex Patterns

EXTRACT_GOID = r"http:\/\/purl\.obolibrary\.org\/obo\/GO_(\d{7})"


def read(path):
    with open(path) as file:
        data = json.load(file)

        for graph in data.get("graphs"):
            for node in graph.get("nodes"):
                id = node.get("id")

                result = re.search(EXTRACT_GOID, id)

                if result is None or result.group(1) is None:
                    continue

                goid = int(result.group(1))
                label = node.get("lbl")

                if label is None:
                    continue

                yield (goid, label)


def download(source, destination):
    temporary = destination.with_suffix(".tmp")

    try:
        with requests.get(source, stream=True) as response:
            response.raise_for_status()

            with open(temporary, "wb") as file:
                for chunk in response.iter_content(CHUNK_SIZE):
                    file.write(chunk)

            temporary.replace(destination)
    except:
        temporary.unlink(missing_ok=True)

        raise


def extract(connection, update=False, cache=True):

    if not SOURCE_FILE.exists() or update:
        download(SOURCE_URL, SOURCE_FILE)

        if not cache:
            SOURCE_FILE.unlink()

    connection.executescript(CREATE_TABLE)

    stale = set([int(e[0]) for e in connection.execute(SELECT_DISTINCT).fetchall()])

    batch = []

    for goid, label in read(SOURCE_FILE):
        batch.append((goid, label))

        if goid in stale:
            stale.remove(goid)

        if len(batch) >= BATCH_SIZE:
            connection.executemany(INSERT_MANY, batch)

            connection.commit()
            batch.clear()

    if len(batch) > 0:
        connection.executemany(INSERT_MANY, batch)

    if len(stale) > 0:
        connection.executemany(DELETE_MANY, [(e,) for e in stale])

    connection.commit()


if __name__ == "__main__":
    execute(extract)
