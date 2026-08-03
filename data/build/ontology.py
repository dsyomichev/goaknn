import json
import re
import requests

from pathlib import Path
from sqlite3 import Connection
from typing import Generator

from lib.util.execute import execute

# Script Configuration

ROOT_DIRECTORY = Path(__file__, "../..").resolve()

CACHE_DIRECTORY = ROOT_DIRECTORY.joinpath("data/cache").resolve()

SOURCE_URL = "https://current.geneontology.org/ontology/go-basic.json"
SOURCE_FILE = ROOT_DIRECTORY.joinpath("data/cache/go-basic.json").resolve()

BATCH_SIZE = 2000
CHUNK_SIZE = 65536

# SQL Queries

CREATE_ONTOLOGY_TABLE = """
    CREATE TABLE IF NOT EXISTS ontology (
        id INTEGER PRIMARY KEY,
        label TEXT NOT NULL
    )
"""

INSERT_TERM = """
    INSERT INTO ontology (id, label)
    VALUES (?, ?)
    ON CONFLICT (id)
    DO UPDATE SET
        label = EXCLUDED.label
"""

SELECT_ALL_TERMS = """
    SELECT id FROM ontology
"""

DELETE_TERM = """
    DELETE FROM ontology
    WHERE id = ?
"""

# Regex Patterns

EXTRACT_ID = r"http:\/\/purl\.obolibrary\.org\/obo\/GO_(\d{7})"


def read(path: Path) -> Generator[tuple[int, str]]:
    with open(path) as file:
        data = json.load(file)

        for graph in data.get("graphs"):
            for node in graph.get("nodes"):
                url = node.get("id")

                result = re.search(EXTRACT_ID, url)

                if result is None or result.group(1) is None:
                    continue

                id = int(result.group(1))
                label = node.get("lbl")

                if label is None:
                    continue

                yield (id, label)


def download(
    source: str,
    destination: Path,
) -> None:
    temporary = destination.with_suffix(".tmp")
    temporary.parent.mkdir(parents=True, exist_ok=True)

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


def extract(
    connection: Connection,
    update: bool = False,
    cache: bool = True,
) -> None:

    if not SOURCE_FILE.exists() or update:
        download(SOURCE_URL, SOURCE_FILE)

        if not cache:
            SOURCE_FILE.unlink()

    connection.execute(CREATE_ONTOLOGY_TABLE)

    stale = [e[0] for e in connection.execute(SELECT_ALL_TERMS).fetchall()]

    batch: list[tuple[int, str]] = []

    for id, label in read(SOURCE_FILE):
        batch.append((id, label))

        if id in stale:
            stale.remove(id)

        if len(batch) >= BATCH_SIZE:
            connection.executemany(INSERT_TERM, batch)

            connection.commit()
            batch.clear()

    if batch:
        connection.executemany(INSERT_TERM, batch)

    if stale:
        connection.executemany(DELETE_TERM, [[e] for e in stale])

    connection.commit()


if __name__ == "__main__":
    execute(extract)
