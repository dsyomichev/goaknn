import gzip
import hashlib
import re
import requests
import sys

from pathlib import Path
from sqlite3 import Connection

# Script Configuration

ROOT_DIRECTORY = Path(__file__, "../..").resolve()
CACHE_DIRECTORY = ROOT_DIRECTORY.joinpath("data/cache").resolve()

SOURCE_URL = "https://ftp.ebi.ac.uk/pub/databases/GO/goa/UNIPROT/goa_uniprot_all.gaf.gz"
SOURCE_FILE = ROOT_DIRECTORY.joinpath("data/cache/goa_uniprot_all.gaf.gz").resolve()

BATCH_SIZE = 20000
CHUNK_SIZE = 65536

# Import Local Modules

sys.path.insert(0, str(ROOT_DIRECTORY))

from lib.util.execute import execute

# SQL Queries

CREATE_ANNOTATIONS_TABLE = """
    CREATE TABLE IF NOT EXISTS annotations (
        term_id INTEGER NOT NULL,
        citation_id INTEGER NOT NULL,
        date INTEGER NOT NULL,
        PRIMARY KEY (term_id, citation_id),
        FOREIGN KEY (term_id) REFERENCES ontology (id)
    )
"""

INSERT_ANNOTATION = """
    INSERT INTO annotations (term_id, citation_id, date)
    VALUES (?, ?, ?)
    ON CONFLICT (term_id, citation_id) DO UPDATE
    SET
        date = MIN(date, excluded.date)
"""

SELECT_ALL_ANNOTATIONS = """
    SELECT term_id, citation_id FROM annotations
"""

DELETE_ANNOTATION = """
    DELETE FROM annotations
    WHERE term_id = ? AND citation_id = ?
"""

# Regex Patterns

EXTRACT_TERM_ID = r"GO:(\d{7})"

EXTRACT_CITATION_ID = r"PMID:(\d+)"


def checksum(path: Path):
    if not path.exists():
        return None

    digest = hashlib.md5()

    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(CHUNK_SIZE), b""):
            digest.update(chunk)

    return digest.hexdigest()


def validate(remote: str, local: Path):
    response = requests.get(remote + ".md5")
    response.raise_for_status()

    return response.text.strip().split()[0] == checksum(local)


def download(source: str, destination: Path):
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


def read(path: Path):
    with gzip.open(path, "rt") as file:
        for line in file:
            if line.startswith("!"):
                continue

            values = line.rstrip().split("\t")

            result = re.search(EXTRACT_TERM_ID, values[4])

            if result is None or result.group(1) is None:
                continue

            term_id = int(result.group(1))

            result = re.search(EXTRACT_CITATION_ID, values[5])

            if result is None or result.group(1) is None:
                continue

            citation_id = int(result.group(1))

            date = values[13]

            yield (term_id, citation_id, date)


def extract(connection: Connection, update: bool = False, cache: bool = True):

    if not SOURCE_FILE.exists() or update:
        if not validate(SOURCE_URL, SOURCE_FILE):
            download(SOURCE_URL, SOURCE_FILE)

            if not cache:
                SOURCE_FILE.unlink()

    connection.execute(CREATE_ANNOTATIONS_TABLE)

    stale = [(e[0], e[1]) for e in connection.execute(SELECT_ALL_ANNOTATIONS).fetchall()]

    batch: list[tuple[int, int, str]] = []

    for term_id, citation_id, date in read(SOURCE_FILE):
        batch.append((term_id, citation_id, date))

        if (term_id, citation_id) in stale:
            stale.remove((term_id, citation_id))

        if len(batch) >= BATCH_SIZE:
            connection.executemany(INSERT_ANNOTATION, batch)

            connection.commit()
            batch.clear()

    if len(batch) > 0:
        connection.executemany(INSERT_ANNOTATION, batch)

    if len(stale) > 0:
        connection.executemany(DELETE_ANNOTATION, stale)

    connection.commit()


if __name__ == "__main__":
    execute(extract)
