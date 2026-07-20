import gzip
import hashlib
import re
import requests
import sys

from pathlib import Path

# Script Configuration

ROOT_DIRECTORY = Path(__file__, "../../..").resolve()
CACHE_DIRECTORY = Path(ROOT_DIRECTORY, "data/cache").resolve()

SOURCE_URL = "https://ftp.ebi.ac.uk/pub/databases/GO/goa/UNIPROT/goa_uniprot_all.gaf.gz"
SOURCE_FILE = Path(ROOT_DIRECTORY, "data/cache/goa_uniprot_all.gaf.gz").resolve()

BATCH_SIZE = 20000
CHUNK_SIZE = 65536

# Import Local Modules

sys.path.insert(0, str(ROOT_DIRECTORY))

from lib.execute import execute

# SQL Queries

CREATE_TABLE = """
    CREATE TABLE IF NOT EXISTS annotations (
        goid INTEGER NOT NULL,
        pmid INTEGER NOT NULL,
        date INTEGER NOT NULL,
        PRIMARY KEY (goid, pmid)
    );
"""

INSERT_MANY = """
    INSERT INTO annotations (goid, pmid, date)
    VALUES (?, ?, ?)
    ON CONFLICT (goid, pmid) DO UPDATE
    SET
        date = MIN(date, excluded.date)
"""

SELECT_DISTINCT = """
    SELECT DISTINCT goid, pmid FROM annotations
"""

DELETE_MANY = """
    DELETE FROM annotations
    WHERE goid = ? AND pmid = ?
"""

# Regex Patterns

EXTRACT_GOID = r"GO:(\d{7})"

EXTRACT_PMID = r"PMID:(\d+)"


def checksum(path):
    if not Path(path).exists():
        return None

    digest = hashlib.md5()

    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(CHUNK_SIZE), b""):
            digest.update(chunk)

    return digest.hexdigest()


def validate(remote, local):
    response = requests.get(remote + ".md5")
    response.raise_for_status()

    return response.text.strip().split()[0] == checksum(local)


def download(source, destination):
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


def read(path):
    with gzip.open(path, "rt") as file:
        for line in file:
            if line.startswith("!"):
                continue

            values = line.rstrip().split("\t")

            result = re.search(EXTRACT_GOID, values[4])

            if result is None or result.group(1) is None:
                continue

            goid = int(result.group(1))

            result = re.search(EXTRACT_PMID, values[5])

            if result is None or result.group(1) is None:
                continue

            pmid = int(result.group(1))

            date = values[13]

            yield (goid, pmid, date)


def extract(connection, update=False, cache=True):

    if not SOURCE_FILE.exists() or update:
        if not validate(SOURCE_URL, SOURCE_FILE):
            download(SOURCE_URL, SOURCE_FILE)

            if not cache:
                SOURCE_FILE.unlink()

    connection.executescript(CREATE_TABLE)

    stale = set([(int(e[0]), int(e[1])) for e in connection.execute(SELECT_DISTINCT).fetchall()])

    batch = []

    for goid, pmid, date in read(SOURCE_FILE):
        batch.append((goid, pmid, date))

        if (goid, pmid) in stale:
            stale.remove((goid, pmid))

        if len(batch) >= BATCH_SIZE:
            connection.executemany(INSERT_MANY, batch)

            connection.commit()
            batch.clear()

    if len(batch) > 0:
        connection.executemany(INSERT_MANY, batch)

    if len(stale) > 0:
        connection.executemany(DELETE_MANY, stale)

    connection.commit()


if __name__ == "__main__":
    execute(extract)
