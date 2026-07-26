import os
import requests
import sys
import time

from pathlib import Path
from xml.etree import ElementTree
from sqlite3 import Connection
from requests import Response

# Script Configuration

ROOT_DIRECTORY = Path(__file__, "../../..").resolve()

PROJECT_NAME = os.environ.get("NCBI_PROJECT_NAME")
PROJECT_EMAIL = os.environ.get("NCBI_PROJECT_EMAIL")

API_ENDPOINT = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
API_KEY = os.environ.get("NCBI_API_KEY")

SOURCE_DIRECTORY = ROOT_DIRECTORY.joinpath("data/cache/documents").resolve()

BATCH_SIZE = 200
REQUEST_INTERVAL = 1 / 10 if API_KEY else 1 / 3
MAX_RETRIES = 3

# Import Local Modules

sys.path.insert(0, str(ROOT_DIRECTORY))

from lib.execute import execute

# Database Queries

CREATE_DOCUMENT_TABLE = """
    CREATE TABLE IF NOT EXISTS documents (
        pmid INTEGER PRIMARY KEY,
        body TEXT
    )
"""

INSERT_DOCUMENT = """
    INSERT INTO documents (pmid, body)
    VALUES (?, ?)
    ON CONFLICT (pmid) DO UPDATE
    SET
        body = excluded.body
"""

SELECT_INCOMPLETE_DOCUMENTS = """
    SELECT DISTINCT pmid FROM annotations
    WHERE pmid NOT IN (SELECT pmid FROM documents)
    ORDER BY pmid
"""

SELECT_STALE_DOCUMENTS = """
    SELECT DISTINCT pmid FROM documents
    WHERE pmid NOT IN (SELECT pmid from annotations)
    ORDER BY pmid
"""

SELECT_DISTINCT_DOCUMENTS = """
    SELECT DISTINCT pmid FROM documents
"""

DELETE_DOCUMENT = """
    DELETE FROM documents
    WHERE pmid = ?
"""


def process(element: ElementTree.Element):
    child = element.find(".//Abstract")

    if child is None:
        return None

    parts = [e.text for e in child.findall("AbstractText") if e.text is not None]

    return "\n\n".join(parts)


def read(path: Path):
    root = ElementTree.parse(path).getroot()
    element = root.find("./MedlineCitation")

    if element is None:
        return None

    return process(element)


def write(response: Response):
    root = ElementTree.fromstring(response.content)

    SOURCE_DIRECTORY.mkdir(parents=True, exist_ok=True)

    for element in root.findall("./PubmedArticle"):
        child = element.find("./MedlineCitation/PMID")

        if child is None or not child.text:
            continue

        try:
            pmid = int(child.text)

        except:
            continue

        path = Path(SOURCE_DIRECTORY, f"{pmid}.xml")

        ElementTree.ElementTree(element).write(path, encoding="utf-8", xml_declaration=True)


def parse(response: Response, batch: list[int]):

    result: dict[int, str | None] = {e: None for e in batch}

    try:
        root = ElementTree.fromstring(response.content)

    except:
        return result

    for element in root.findall("./PubmedArticle/MedlineCitation"):
        child = element.find("./PMID")

        if child is None or not child.text:
            continue

        try:
            pmid = int(child.text)
        except:
            continue

        result[pmid] = process(element)

    return result


def fetch(batch: list[int], cache: bool = False):
    payload = {"db": "pubmed", "rettype": "abstract", "retmod": "xml"}

    if PROJECT_NAME:
        payload["tool"] = PROJECT_NAME

    if PROJECT_EMAIL:
        payload["email"] = PROJECT_EMAIL

    if API_KEY:
        payload["api_key"] = API_KEY

    payload["id"] = ",".join(str(e) for e in batch)

    for i in range(MAX_RETRIES):
        try:
            response = requests.post(API_ENDPOINT, payload)
            response.raise_for_status()

            if cache:
                write(response)

            return parse(response, batch)

        except:
            if i < MAX_RETRIES - 1:
                time.sleep(REQUEST_INTERVAL)

                continue

            raise


def extract(connection: Connection, update: bool = False, cache: bool = True):

    connection.execute(CREATE_DOCUMENT_TABLE)

    incomplete = set([e[0] for e in connection.execute(SELECT_INCOMPLETE_DOCUMENTS).fetchall()])

    if update:
        stale = [e[0] for e in connection.execute(SELECT_STALE_DOCUMENTS).fetchall()]
        connection.executemany(DELETE_DOCUMENT, stale)

        incomplete = incomplete | set([e[0] for e in connection.execute(SELECT_DISTINCT_DOCUMENTS).fetchall()])

    incomplete = list(incomplete)

    previous = 0.0

    for i in range(0, len(incomplete), BATCH_SIZE):

        batch = incomplete[i : i + BATCH_SIZE]

        result: dict[int, str | None] = {}
        missing: list[int] = []

        for pmid in batch:
            if Path(SOURCE_DIRECTORY, f"{pmid}.xml").exists():
                result[pmid] = read(Path(SOURCE_DIRECTORY, f"{pmid}.xml"))

            else:
                missing.append(pmid)

        if missing:
            remaining = REQUEST_INTERVAL - time.monotonic() - previous
            if remaining > 0:
                time.sleep(remaining)

            previous = time.monotonic()

            try:
                fetched = fetch(missing, cache)

            except:
                fetched = None

            if fetched is not None:
                result.update(fetched)

        if not result:
            continue

        connection.executemany(INSERT_DOCUMENT, result.items())

        connection.commit()


if __name__ == "__main__":
    execute(extract)
