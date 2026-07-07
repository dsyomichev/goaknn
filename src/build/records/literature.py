import sqlite3
import os
import time
import requests

from pathlib import Path
from requests import Response
from xml.etree import ElementTree

# Script Configuration

ROOT_DIRECTORY = Path(__file__, "../../../..").resolve()
SOURCE_DIRECTORY = Path(ROOT_DIRECTORY, "data/cache/literature")
DATABASE_FILE = Path(ROOT_DIRECTORY, "data/records.db").resolve()

PROJECT_NAME = os.environ.get("NCBI_PROJECT_NAME")
PROJECT_EMAIL = os.environ.get("NCBI_PROJECT_EMAIL")

API_ENDPOINT = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
API_KEY = os.environ.get("NCBI_API_KEY")

BATCH_SIZE = 200
REQUEST_INTERVAL = 1 / 10 if API_KEY else 1 / 3
MAX_RETRIES = 3

# Database Queries

INITIALIZE_DATABASE = """
    PRAGMA journal_mode = WAL;
    PRAGMA synchronous  = NORMAL;
    PRAGMA cache_size   = -65536;
"""

CREATE_TABLE = """
    CREATE TABLE IF NOT EXISTS literature (
        pmid INTEGER PRIMARY KEY,
        abstract TEXT
    );
"""

INSERT_MANY = """
    INSERT INTO literature (pmid, abstract)
    VALUES (?, ?)
    ON CONFLICT (pmid) DO UPDATE
    SET abstract = excluded.abstract
"""

SELECT_INCOMPLETE = """
    SELECT DISTINCT pmid FROM annotations
    WHERE pmid NOT IN (SELECT pmid FROM literature)
    ORDER BY pmid
"""


def parse(response, batch):

    result = {e: None for e in batch}

    try:
        root = ElementTree.fromstring(response.content)
    except:
        return result

    for citation in root.findall("./PubmedArticle/MedlineCitation"):
        pmid = citation.find("./PMID")

        if pmid is None or not pmid.text:
            continue

        try:
            pmid = int(pmid.text)
        except:
            continue

        abstract = citation.find(".//Abstract")

        if abstract is None:
            continue

        sections = [e.text for e in abstract.findall("AbstractText") if e.text is not None]

        result[pmid] = "\n\n".join(sections)

    return result


def fetch(batch):
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

            return parse(response, batch)

        except:
            if i < MAX_RETRIES:
                time.sleep(REQUEST_INTERVAL)
                continue

            raise


def main():
    connection = sqlite3.connect(DATABASE_FILE)

    connection.executescript(INITIALIZE_DATABASE)
    connection.executescript(CREATE_TABLE)

    try:
        incomplete = [e[0] for e in connection.execute(SELECT_INCOMPLETE).fetchall()]

        previous = 0.0

        for i in range(0, len(incomplete), BATCH_SIZE):

            batch = incomplete[i : i + BATCH_SIZE]

            remaining = REQUEST_INTERVAL - time.monotonic() - previous
            if remaining > 0:
                time.sleep(remaining)

            previous = time.monotonic()

            try:
                result = fetch(batch)
            except:
                continue

            if result is None:
                continue

            connection.executemany(INSERT_MANY, result.items())

            connection.commit()
    finally:
        connection.close()


if __name__ == "__main__":
    main()
