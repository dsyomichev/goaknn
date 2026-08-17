import os
import time
import requests

from pathlib import Path
from xml.etree import ElementTree

from src.db.util.execute import execute

ROOT_DIRECTORY = Path(__file__, "../..").resolve()
SOURCE_DIRECTORY = Path(ROOT_DIRECTORY, "data/source/efetch")

PROJECT_NAME = os.environ.get("NCBI_PROJECT_NAME")
PROJECT_EMAIL = os.environ.get("NCBI_PROJECT_EMAIL")
API_KEY = os.environ.get("NCBI_API_KEY")

REMOTE_ENDPOINT = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

BATCH_SIZE = 200
REQUEST_INTERVAL = 1 / 10 if API_KEY else 1 / 3

CREATE_LITERATURE_TABLE = """
    CREATE TABLE IF NOT EXISTS corpus (
        id INTEGER PRIMARY KEY,
        content TEXT
    )
"""

INSERT_DOCUMENT = """
    INSERT INTO corpus (id, content)
    VALUES (?, ?)
    ON CONFLICT (id) DO UPDATE
    SET content = excluded.content
"""

SELECT_INCOMPLETE = """
    SELECT citation_id FROM annotations
    WHERE citation_id NOT IN (SELECT id FROM corpus)
    ORDER BY citation_id
"""


def parse(response):
    result = {}

    root = ElementTree.fromstring(response.content)

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

        sections = [section.text for section in abstract.findall("AbstractText") if section.text is not None]

        result[pmid] = "\n\n".join(sections)

    return result


def fetch(batch):
    payload = {"db": "pubmed", "rettype": "abstract", "retmode": "xml"}

    if PROJECT_NAME:
        payload["tool"] = PROJECT_NAME

    if PROJECT_EMAIL:
        payload["email"] = PROJECT_EMAIL

    if API_KEY:
        payload["api_key"] = API_KEY

    payload["id"] = ",".join(str(id) for id in batch)

    response = requests.post(REMOTE_ENDPOINT, payload)
    response.raise_for_status()

    return parse(response, batch)


def main(connection):
    connection.execute(CREATE_LITERATURE_TABLE)

    incomplete = [id for (id,) in connection.execute(SELECT_INCOMPLETE).fetchall()]

    previous = 0.0

    for i in range(0, len(incomplete), BATCH_SIZE):
        batch = incomplete[i : i + BATCH_SIZE]

        remaining = REQUEST_INTERVAL + previous - time.monotonic()

        if remaining > 0:
            time.sleep(remaining)

        previous = time.monotonic()

        result = fetch(batch)

        connection.executemany(INSERT_DOCUMENT, result.items())
        connection.commit()


if __name__ == "__main__":
    execute(main)
