import os
import time
import requests

from xml.etree import ElementTree

from src.db.execute import execute

PROJECT_NAME = os.environ.get("NCBI_PROJECT_NAME")
PROJECT_EMAIL = os.environ.get("NCBI_PROJECT_EMAIL")
API_KEY = os.environ.get("NCBI_API_KEY")

REMOTE_ENDPOINT = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

BATCH_SIZE = 200
REQUEST_INTERVAL = 1 / 10 if API_KEY else 1 / 3
MAX_RETRY = 3

EXTRACT_WHITESPACE = r"^\s*$"

CREATE_CITATIONS_TABLE = """
    CREATE TABLE IF NOT EXISTS citations (
        id INTEGER PRIMARY KEY,
        date INTEGER NOT NULL,
        content TEXT NOT NULL
    )
"""

INSERT_DOCUMENT = """
    INSERT INTO citations (id, date, content)
    VALUES (?, ?, ?)
    ON CONFLICT (id) DO UPDATE
    SET date = excluded.date, content = excluded.content
"""

SELECT_INCOMPLETE = """
    SELECT citation_id FROM annotations
    WHERE citation_id NOT IN (SELECT id FROM citations)
    ORDER BY citation_id
"""


def parse(response):
    result = {}

    root = ElementTree.fromstring(response.content)

    for article in root.findall("./PubmedArticle"):
        citation = article.find("./MedlineCitation")

        if citation is None:
            continue

        try:
            id = int(citation.findtext("./PMID"))
        except:
            continue

        abstract = citation.find("./Article/Abstract")

        if abstract is None:
            continue

        sections = []

        for text in abstract.findall("./AbstractText"):
            section = "".join(text.itertext()).strip()

            if not section:
                continue

            sections.append(section)

        content = "\n\n".join(sections)

        if not content:
            continue

        date = 0

        history = article.find("./PubmedData/History/PubMedPubDate[@PubStatus='pubmed']")

        if history is None:
            result[id] = (date, content)

            continue

        try:
            year = int(history.findtext("./Year"))
            month = int(history.findtext("./Month"))
            day = int(history.findtext("./Day"))
        except:
            result[id] = (date, content)

            continue

        date = int(f"{year:0>4}{month:0>2}{day:0>2}")

        result[id] = (date, content)

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

    for i in range(MAX_RETRY):
        try:
            response = requests.post(REMOTE_ENDPOINT, payload)
            response.raise_for_status()

            return parse(response)
        except:
            if i == MAX_RETRY - 1:
                raise

            continue


def main(connection):
    connection.execute(CREATE_CITATIONS_TABLE)

    incomplete = [id for (id,) in connection.execute(SELECT_INCOMPLETE).fetchall()]

    for i in range(0, len(incomplete), BATCH_SIZE):
        batch = incomplete[i : i + BATCH_SIZE]

        time.sleep(REQUEST_INTERVAL)

        result = fetch(batch)

        connection.executemany(INSERT_DOCUMENT, [(id, date, content) for id, (date, content) in result.items()])
        connection.commit()


if __name__ == "__main__":
    execute(main)
