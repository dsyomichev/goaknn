import os
import requests
import time

from pathlib import Path
from xml.etree import ElementTree
from sqlite3 import Connection
from requests import Response

from lib.util.execute import execute

ROOT_DIRECTORY = Path(__file__, "../..").resolve()

PROJECT_NAME = os.environ.get("NCBI_PROJECT_NAME")
PROJECT_EMAIL = os.environ.get("NCBI_PROJECT_EMAIL")

API_ENDPOINT = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
API_KEY = os.environ.get("NCBI_API_KEY")

SOURCE_DIRECTORY = ROOT_DIRECTORY.joinpath("data/cache/efetch").resolve()

BATCH_SIZE = 200
REQUEST_INTERVAL = 1 / 10 if API_KEY else 1 / 3
MAX_RETRIES = 3

CREATE_CORPUS_TABLE = """
    CREATE TABLE IF NOT EXISTS corpus (
        id INTEGER PRIMARY KEY,
        content TEXT,
        FOREIGN KEY (id) REFERENCES annotations (citation_id)
    )
"""

INSERT_CITATION = """
    INSERT INTO corpus (id, content)
    VALUES (?, ?)
    ON CONFLICT (id)
    DO UPDATE SET
        content = excluded.content
"""

SELECT_INCOMPLETE_CITATIONS = """
    SELECT DISTINCT citation_id FROM annotations
    WHERE citation_id NOT IN (SELECT id FROM corpus)
    ORDER BY citation_id
"""

SELECT_STALE_CITATIONS = """
    SELECT id FROM corpus
    WHERE id NOT IN (SELECT DISTINCT citation_id from annotations)
    ORDER BY id
"""

SELECT_ALL_CITATIONS = """
    SELECT id FROM corpus
"""

DELETE_CITATION = """
    DELETE FROM corpus
    WHERE id = ?
"""


def process(
    element: ElementTree.Element,
) -> str | None:
    child = element.find(".//Abstract")

    if child is None:
        return None

    parts = [e.text for e in child.findall("AbstractText") if e.text is not None]

    return "\n\n".join(parts)


def read(
    path: Path,
) -> str | None:
    root = ElementTree.parse(path).getroot()
    element = root.find("./MedlineCitation")

    if element is None:
        return None

    return process(element)


def write(
    response: Response,
) -> None:
    root = ElementTree.fromstring(response.content)

    SOURCE_DIRECTORY.mkdir(parents=True, exist_ok=True)

    for element in root.findall("./PubmedArticle"):
        child = element.find("./MedlineCitation/PMID")

        if child is None or not child.text:
            continue

        try:
            id = int(child.text)

        except:
            continue

        path = Path(SOURCE_DIRECTORY, f"{id}.xml")

        ElementTree.ElementTree(element).write(
            path, encoding="utf-8", xml_declaration=True
        )


def parse(
    response: Response,
    batch: list[int],
) -> dict[int, str | None]:

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
            id = int(child.text)
        except:
            continue

        result[id] = process(element)

    return result


def fetch(
    batch: list[int],
    cache: bool = False,
) -> dict[int, str | None] | None:
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


def extract(
    connection: Connection,
    update: bool = False,
    cache: bool = True,
) -> None:

    connection.execute(CREATE_CORPUS_TABLE)

    incomplete = set(
        [e[0] for e in connection.execute(SELECT_INCOMPLETE_CITATIONS).fetchall()]
    )

    if update:
        stale = [e[0] for e in connection.execute(SELECT_STALE_CITATIONS).fetchall()]
        connection.executemany(DELETE_CITATION, stale)

        incomplete = incomplete | set(
            [e[0] for e in connection.execute(SELECT_ALL_CITATIONS).fetchall()]
        )

    incomplete = list(incomplete)

    previous = 0.0

    for i in range(0, len(incomplete), BATCH_SIZE):

        batch = incomplete[i : i + BATCH_SIZE]

        result: dict[int, str | None] = {}
        missing: list[int] = []

        for id in batch:
            if Path(SOURCE_DIRECTORY, f"{id}.xml").exists():
                result[id] = read(Path(SOURCE_DIRECTORY, f"{id}.xml"))

            else:
                missing.append(id)

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

        connection.executemany(INSERT_CITATION, result.items())

        connection.commit()


if __name__ == "__main__":
    execute(extract)
