import gzip
import re

from pathlib import Path

from src.db.execute import execute

ROOT_DIR = Path(__file__, "../../..").resolve()
SOURCE_FILE = ROOT_DIR.joinpath("data/source/goa_uniprot_all.gaf.gz").resolve()

BATCH_SIZE = 100000

EXTRACT_GOID = r"GO:(\d{7})"
EXTRACT_PMID = r"PMID:(\d+)"

CREATE_ANNOTATIONS_TABLE = """
    CREATE TABLE IF NOT EXISTS annotations (
        term_id INTEGER NOT NULL,
        citation_id INTEGER NOT NULL,
        date INTEGER NOT NULL,
        PRIMARY KEY (term_id, citation_id)
    )
"""

INSERT_ANNOTATION = """
    INSERT INTO annotations (term_id, citation_id, date)
    VALUES (?, ?, ?)
    ON CONFLICT (term_id, citation_id) DO UPDATE
    SET date = MIN(date, excluded.date)
"""


def read(path):
    with gzip.open(path, "rt") as file:
        for line in file:
            if line.startswith("!"):
                continue

            values = line.strip().split("\t")

            try:
                date = int(values[13])
            except:
                continue

            result = re.search(EXTRACT_GOID, values[4])

            if result is None or result.group(1) is None:
                continue

            try:
                term_id = int(result.group(1))
            except:
                continue

            result = re.search(EXTRACT_PMID, values[5])

            if result is None or result.group(1) is None:
                continue

            try:
                citation_id = int(result.group(1))
            except:
                continue

            yield term_id, citation_id, date


def main(connection):
    connection.execute(CREATE_ANNOTATIONS_TABLE)

    batch = []

    for row in read(SOURCE_FILE):
        batch.append(row)

        if len(batch) >= BATCH_SIZE:
            connection.executemany(INSERT_ANNOTATION, batch)
            connection.commit()

            batch.clear()

    if batch:
        connection.executemany(INSERT_ANNOTATION, batch)
        connection.commit()


if __name__ == "__main__":
    execute(main)
