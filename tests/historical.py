import warnings
import sys
import datetime
import pickle
import json

from argparse import ArgumentParser
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from joblib import Memory
from sklearn.pipeline import Pipeline

from lib.OntologyClassifier import OntologyClassifier
from lib.preprocess import tokenize

from src.db.execute import execute
from src.db.DatabaseClient import DatabaseClient

from tests.util.mock import MockVectorizer, SELECT_CONTENT_EMBEDDING_MAPPING
from tests.util.metrics import hp_hr_hf, mrr_mar_f
from tests.util.serializers import primitive

SCRIPT_NAME = Path(__file__).stem
CACHE_DIR = Path(__file__, "../../data/cache/").resolve()
MEMORY_DIR = CACHE_DIR.joinpath("memory/").resolve()

SPLIT_TEMPORAL_START = 2006
SPLIT_TEMPORAL_STEP = 1

SELECT_ANNOTATIONS_WITH_DATES_IN = """
    SELECT citation_id, term_id, date FROM annotations
    WHERE citation_id IN (SELECT id FROM {})
    ORDER BY date ASC
"""

N_NEIGHBORS = 200
N_LABELS = 5
N_JOBS = -1

E_AMPLIFICATION = 3


def init_custom_score(relations):

    def custom_score(classifier, X_test, y_true):
        y_pred, _ = classifier.predict(X_test)

        _, _, hf = hp_hr_hf(y_true, y_pred, relations)
        _, _, f = mrr_mar_f(y_true, y_pred)

        return {"hf": hf, "f": f}

    return custom_score


def train_test_split_temporal(client, start=None, step=1):
    citation_contents, citation_ids = client.get_citations()
    citations = {}

    for citation_content, citation_id in zip(citation_contents, citation_ids):
        citations[citation_id] = citation_content

    annotations = client.connection.execute(SELECT_ANNOTATIONS_WITH_DATES_IN.format("citations"))

    earliest = datetime.date.max.year
    latest = datetime.date.min.year

    annual = {}

    for citation_id, term_id, date in annotations:
        year = int(str(date)[:4])

        earliest = year if year < earliest else earliest
        latest = year if year > latest else latest

        citation_ids = annual.setdefault(year, {})
        term_ids = citation_ids.setdefault(citation_id, [])

        term_ids.append(term_id)

    train = {}

    for year in range(earliest, start or earliest):
        citation_ids = annual.get(year, {})

        for citation_id, term_ids in citation_ids.items():
            term_ids_set = train.setdefault(citation_id, set())
            term_ids_set |= set(term_ids)

    for year in range(start or earliest, latest + 1, step):
        split_test = {}

        years_test = list(range(year, year + step if year + step <= latest + 1 else latest + 1))

        for year_test in years_test:
            citation_ids = annual.get(year_test, {})

            for citation_id, term_ids in citation_ids.items():
                term_ids_set = split_test.setdefault(citation_id, set())
                term_ids_set |= set(term_ids)

        split_X_train, split_X_test = [], []
        split_y_train, split_y_test = [], []

        cit_ids_train, cit_ids_test = [], []

        for citation_id, term_ids in sorted(train.items(), key=lambda item: item[0]):
            split_X_train.append(citations[citation_id])
            split_y_train.append(list(term_ids))

            cit_ids_train.append(citation_id)

        for citation_id, term_ids in sorted(split_test.items(), key=lambda item: item[0]):
            term_ids_set = train.setdefault(citation_id, set())
            term_ids_set |= set(term_ids)

            split_X_test.append(citations[citation_id])
            split_y_test.append(list(term_ids_set))

            cit_ids_test.append(citation_id)

        yield split_X_train, split_X_test, split_y_train, split_y_test


def main(connection, method):
    client = DatabaseClient(connection)
    memory = Memory(MEMORY_DIR, verbose=0)

    relations = client.get_relations()
    custom_score = init_custom_score(relations)

    if method == "bert":
        cursor = connection.execute(SELECT_CONTENT_EMBEDDING_MAPPING)
        mapping = {content: pickle.loads(data) for content, data in cursor}

        transformer = MockVectorizer(mapping)
    else:
        transformer = TfidfVectorizer(tokenizer=tokenize, token_pattern=None)

    classifier = OntologyClassifier(
        n_jobs=N_JOBS,
        n_neighbors=N_NEIGHBORS,
        n_labels=N_LABELS,
        e_amplification=E_AMPLIFICATION,
        memory=memory,
        ignore_exact=True,
    )

    steps = [("transformer", transformer), ("classifier", classifier)]
    pipeline = Pipeline(steps, memory=memory if method == "tfidf" else None)

    result = {}

    train_test_split_generator = train_test_split_temporal(client, start=SPLIT_TEMPORAL_START, step=SPLIT_TEMPORAL_STEP)

    for i, (X_train, X_test, y_train, y_test) in enumerate(train_test_split_generator):
        year = SPLIT_TEMPORAL_START + i

        pipeline.fit(X_train, y_train)

        scores = custom_score(pipeline, X_test, y_test)

        print(year, scores)

        result[year] = scores

    output_file = Path.joinpath(CACHE_DIR, f"{method}_{SCRIPT_NAME}.json")
    with open(output_file, "w") as file:
        json.dump(result, file, default=primitive, indent=4)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    parser = ArgumentParser()
    parser.add_argument("target", choices=["tfidf", "bert"])

    namespace = parser.parse_args(sys.argv[1:])

    execute(main, method=namespace.target)
