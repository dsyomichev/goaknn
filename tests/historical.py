import warnings
import sys
import datetime
import pickle
import json

from argparse import ArgumentParser
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from joblib import Memory
from sklearn.pipeline import Pipeline

from lib.OntologyClassifier import OntologyClassifier
from lib.preprocess import tokenize

from src.db.execute import execute
from src.db.DatabaseClient import DatabaseClient

from tests.util.mock import MockVectorizer, SELECT_CONTENT_EMBEDDING_MAPPING
from tests.util.metrics import hp_hr_hf, mrr_mar
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

N_JOBS = -1
N_LABELS = 20

BEST_PARAMS = {
    "bert": {"n_neighbors": 50, "e_amplification": 5},
    "tfidf": {"n_neighbors": 200, "e_amplification": 3},
}

CACHE_DIR.mkdir(parents=True, exist_ok=True)


def init_hierarchical_score(relations):

    def hierarchical_score(classifier, X_test, y_true):
        y_pred, _ = classifier.predict(X_test)

        hp, hr, hf = hp_hr_hf(y_true, y_pred, relations)

        return {"hp": hp, "hr": hr, "hf": hf}

    return hierarchical_score


def exact_score(classifier, X_test, y_true):
    y_pred, _ = classifier.predict(X_test)

    mrr, mar = mrr_mar(y_true, y_pred)

    return {"mrr": mrr, "mar": mar}


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


def main(connection, method, metric):
    client = DatabaseClient(connection)
    memory = Memory(MEMORY_DIR, verbose=0)

    if metric == "hierarchical":
        relations = client.get_relations()
        custom_score = init_hierarchical_score(relations)
    else:
        custom_score = exact_score

    if method == "bert":
        cursor = connection.execute(SELECT_CONTENT_EMBEDDING_MAPPING)
        mapping = {content: pickle.loads(data) for content, data in cursor}

        transformer = MockVectorizer(mapping)
    else:
        transformer = TfidfVectorizer(tokenizer=tokenize, token_pattern=None)

    classifier = OntologyClassifier(
        n_jobs=N_JOBS,
        n_neighbors=BEST_PARAMS[method]["n_neighbors"],
        n_labels=N_LABELS,
        e_amplification=BEST_PARAMS[method]["e_amplification"],
        memory=memory,
        ignore_exact=True,
    )

    scaler = StandardScaler(with_std=False) if method == "bert" else "passthrough"

    steps = [("transformer", transformer), ("scaler", scaler), ("classifier", classifier)]
    pipeline = Pipeline(steps, memory=memory if method == "tfidf" else None)

    result = {}

    train_test_split_generator = train_test_split_temporal(client, start=SPLIT_TEMPORAL_START, step=SPLIT_TEMPORAL_STEP)

    for i, (X_train, X_test, y_train, y_test) in enumerate(train_test_split_generator):
        year = SPLIT_TEMPORAL_START + i

        pipeline.fit(X_train, y_train)

        scores = custom_score(pipeline, X_test, y_test)

        print(year, scores)

        result[year] = scores

    output_file = Path.joinpath(CACHE_DIR, f"historical_{method}_{metric}.json")
    with open(output_file, "w") as file:
        json.dump(result, file, default=primitive, indent=4)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    parser = ArgumentParser()
    parser.add_argument("method", choices=["tfidf", "bert"])
    parser.add_argument("metric", choices=["exact", "hierarchical"])

    namespace = parser.parse_args(sys.argv[1:])

    execute(main, method=namespace.method, metric=namespace.metric)
