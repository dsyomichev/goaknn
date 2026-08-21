import json
import pickle
import warnings
import sys

from joblib import Memory
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from pathlib import Path
from argparse import ArgumentParser

from lib.OntologyClassifier import OntologyClassifier
from lib.preprocess import tokenize

from src.db.DatabaseClient import DatabaseClient
from src.db.execute import execute

from tests.util.metrics import mrr_mar_f
from tests.util.serializers import primitive
from tests.util.mock import MockVectorizer, SELECT_CONTENT_EMBEDDING_MAPPING

SCRIPT_NAME = Path(__file__).stem
CACHE_DIR = Path(__file__, "../../data/cache/").resolve()
MEMORY_DIR = CACHE_DIR.joinpath("memory/").resolve()

PARAM_GRID = {
    "classifier__n_neighbors": [25, 50, 100, 200],
    "classifier__e_amplification": [1, 3, 5],
    "classifier__n_labels": [5, 10, 20],
}

N_LABELS = 20
N_JOBS = -1
N_SPLITS = 10

RANDOM_STATE = 0


def custom_score(classifier, X_test, y_true):
    y_pred, _ = classifier.predict(X_test)

    mrr, mar, f = mrr_mar_f(y_true, y_pred)

    return {"mrr": mrr, "mar": mar, "f": f}


def main(connection, method):
    client = DatabaseClient(connection)
    memory = Memory(MEMORY_DIR, verbose=0)

    contents, _ = client.get_citations()
    _, annotations = client.get_annotations_in("citations")

    m_neighbors = max(PARAM_GRID["classifier__n_neighbors"])

    if method == "bert":
        cursor = connection.execute(SELECT_CONTENT_EMBEDDING_MAPPING)
        mapping = {content: pickle.loads(data) for content, data in cursor}

        transformer = MockVectorizer(mapping)
    else:
        transformer = TfidfVectorizer(tokenizer=tokenize, token_pattern=None)

    classifier = OntologyClassifier(n_jobs=N_JOBS, m_neighbors=m_neighbors, memory=memory)

    steps = [("transformer", transformer), ("classifier", classifier)]
    pipeline = Pipeline(steps, memory=memory if method == "tfidf" else None)

    kfold = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    gridsearch = GridSearchCV(
        pipeline,
        param_grid=PARAM_GRID,
        scoring=custom_score,
        refit=False,
        cv=kfold,
        verbose=10,
        error_score="raise",
    )

    gridsearch.fit(contents, annotations)

    output_file = Path.joinpath(CACHE_DIR, f"{method}_{SCRIPT_NAME}.json")
    with open(output_file, "w") as file:
        json.dump(gridsearch.cv_results_, file, default=primitive, indent=4)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    parser = ArgumentParser()
    parser.add_argument("target", choices=["tfidf", "bert"])

    namespace = parser.parse_args(sys.argv[1:])

    execute(main, method=namespace.target)
