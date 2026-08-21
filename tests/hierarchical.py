import json
import warnings
import sys
import pickle

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

from tests.util.metrics import hp_hr_hf
from tests.util.serializers import primitive
from tests.util.mock import MockVectorizer, SELECT_CONTENT_EMBEDDING_MAPPING

SCRIPT_NAME = Path(__file__).stem
CACHE_DIR = Path(__file__, "../../data/cache/").resolve()
MEMORY_DIR = CACHE_DIR.joinpath("memory/").resolve()

PARAM_GRID = {"classifier__n_labels": [5, 10, 20]}

N_LABELS = 20
N_JOBS = -1
N_SPLITS = 10
N_NEIGHBORS = 25

E_AMPLIFICATION = 3

RANDOM_STATE = 0


def init_custom_score(relations):

    def custom_score(classifier, X_test, y_true):
        y_pred, _ = classifier.predict(X_test)

        hp, hr, hf = hp_hr_hf(y_true, y_pred, relations)

        return {"hp": hp, "hr": hr, "hf": hf}

    return custom_score


def main(connection, method):
    client = DatabaseClient(connection)
    memory = Memory(MEMORY_DIR, verbose=0)

    relations = client.get_relations()
    custom_score = init_custom_score(relations)

    contents, _ = client.get_citations()
    _, annotations = client.get_annotations_in("citations")

    if method == "bert":
        cursor = connection.execute(SELECT_CONTENT_EMBEDDING_MAPPING)
        mapping = {content: pickle.loads(data) for content, data in cursor}

        transformer = MockVectorizer(mapping)
    else:
        transformer = TfidfVectorizer(tokenizer=tokenize, token_pattern=None)

    classifier = OntologyClassifier(
        n_neighbors=N_NEIGHBORS,
        e_amplification=E_AMPLIFICATION,
        n_jobs=N_JOBS,
        memory=memory,
    )

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
