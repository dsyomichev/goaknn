import functools
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

from lib.MultiLabelNearestNeighborsClassifier import MultiLabelNearestNeighborsClassifier
from lib.preprocess import tokenize

from src.db.DatabaseClient import DatabaseClient
from src.db.execute import execute

from tests.util.metrics import hierarchical_precision, hierarchical_recall
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

hierarchy = {}


@functools.cache
def trace(label):
    relations = hierarchy.get(int(label), {}).items()
    parents = [parent for parent, predicate in relations if predicate == "is_a" or predicate == "part_of"]

    return functools.reduce(lambda acc, parent: acc.union(trace(parent)), parents, {int(label)})


def expand(y):
    aux = lambda labels: functools.reduce(lambda acc, parent: acc.union(trace(parent)), labels, set())

    return list(map(lambda labels: aux(labels), y))


def hp_hr_f1(classifier, X_test, y_true):
    y_pred, _ = classifier.predict(X_test)

    y_true = expand(y_true)
    y_pred = expand(y_pred)

    hp = hierarchical_precision(y_true, y_pred)
    hr = hierarchical_recall(y_true, y_pred)

    f1 = 0 if not hp and not hr else 2 * (hp * hr) / (hp + hr)

    return {"hp": hp, "hr": hr, "f1": f1}


def main(connection, method):
    global hierarchy

    client = DatabaseClient(connection)
    memory = Memory(MEMORY_DIR, verbose=0)

    hierarchy = client.get_relations()

    contents, _ = client.get_citations()
    _, annotations = client.get_annotations_in("citations")

    if method == "bert":
        cursor = connection.execute(SELECT_CONTENT_EMBEDDING_MAPPING)
        mapping = {content: pickle.loads(data) for content, data in cursor}

        transformer = MockVectorizer(mapping)
    else:
        transformer = TfidfVectorizer(tokenizer=tokenize, token_pattern=None)

    classifier = MultiLabelNearestNeighborsClassifier(
        n_neighbors=N_NEIGHBORS,
        e_amplification=E_AMPLIFICATION,
        n_jobs=N_JOBS,
        memory=memory,
    )

    pipeline = Pipeline([("transformer", transformer), ("classifier", classifier)], memory=memory)
    kfold = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    gridsearch = GridSearchCV(pipeline, param_grid=PARAM_GRID, scoring=hp_hr_f1, refit=False, cv=kfold, verbose=10)

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
