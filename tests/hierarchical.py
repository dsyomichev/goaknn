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

from tests.util.metrics import mean_reciprocal_rank, macro_average_recall
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

relations = {}


@functools.cache
def relatives(child):
    relation = relations.get(int(child), {})

    parents = set()

    for parent, predicate in relation.items():
        if not predicate == "is a" and not predicate == "part of":
            continue

        parents.add(int(parent))

    return parents


@functools.cache
def trace(child):
    lineage = {int(child)}
    parents = relatives(child)

    while len(parents) > 0:
        lineage |= parents
        parents = functools.reduce(lambda acc, parent: acc.union(relatives(parent)), parents, set()) - lineage

    return lineage


def expand(y):
    return list(map(lambda yi: list(map(lambda yii: trace(yii), yi)), y))


def hierarchical_p_r_f(classifier, X_test, y_true):
    y_pred, _ = classifier.predict(X_test)

    n = len(y_true)

    hp_ag = 0
    hr_ag = 0
    hf_ag = 0

    for y_true_expanded, y_pred_expanded in zip(expand(y_true), expand(y_pred)):
        expansion_precisions = []
        expansion_recalls = []

        for y_pred_lineage in y_pred_expanded:
            pool = []

            for y_true_lineage in y_true_expanded:
                intersect = y_pred_lineage & y_true_lineage

                pool.append(len(intersect) / len(y_pred_lineage))

            expansion_precisions.append(max(pool))

        for y_true_lineage in y_true_expanded:
            pool = []

            for y_pred_lineage in y_pred_expanded:
                intersect = y_pred_lineage & y_true_lineage

                pool.append(len(intersect) / len(y_true_lineage))

            expansion_recalls.append(max(pool))

        hpi = 0 if not expansion_precisions else sum(expansion_precisions) / len(y_pred_expanded)
        hri = 0 if not expansion_recalls else sum(expansion_recalls) / len(y_true_expanded)
        hfi = 0 if not hpi and not hri else 2 * (hpi * hri) / (hpi + hri)

        hp_ag += hpi
        hr_ag += hri
        hf_ag += hfi

    hp = 0 if not n else hp_ag / n
    hr = 0 if not n else hr_ag / n
    hf = 0 if not n else hf_ag / n

    return {"hp": hp, "hr": hr, "hf": hf}


def main(connection, method):
    global relations

    client = DatabaseClient(connection)
    memory = Memory(MEMORY_DIR, verbose=0)

    relations = client.get_relations()

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

    steps = [("transformer", transformer), ("classifier", classifier)]
    pipeline = Pipeline(steps, memory=memory if method == "tfidf" else None)

    kfold = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    gridsearch = GridSearchCV(
        pipeline,
        param_grid=PARAM_GRID,
        scoring=hierarchical_p_r_f,
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
