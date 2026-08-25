import json
import warnings
import sys
import pickle

from joblib import Memory
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from argparse import ArgumentParser

from lib.OntologyClassifier import OntologyClassifier
from lib.preprocess import tokenize

from src.db.DatabaseClient import DatabaseClient
from src.db.execute import execute

from tests.util.metrics import hp_hr_hf, mrr_mar
from tests.util.serializers import primitive
from tests.util.mock import MockVectorizer, SELECT_CONTENT_EMBEDDING_MAPPING

CACHE_DIR = Path(__file__, "../../data/cache/").resolve()
MEMORY_DIR = CACHE_DIR.joinpath("memory/").resolve()

PARAM_GRID = {
    "classifier__n_neighbors": [25, 50, 100, 200],
    "classifier__e_amplification": [1, 3, 5],
    "classifier__n_labels": [5, 10, 20],
}

N_JOBS = -1
N_SPLITS = 10

RANDOM_STATE = 0


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


def main(connection, method, metric):
    client = DatabaseClient(connection)
    memory = Memory(MEMORY_DIR, verbose=0)

    if metric == "hierarchical":
        relations = client.get_relations()
        custom_score = init_hierarchical_score(relations)
    else:
        custom_score = exact_score

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
    scaler = StandardScaler(with_std=False) if method == "bert" else "passthrough"

    steps = [("transformer", transformer), ("scaler", scaler), ("classifier", classifier)]
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

    output_file = Path.joinpath(CACHE_DIR, f"hyperparameters_{method}_{metric}.json")
    with open(output_file, "w") as file:
        json.dump(gridsearch.cv_results_, file, default=primitive, indent=4)


if __name__ == "__main__":

    warnings.filterwarnings("ignore")

    parser = ArgumentParser()
    parser.add_argument("method", choices=["tfidf", "bert"])
    parser.add_argument("metric", choices=["exact", "hierarchical"])

    namespace = parser.parse_args(sys.argv[1:])

    execute(main, method=namespace.method, metric=namespace.metric)
