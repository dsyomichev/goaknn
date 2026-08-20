import json
import pickle
import warnings
import numpy

from joblib import Memory
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.base import BaseEstimator
from pathlib import Path
from sklearn.utils.validation import check_is_fitted

from lib.OntologyEstimator import OntologyEstimator
from lib.metrics import mean_reciprocal_rank, macro_average_recall

from src.db.DatabaseClient import DatabaseClient
from src.db.execute import execute

from tests.util.default import default

SCRIPT_NAME = Path(__file__).stem

ROOT_DIR = Path(__file__, "../..").resolve()
CACHE_DIR = ROOT_DIR.joinpath(f"data/cache/{SCRIPT_NAME}/").resolve()

MEMORY_DIR = CACHE_DIR.joinpath("memory/").resolve()
OUTPUT_FILE = CACHE_DIR.joinpath("output.json").resolve()

N_LABELS = 20
N_JOBS = -1
N_SPLITS = 10

PARAM_GRID = {"estimator__n_neighbors": [25, 50, 100, 200], "estimator__e_amplification": [1, 3, 5]}

RANDOM_STATE = 0

SELECT_CONTENT_EMBEDDING_MAPPING = """
    SELECT citations.content, embeddings.data
    FROM citations JOIN embeddings ON citations.id = embeddings.id
"""

warnings.filterwarnings("ignore")


def score_custom(estimator, X_test, y_true):
    y_pred, _ = estimator.predict(X_test)

    mrr = mean_reciprocal_rank(y_true, y_pred)
    mar = macro_average_recall(y_true, y_pred)

    fscore = 0 if not mrr and not mar else 2 * (mrr * mar) / (mrr + mar)

    return {"mrr": mrr, "mar": mar, "f": fscore}


def main(connection):
    class MockBertVectorizer(BaseEstimator):

        def __init__(self):
            contents_embeddings = connection.execute(SELECT_CONTENT_EMBEDDING_MAPPING)
            self.mapping = {content: pickle.loads(data) for content, data in contents_embeddings}

        def __sklearn_is_fitted__(self):
            return hasattr(self, "_is_fitted")

        def fit(self, X, y=None):
            self._is_fitted = True

            return self

        def transform(self, X):
            check_is_fitted(self)

            return numpy.array([self.mapping[text] for text in X])

        def fit_transform(self, X, y=None):
            return self.fit(X, y).transform(X)

    client = DatabaseClient(connection)

    contents, _ = client.get_citations()
    _, annotations = client.get_annotations_in("citations")

    memory = Memory(MEMORY_DIR, verbose=0)

    m_neighbors = max(PARAM_GRID["estimator__n_neighbors"])

    transformer = MockBertVectorizer()
    estimator = OntologyEstimator(n_labels=N_LABELS, n_jobs=N_JOBS, m_neighbors=m_neighbors, memory=memory)

    pipeline = Pipeline([("transformer", transformer), ("estimator", estimator)])
    kfold = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    gridsearch = GridSearchCV(pipeline, param_grid=PARAM_GRID, scoring=score_custom, refit=False, cv=kfold, verbose=10)

    gridsearch.fit(contents, annotations)

    with open(OUTPUT_FILE, "w") as file:
        json.dump(gridsearch.cv_results_, file, default=default, indent=4)


if __name__ == "__main__":
    execute(main)
