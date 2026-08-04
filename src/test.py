import warnings

from numpy import isin, nonzero, iinfo, int_
from joblib import dump, Memory  # type: ignore
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from pathlib import Path

from sqlite3 import Connection
from typing import Any

from lib.util.execute import execute
from lib.util.tokenizer import tokenizer
from lib.util.stop_words import stop_words
from lib.DatabaseClient import DatabaseClient
from lib.OntologyEstimator import OntologyEstimator

from lib.BertVectorizer import BertVectorizer

warnings.filterwarnings("ignore")

# Script Configuration

ROOT_DIRECTORY = Path(__file__, "../..").resolve()
CACHE_DIRECTORY = ROOT_DIRECTORY.joinpath("data/cache").resolve()

PARAMETER_GRID: dict[str, list[Any]] = {
    "estimator__t_threshold": [0.0, 0.05, 0.1, 0.15, 0.2, 0.25],
    "estimator__w_exponent": [1, 2, 3, 4, 5],
    "estimator__n_neighbors": [5, 10, 15, 20, 25],
}


def search(
    connection: Connection,
    vectorizer: TfidfVectorizer | BertVectorizer,
    **kwargs: Any,
) -> GridSearchCV:
    client = DatabaseClient(connection)

    X, y_ = client.get_citations()
    y, Z = client.get_annotations()

    X = [X[i] for i in nonzero(isin(y_, y))[0]]  # type: ignore
    Z = Z[nonzero(isin(y, y_))[0]]  # type: ignore

    location = CACHE_DIRECTORY.joinpath("pipeline/memory").resolve()
    location.mkdir(parents=True, exist_ok=True)

    pipeline = Pipeline(
        [
            ("vectorizer", vectorizer),
            ("estimator", OntologyEstimator(n_jobs=-1)),
        ],
        memory=Memory(location=str(location), verbose=0),
    )

    search = GridSearchCV(pipeline, **kwargs)

    search.fit(X, Z.filled(iinfo(int_).min))  # type: ignore

    return search


def proxy(connection: Connection):
    vectorizer = TfidfVectorizer(tokenizer=tokenizer, stop_words=stop_words)

    result = search(
        connection,
        vectorizer,
        param_grid=PARAMETER_GRID,
        cv=5,
        refit=False,
        verbose=2,
        error_score="raise",
    )

    location = CACHE_DIRECTORY.joinpath("pipeline/gridsearch.pk").resolve()
    location.parent.mkdir(parents=True, exist_ok=True)

    dump(result, location, compress=True)


if __name__ == "__main__":
    execute(proxy)
