import joblib
import warnings

from joblib import Memory
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from pathlib import Path

from lib.OntologyEstimator import OntologyEstimator
from lib.preprocess import tokenize
from lib.scoring import score_mrr, score_mar

from src.db.DatabaseClient import DatabaseClient
from src.db.execute import execute

SCRIPT_NAME = Path(__file__).stem

ROOT_DIR = Path(__file__, "../..").resolve()
CACHE_DIR = ROOT_DIR.joinpath(f"data/cache/{SCRIPT_NAME}/").resolve()

MEMORY_DIR = CACHE_DIR.joinpath("memory/").resolve()
OUTPUT_FILE = CACHE_DIR.joinpath(f"{SCRIPT_NAME}.pkl").resolve()

N_LABELS = 20
N_JOBS = -1
N_SPLITS = 10

PARAM_GRID = {"estimator__n_neighbors": [25, 50, 100, 200]}

RANDOM_STATE = 0

warnings.filterwarnings("ignore")


def score_custom(estimator, X_test, y_true):
    y_pred, _ = estimator.predict(X_test)

    mrr = score_mrr(y_true, y_pred)
    mar = score_mar(y_true, y_pred)

    fscore = 0 if not mrr and not mar else 2 * (mrr * mar) / (mrr + mar)

    return {"mrr": mrr, "mar": mar, "fscore": fscore}


def main(connection):
    client = DatabaseClient(connection)

    contents, _ = client.get_citations()
    _, annotations = client.get_annotations("citations")

    memory = Memory(MEMORY_DIR, verbose=0)
    max_neighbors = max(PARAM_GRID["estimator__n_neighbors"])

    transformer = TfidfVectorizer(tokenizer=tokenize, token_pattern=None)
    estimator = OntologyEstimator(n_labels=N_LABELS, n_jobs=N_JOBS, max_neighbors=max_neighbors, memory=memory)

    pipeline = Pipeline([("transformer", transformer), ("estimator", estimator)], memory=memory)
    kfold = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    gridsearch = GridSearchCV(pipeline, param_grid=PARAM_GRID, scoring=score_custom, refit=False, cv=kfold, verbose=10)

    gridsearch.fit(contents, annotations)

    joblib.dump(gridsearch, OUTPUT_FILE)


if __name__ == "__main__":
    execute(main)
