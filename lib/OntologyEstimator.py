import functools

from sklearn.base import BaseEstimator
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.validation import check_is_fitted, check_memory

from lib.scoring import score_mrr, score_mar


def evaluate(X_train, X_test, n_neighbors, n_jobs):
    estimator = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=n_jobs, metric="cosine")

    estimator.fit(X_train)

    return estimator.kneighbors(X_test)


class OntologyEstimator(BaseEstimator):

    def __init__(self, n_neighbors=5, n_labels=None, n_jobs=-1, max_neighbors=None, memory=None):
        self.n_neighbors = n_neighbors
        self.n_labels = n_labels
        self.n_jobs = n_jobs

        self.max_neighbors = n_neighbors if not max_neighbors else max_neighbors
        self.memory = memory

    def __sklearn_is_fitted__(self):
        return hasattr(self, "X_train") and hasattr(self, "y_train") and self._is_fitted

    def fit(self, X_train, y_train):
        if self.memory:
            check_memory(self.memory)

        self.X_train = X_train
        self.y_train = y_train

        self._is_fitted = True

        return self

    def predict(self, X_test):
        check_is_fitted(self)

        evaluate_proxy = self.memory.cache(evaluate, ignore=["n_jobs"]) if self.memory else evaluate
        distances, indices = evaluate_proxy(self.X_train, X_test, self.max_neighbors, self.n_jobs)

        y_pred, sim_pred = [], []

        for pred_distances, pred_indices in zip(distances, indices):
            result = {}

            similarities = (1 - pred_distances[: self.n_neighbors]) ** 3

            for i, similarity in zip(pred_indices[: self.n_neighbors], similarities):
                for label in self.y_train[i]:
                    result[label] = result.get(label, 0) + similarity

            sorted_result = sorted(result.items(), key=lambda item: item[1], reverse=True)
            trunc_result = sorted_result if not self.n_labels else sorted_result[: self.n_labels]

            y_pred.append([labels for (labels, _) in trunc_result])
            sim_pred.append([similarity for (_, similarity) in trunc_result])

        return y_pred, sim_pred

    def score(self, X_test, y_true):
        check_is_fitted(self)

        y_pred, _ = self.predict(X_test)

        mrr = score_mrr(y_true, y_pred)
        mar = score_mar(y_true, y_pred)

        return 0 if not mrr and not mar else 2 * (mrr * mar) / (mrr + mar)
