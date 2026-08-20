from sklearn.base import BaseEstimator
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.validation import check_is_fitted, check_memory


def compute(X_train, X_test, n_neighbors, n_jobs):
    estimator = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=n_jobs, metric="cosine")

    estimator.fit(X_train)

    return estimator.kneighbors(X_test)


class MultiLabelNearestNeighborsClassifier(BaseEstimator):

    def __init__(self, n_neighbors=5, m_neighbors=None, n_labels=None, e_amplification=1, n_jobs=None, memory=None):
        self.n_neighbors = n_neighbors
        self.m_neighbors = m_neighbors
        self.n_labels = n_labels
        self.e_amplification = e_amplification
        self.n_jobs = n_jobs
        self.memory = memory

    def __sklearn_is_fitted__(self):
        return hasattr(self, "_is_fitted")

    def fit(self, X_train, y_train):
        self.X_train_ = X_train
        self.y_train_ = y_train

        self.m_neighbors_ = self.n_neighbors if not self.m_neighbors else self.m_neighbors

        self.compute_ = check_memory(self.memory).cache(compute, ignore=["n_jobs"]) if self.memory else compute

        self._is_fitted = True

        return self

    def predict(self, X_test):
        check_is_fitted(self)

        distances, indices = self.compute_(self.X_train_, X_test, self.m_neighbors_, self.n_jobs)

        y_pred, sim_pred = [], []

        for pred_distances, pred_indices in zip(distances, indices):
            result = {}

            similarities = (1 - pred_distances[: self.n_neighbors]) ** self.e_amplification

            for i, similarity in zip(pred_indices[: self.n_neighbors], similarities):
                for label in self.y_train_[i]:
                    result[label] = result.get(label, 0) + similarity

            sorted_result = sorted(result.items(), key=lambda item: item[1], reverse=True)
            trunc_result = sorted_result if not self.n_labels else sorted_result[: self.n_labels]

            y_pred.append([labels for (labels, _) in trunc_result])
            sim_pred.append([similarity for (_, similarity) in trunc_result])

        return y_pred, sim_pred
