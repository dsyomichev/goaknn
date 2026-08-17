import numpy
import sklearn.metrics

from sklearn.base import BaseEstimator
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MultiLabelBinarizer


class OntologyEstimator(BaseEstimator):

    def __init__(self, n_neighbors, n_jobs):
        self.estimator = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=n_jobs, metric="cosine")

    def fit(self, X_train, y_train):
        self.y_train = y_train

        self.estimator.fit(X_train)

        return self

    def predict(self, X_test):
        distances, indicies = self.estimator.kneighbors(X_test)
        y_pred, sim_pred = [], []

        for pred_distances, pred_indicies in zip(distances, indicies):
            result = {}

            similarities = 1 - pred_distances

            for i, similarity in zip(pred_indicies, similarities):
                for label in self.y[i]:
                    result[label] = result.get(label, 0) + similarity

            sorted_result = sorted(result.items(), lambda item: item[1], reverse=True)

            y_pred.append([labels for (labels, _) in sorted_result])
            sim_pred.append([similarity for (_, similarity) in sorted_result])

        return y_pred, sim_pred

    def evaluate(self, y_true, y_pred):
        reciprocal_ranks = [0] * len(y_true)

        for i, (labels_true, labels_pred) in enumerate(zip(y_true, y_pred)):
            for j, label in enumerate(labels_pred):
                if label in labels_true:
                    reciprocal_ranks[i] = 1 / (j + 1)

                    break

        mrr = numpy.mean(reciprocal_ranks)

        binarizer = MultiLabelBinarizer()
        binarizer.fit([*y_true, *y_pred])

        recall = sklearn.metrics.recall_score(
            binarizer.transform(y_true),
            binarizer.transform(y_pred),
            average="macro",
            zero_division=0.0,
        )

        fscore = 2 * (mrr * recall) / (mrr + recall)

        return fscore, mrr, recall

    def score(self, X_test, y_true):
        y_pred = self.predict(X_test)

        fscore, _, _ = self.evaluate(y_true, y_pred)

        return fscore
