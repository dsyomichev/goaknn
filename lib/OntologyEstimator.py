import numpy

from sklearn.base import BaseEstimator
from sklearn.neighbors import NearestNeighbors
from typing import Any
from numpy import ndarray
from scipy.sparse import sparray  # type: ignore

from lib.DatabaseClient import DatabaseClient


class OntologyEstimator(BaseEstimator):

    def __init__(self, client: DatabaseClient, **kwargs: Any):
        self.estimator = NearestNeighbors(**kwargs)
        self.client = client

    def fit(self, X: ndarray | sparray, y: ndarray):
        self.y = y

        self.estimator.fit(X)  # type: ignore

        return self

    def predict(self, X: ndarray | sparray, **kwargs: Any):
        distances, indices = self.estimator.kneighbors(X, **kwargs)  # type: ignore

        predictions: list[list[tuple[int, float]]] = []

        for distance, index in zip(distances, indices):
            result: dict[int, float] = {}

            citation_ids = numpy.array(self.y)[index]
            annotations = self.client.get_annotations(citation_ids)

            for weight, citation_id in zip(distance, citation_ids):
                term_ids = annotations[citation_id]

                for term_id in term_ids:
                    result[term_id] = result[term_id] + 1 / (weight + 1) if term_id in result else 1 / (weight + 1)

            predictions.append(sorted(result.items(), key=lambda item: item[1], reverse=True))

        return predictions
