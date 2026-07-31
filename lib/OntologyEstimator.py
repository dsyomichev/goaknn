import numpy

from sklearn.base import BaseEstimator
from sklearn.neighbors import NearestNeighbors
from typing import Any
from numpy import float64, int32
from numpy.typing import NDArray
from scipy.sparse import sparray  # type: ignore

from lib.DatabaseClient import DatabaseClient


class OntologyEstimator(BaseEstimator):

    def __init__(self, client: DatabaseClient, **kwargs: Any):
        self.estimator = NearestNeighbors(**kwargs)
        self.client = client

    def fit(self, X: NDArray[float64] | sparray, y: NDArray[int32]):
        self.y = y

        self.estimator.fit(X)  # type: ignore

        return self

    def predict(self, X: NDArray[float64] | sparray, **kwargs: Any):
        distances, indices = self.estimator.kneighbors(X, **kwargs)  # type: ignore

        predictions: list[NDArray[int32]] = []
        scores: list[NDArray[float64]] = []

        for distances_, indices_ in zip(distances, indices):
            result: dict[int, float] = {}

            citation_ids = self.y[indices_]
            term_ids = self.client.get_annotations(citation_ids)

            distances_ = (distances_ - distances_.min()) / (distances_.max() - distances_.min())

            for distance, term_ids_ in zip(distances_, term_ids):
                weight = 1 - distance

                for term_id in term_ids_:
                    result[term_id] = result[term_id] + weight if term_id in result else weight

            ordered = sorted(result.items(), key=lambda item: item[1], reverse=True)
            predictions.append(numpy.array([term_id for (term_id, _) in ordered], dtype=int32))
            scores.append(numpy.array([score for (_, score) in ordered], dtype=float64))

        return predictions, scores
