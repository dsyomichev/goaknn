from numpy import array, float64, int_
from sklearn.neighbors import NearestNeighbors
from lib.DatabaseClient import DatabaseClient

from sklearn.base import BaseEstimator
from typing import Any, Self
from numpy import ndarray, dtype
from numpy.ma import masked_array
from scipy.sparse import sparray  # type: ignore


class OntologyEstimator(BaseEstimator):

    def __init__(
        self,
        client: DatabaseClient,
        **kwargs: Any,
    ) -> None:
        self.estimator = NearestNeighbors(**kwargs)
        self.client = client

    def fit(
        self,
        X: ndarray[tuple[int, int], dtype[float64]] | sparray,
        Z: masked_array[tuple[int, int], dtype[int_]],
    ) -> Self:
        self.Z = Z

        self.estimator.fit(X)  # type: ignore

        return self

    def predict(
        self,
        X_: ndarray[tuple[int, int], dtype[float64]] | sparray,
        **kwargs: Any,
    ) -> tuple[
        ndarray[tuple[int, int], dtype[int_]],
        ndarray[tuple[int, int], dtype[float64]],
    ]:
        distances, indices = self.estimator.kneighbors(X_, **kwargs)  # type: ignore

        Z_: list[ndarray[tuple[int, int], dtype[int_]]] = []
        scores: list[ndarray[tuple[int, int], dtype[float64]]] = []

        for distances_, indices_ in zip(distances, indices):
            result: dict[int, float] = {}

            distances_ = (distances_ - distances_.min()) / (
                distances_.max() - distances_.min()
            )

            for distance, term_ids_ in zip(distances_, self.Z[indices_]):
                weight = 1 - distance

                for term_id in term_ids_.compressed():
                    result[term_id] = (
                        result[term_id] + weight if term_id in result else weight
                    )

            ordered = sorted(result.items(), key=lambda item: item[1], reverse=True)
            Z_.append(array([term_id for (term_id, _) in ordered], dtype=int_))
            scores.append(array([score for (_, score) in ordered], dtype=float64))

        return array(Z_, dtype=int_), array(scores, dtype=float64)
