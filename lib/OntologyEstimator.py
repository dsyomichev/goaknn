from numpy import array, pad, iinfo, finfo, float64, int_
from numpy.ma import masked_equal
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import precision_recall_fscore_support  # type: ignore
from itertools import chain

from sklearn.base import BaseEstimator
from typing import Any, Self, Generator
from numpy import ndarray, dtype
from numpy.ma import masked_array
from scipy.sparse import sparray  # type: ignore


class OntologyEstimator(BaseEstimator):

    def __init__(
        self,
        t_threshold: float = 0.0,
        w_exponent: int = 1,
        **kwargs: Any,
    ) -> None:
        self.t_threshold = t_threshold
        self.w_exponent = w_exponent

        self.estimator = NearestNeighbors(metric="cosine", **kwargs)

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
    ) -> tuple[
        masked_array[tuple[int, int], dtype[int_]],
        masked_array[tuple[int, int], dtype[float64]],
    ]:
        distances, indices = self.estimator.kneighbors(X_)  # type: ignore

        Z_: list[ndarray[tuple[int, int], dtype[int_]]] = []
        T: list[ndarray[tuple[int, int], dtype[float64]]] = []

        width = 0

        for distances_, indices_ in zip(distances, indices):
            result: dict[int, float] = {}

            weights = (1 - distances_) ** self.w_exponent

            aggregate = weights.sum()

            for weight, term_ids_ in zip(weights, self.Z[indices_]):
                for term_id in term_ids_.compressed():
                    result[term_id] = (
                        result[term_id] + weight if term_id in result else weight
                    )

            ordered = sorted(
                ((term_id, score / aggregate) for (term_id, score) in result.items()),
                key=lambda item: item[1],
                reverse=True,
            )

            ordered = [
                (term_id, score / aggregate)
                for (term_id, score) in ordered
                if score >= self.t_threshold
            ]

            Z_.append(array([term_id for (term_id, _) in ordered], dtype=int_))
            T.append(array([score for (_, score) in ordered], dtype=float64))

            width = len(ordered) if len(ordered) > width else width

        Z_ = [pad(z, (0, width - len(z)), constant_values=iinfo(int_).min) for z in Z_]  # type: ignore
        T = [pad(t, (0, width - len(t)), constant_values=finfo(float64).min) for t in T]  # type: ignore

        return masked_equal(Z_, iinfo(int_).min), masked_equal(T, finfo(float64).min)

    def evaluate(
        self,
        Z: masked_array[tuple[int, int], dtype[int_]],
        Z_: masked_array[tuple[int, int], dtype[int_]],
    ) -> tuple[float, float, float]:

        expected: Generator[ndarray[tuple[int], dtype[int_]]] = (z.compressed() for z in Z)  # type: ignore
        predicted: Generator[ndarray[tuple[int], dtype[int_]]] = (z_.compressed() for z_ in Z_)  # type: ignore

        binarizer = MultiLabelBinarizer(sparse_output=True).fit(chain(predicted, expected))  # type: ignore

        precision, recall, fscore, _ = precision_recall_fscore_support(
            binarizer.transform(expected),  # type: ignore
            binarizer.transform(predicted),  # type: ignore
            average="samples",
            zero_division=0,
        )

        return precision, recall, fscore  # type: ignore

    def score(
        self,
        X_: ndarray[tuple[int, int], dtype[float64]] | sparray,
        Z: masked_array[tuple[int, int], dtype[int_]],
    ) -> float:
        Z_, _ = self.predict(X_)

        _, _, fscore = self.evaluate(Z, Z_)

        return fscore
