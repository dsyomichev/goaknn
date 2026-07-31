from torch import cuda, no_grad
from numpy import array, float64
from transformers import AutoModel, AutoTokenizer

from sklearn.base import BaseEstimator
from collections.abc import Iterable
from typing import Any, Self
from numpy import ndarray, dtype


class BertVectorizer(BaseEstimator):

    def __init__(
        self, model: str, device: str = "cuda" if cuda.is_available() else "cpu"
    ) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model)  # type: ignore
        self.model = AutoModel.from_pretrained(model).to(device).eval()  # type: ignore

    def fit(
        self,
        X: Iterable[str],
        y: Any = None,
    ) -> Self:
        return self

    def transform(
        self,
        X: Iterable[str],
    ) -> ndarray[tuple[int, int], dtype[float64]]:
        result: list[ndarray[tuple[int], dtype[float64]]] = []

        for sample in X:

            tokens = self.tokenizer(sample, return_tensors="pt", truncation=True).to(DEVICE_TYPE)  # type: ignore

            with no_grad():
                output = self.model(**tokens)  # type: ignore

            vectors = output.last_hidden_state.squeeze(0)  # type: ignore

            result.append(vectors.cpu().numpy())  # type: ignore

        return array(result, dtype=float64)

    def fit_tranform(
        self,
        X: Iterable[str],
        y: Any = None,
    ) -> ndarray[tuple[int, int], dtype[float64]]:
        return self.fit(X, y).transform(X)
