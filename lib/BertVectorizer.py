import torch
import numpy

from sklearn.base import BaseEstimator
from transformers import AutoModel, AutoTokenizer
from collections.abc import Iterable
from typing import Any
from numpy import ndarray


class BertVectorizer(BaseEstimator):

    def __init__(self, model: str, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.tokenizer = AutoTokenizer.from_pretrained(model)  # type: ignore
        self.model = AutoModel.from_pretrained(model).to(device).eval()  # type: ignore

    def fit(self, X: Iterable[str], y: Any = None):
        return self

    def transform(self, X: Iterable[str]):
        result: list[ndarray] = []

        for sample in X:

            tokens = self.tokenizer(sample, return_tensors="pt", truncation=True).to(DEVICE_TYPE)  # type: ignore

            with torch.no_grad():
                output = self.model(**tokens)  # type: ignore

            vectors = output.last_hidden_state.squeeze(0)  # type: ignore

            result.append(vectors.cpu().numpy())  # type: ignore

        return numpy.array(result)

    def fit_tranform(self, X: Iterable[str], y: Any = None):
        return self.fit(X, y).transform(X)
