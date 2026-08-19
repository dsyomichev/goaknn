import numpy
import torch

from transformers import AutoModel, AutoTokenizer
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted


class BertVectorizer(BaseEstimator):

    def __init__(self, model_name):
        self.model_name = model_name

    def __sklearn_is_fitted__(self):
        return hasattr(self, "tokenizer") and hasattr(self, "model") and self._is_fitted

    def fit(self, X, y=None):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name).eval()

        self.max_length = self.model_.config.max_position_embeddings

        self._is_fitted = True

        return self

    def transform(self, X):
        check_is_fitted(self)

        result = []

        for sample in X:
            tokens = self.tokenizer_(sample, return_tensors="pt", truncation=True, max_length=self.max_length)

            with torch.inference_mode():
                output = self.model_(**tokens)

            vectors = output.last_hidden_state.squeeze(0)
            result.append(vectors.numpy().mean(axis=0))

        return numpy.array(result)

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)
