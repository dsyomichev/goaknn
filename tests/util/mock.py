import numpy

from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted

SELECT_CONTENT_EMBEDDING_MAPPING = """
    SELECT citations.content, embeddings.data
    FROM citations JOIN embeddings ON citations.id = embeddings.id
"""


class MockVectorizer(BaseEstimator):

    def __init__(self, mapping):
        self.mapping = mapping

    def __sklearn_is_fitted__(self):
        return hasattr(self, "_is_fitted")

    def fit(self, X, y=None):
        self._is_fitted = True

        return self

    def transform(self, X):
        check_is_fitted(self)

        return numpy.array([self.mapping[text] for text in X])

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)
