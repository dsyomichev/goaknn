import torch

from transformers import AutoModel, AutoTokenizer
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted


class BertVectorizer(BaseEstimator):

    def __init__(self, model_name, device_name=None):
        self.model_name = model_name
        self.device_name = device_name

    def __sklearn_is_fitted__(self):
        return hasattr(self, "_is_fitted")

    def fit(self, X, y=None):
        if self.device_name:
            self.device_ = torch.device(self.device_name)
        else:
            self.device_ = torch.accelerator.current_accelerator(check_available=True) or torch.device("cpu")

        self.tokenizer_ = AutoTokenizer.from_pretrained(self.model_name)
        self.model_ = AutoModel.from_pretrained(self.model_name).eval()

        self.model_.to(self.device_)

        self.max_length_ = self.model_.config.max_position_embeddings

        self._is_fitted = True

        return self

    def transform(self, X):
        check_is_fitted(self)

        tokens = self.tokenizer_(X, return_tensors="pt", truncation=True, padding=True, max_length=self.max_length_)

        tokens.to(self.device_)

        with torch.inference_mode():
            hidden = self.model_(**tokens).last_hidden_state

        att_mask = tokens["attention_mask"].unsqueeze(-1)
        masked_hidden = hidden * att_mask
        results = masked_hidden.sum(axis=1) / att_mask.sum(axis=1)

        return results.cpu().numpy()

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)
