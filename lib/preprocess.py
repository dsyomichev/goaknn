import re
import nltk

from nltk.corpus import stopwords
from nltk.downloader import Downloader
from nltk.stem import PorterStemmer

if Downloader().status("stopwords") != Downloader.INSTALLED:
    nltk.download("stopwords")

TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9]+")
STOPWORDS = set(stopwords.words("english"))

stemmer = PorterStemmer()


def preprocess(input):
    tokens = TOKEN_PATTERN.findall(input.lower())
    tokens = [token for token in tokens if token not in STOPWORDS]
    tokens = [stemmer.stem(token) for token in tokens]

    return tokens
