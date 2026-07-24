import re
import nltk

from nltk.corpus import stopwords
from nltk.downloader import Downloader
from nltk.stem import PorterStemmer

if Downloader().status("stopwords") != Downloader.INSTALLED:
    nltk.download("stopwords")

TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9]+")
NUMERICAL_PATTERN = re.compile(r"[0-9]+")
STOPWORDS = set(stopwords.words("english"))

stemmer = PorterStemmer()


def tokenize(input: str):
    tokens = TOKEN_PATTERN.findall(input.lower())
    tokens = [token for token in tokens if token not in STOPWORDS and not NUMERICAL_PATTERN.match(token)]
    tokens = [stemmer.stem(token) for token in tokens]

    return tokens


def index(tokens: list[str]):

    frequencies: dict[str, int] = {}
    for token in tokens:
        frequencies[token] = frequencies.get(token, 0) + 1

    return frequencies.items()
