import nltk
import functools

from nltk.stem import PorterStemmer

TOKEN_PATTERN = r"[a-zA-Z0-9]*[a-zA-Z][a-zA-Z0-9]*"

stemmer = PorterStemmer()

nltk.download("stopwords", quiet=True)


def normalize(text):
    return text.replace("'", "").lower()


# https://stackoverflow.com/q/1988804
@functools.cache
def stem(word):
    return stemmer.stem(word)


stop_words = {stem(normalize(word)) for word in nltk.corpus.stopwords.words("english")}


def tokenize(text):
    tokenized = nltk.regexp_tokenize(normalize(text), TOKEN_PATTERN)
    stemmed = [stem(token) if token.isalpha() else token for token in tokenized]

    return [token for token in stemmed if token not in stop_words]
