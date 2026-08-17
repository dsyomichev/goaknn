import nltk

from nltk.stem import PorterStemmer

TOKEN_PATTERN = r"[a-zA-Z0-9]*[a-zA-Z][a-zA-Z0-9]*"

stemmer = PorterStemmer()

nltk.download("stopwords", quiet=True)


def tokenizer(input):
    tokens = nltk.regexp_tokenize(input, TOKEN_PATTERN)
    return [stemmer.stem(token) for token in tokens]


stop_words = [tokenizer(word) for word in nltk.corpus.stopwords.words("english")]
