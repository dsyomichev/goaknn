from nltk.tokenize import regexp_tokenize  # type: ignore
from nltk.stem import PorterStemmer  # type: ignore

TOKEN_PATTERN = r"(?=[a-zA-Z0-9]*[a-zA-Z])[a-zA-Z0-9]+"

stemmer = PorterStemmer()


def tokenizer(input: str):
    tokens: list[str] = regexp_tokenize(input, pattern=TOKEN_PATTERN)  # type: ignore
    tokens = [stemmer.stem(token) for token in tokens]  # type: ignore

    return tokens
