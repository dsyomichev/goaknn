import sys
import warnings

from sqlite3 import Connection
from sklearn.feature_extraction.text import TfidfVectorizer
from pathlib import Path
from scipy.sparse import sparray  # type: ignore

# Script Configuration

ROOT_DIRECTORY = Path(__file__, "../..").resolve()

# Import Local Modules

sys.path.insert(0, str(ROOT_DIRECTORY))

from lib.util.execute import execute
from lib.util.tokenizer import tokenizer
from lib.util.stop_words import stop_words
from lib.DatabaseClient import DatabaseClient

warnings.filterwarnings("ignore")


def main(connection: Connection):
    client = DatabaseClient(connection)

    X_, y = client.get_vectorizations()

    print(X_.shape, y.shape)


"""
    X, y = client.get_citations()

    vectorizer = TfidfVectorizer(tokenizer=tokenizer, stop_words=stop_words, min_df=5)

    client.get_vectorizer(vectorizer)

    X_: sparray = vectorizer.transform(X)  # type: ignore

    client.set_vectorizations(X_, y)
"""


if __name__ == "__main__":
    execute(main)
