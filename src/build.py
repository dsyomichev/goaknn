import dotenv

from sqlite3 import Connection

from data.build import annotations, corpus, ontology
from lib.util.execute import execute

dotenv.load_dotenv()


def proxy(connection: Connection):
    annotations.extract(connection)
    corpus.extract(connection)
    ontology.extract(connection)


execute(proxy)
