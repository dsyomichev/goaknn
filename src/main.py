import sys
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

from lib.OntologyClassifier import OntologyClassifier
from lib.BertEmbeddingsVectorizer import BertEmbeddingsVectorizer
from lib.preprocess import tokenize

from src.db.DatabaseClient import DatabaseClient
from src.db.execute import execute
from src.build import terms, annotations, citations
from src.parser import parser
from src.output import output

BATCH_SIZE = 1

MODEL_NAME = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"

BUILD_ORDER = {"terms": 0, "annotations": 1, "citations": 2}
BUILD_MAPPING = {"terms": terms.main, "annotations": annotations.main, "citations": citations.main}

STDIN_DOC_SEP = r"\n\s*\n"

N_NEIGHBORS = 200
N_LABELS = 20
N_JOBS = -1


def main(connection):
    args = parser.parse_args(sys.argv[1:])

    client = DatabaseClient(connection)

    if args.command == "build":
        if not args.target:
            args.target = ["terms", "annotations", "citations"]

        args.target = sorted(args.target, key=lambda item: BUILD_ORDER[item])

        for table in args.target:
            BUILD_MAPPING[table](connection)

    if args.command == "index":
        if args.method == "tfidf":
            transformer = TfidfVectorizer(tokenizer=tokenize, token_pattern=None)

            contents, ids = client.get_citations()

            transformer.fit(contents, ids)

            client.delete_vocabulary()
            client.set_vocabulary(transformer.vocabulary_, transformer.idf_)

            vectors = transformer.transform(contents)

            client.delete_vectorizations()
            client.set_vectorizations(vectors, ids)

        else:
            transformer = BertEmbeddingsVectorizer(model_name=MODEL_NAME)

            client.set_embeddings([], [])

            contents, ids = client.get_citations_not_in("embeddings")

            transformer.fit(None)

            for i in range(0, len(ids), BATCH_SIZE):
                batch_ids = ids[i : i + BATCH_SIZE]
                batch_contents = contents[i : i + BATCH_SIZE]

                batch_vectors = transformer.transform(batch_contents)

                client.set_embeddings(batch_vectors, batch_ids)

    if args.command == "query":
        if not args.target:
            test_inputs = re.split(STDIN_DOC_SEP, sys.stdin.read())
            test_sanitized = [target.strip() for target in test_inputs]

            args.target = [target for target in test_sanitized if target]

            if not args.target:
                return

        if args.method == "tfidf":
            transformer = TfidfVectorizer(tokenizer=tokenize, token_pattern=None)

            vocabulary, idf = client.get_vocabulary()
            vectors, ids = client.get_vectorizations()
            _, annotations = client.get_annotations_in("vectorizations")

            transformer.vocabulary_ = vocabulary
            transformer.idf_ = idf

        else:
            scaler = StandardScaler(with_std=False)
            transformer = BertEmbeddingsVectorizer(model_name=MODEL_NAME).fit([])

            vectors, ids = client.get_embeddings()
            _, annotations = client.get_annotations_in("embeddings")

            vectors = scaler.fit_transform(vectors)

        classifier = OntologyClassifier(n_neighbors=N_NEIGHBORS, n_labels=N_LABELS, n_jobs=N_JOBS)

        classifier.fit(vectors, annotations)

        if args.method == "tfidf":
            test_vectors = transformer.transform(args.target)

        else:
            test_vectors = transformer.transform(args.target)
            test_vectors = scaler.transform(test_vectors)

        pred_term_ids, pred_term_scores = classifier.predict(test_vectors)

        term_label_mapping = client.get_terms()

        output(pred_term_ids, pred_term_scores, term_label_mapping)


if __name__ == "__main__":
    execute(main)
