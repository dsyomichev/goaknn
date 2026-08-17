import warnings
import numpy

from sqlite3 import Connection
from sklearn.feature_extraction.text import TfidfVectorizer

from lib.util.execute import execute
from lib.util.tokenizer import tokenizer
from lib.util.stop_words import stop_words
from lib.DatabaseClient import DatabaseClient
from lib.OntologyEstimator import OntologyEstimator

warnings.filterwarnings("ignore")

TEST_STRING = """
Dihydrofolate reductase has been purified 40-fold to apparent homogeneity from a trimethoprim-resistant strain of Escherichia coli (RT 500) using a procedure that includes methotrexate affinity column chromatography. Determinations of the molecular weight of the enzyme based on its amino acid composition, sedimentation velocity, and sodium dodecyl sulfate gel electrophoresis gave values of 17680, 17470 and 18300, respectively. An aggregated form of the enzyme with a low specific activity can be separated from the monomer by gel filtration; treatment of the aggregate with mercaptoethanol or dithiothreitol results in an increase in enzymic activity and a regeneration of the monomer. Also, multiple molecular forms of the monomer have been detected by polyacrylamide gel electrophoresis. The unresolved enzyme exhibits two pH optima (pH 4.5 and pH 7.0) with dihydrofolate as a substrate. Highest activities are observed in buffers containing large organic cations. In 100 mM imidazolium chloride (pH 7), the specific activity is 47 mumol of dihydrofolate reduced per min per mg at 30 degrees. Folic acid also serves as a substrate with a single pH optimum of pH 4.5. At this pH the Km for folate is 16 muM, and the Vmax is 1/1000 of the rate observed with dihydrofolate as the substrate. Monovalent cations (Na+, K+, Rb+, and Cs+) inhibit dihydrofolate reductase; at a given ionic strength the degree of inhibition is a function of the ionic radius of the cation. Divalent cations are more potent inhibitors; the I50 of BaCl2 is 250 muM, as compared to 125 mM for KCl. Anions neither inhibit nor activate the enzyme.
"""


def main(connection: Connection):
    client = DatabaseClient(connection)

    transformer = TfidfVectorizer(tokenizer=tokenizer, stop_words=stop_words)
    estimator = OntologyEstimator(n_neighbors=5, n_labels=5, n_jobs=-1)

    vocabulary, idf = client.get_vocabulary()

    transformer.vocabulary_ = vocabulary
    transformer.idf_ = idf

    citation_vectors, vector_citation_ids = client.get_vectorizations()
    annotation_citation_ids, citation_annotations = client.get_annotations()

    intersection_idxs = numpy.nonzero(numpy.isin(vector_citation_ids, annotation_citation_ids))[0]
    avail_citation_vectors = citation_vectors[intersection_idxs]
    avail_citation_annot = [citation_annotations[i] for i in intersection_idxs]

    estimator.fit(avail_citation_vectors, avail_citation_annot)

    predictions, scores = estimator.predict(transformer.transform([TEST_STRING]))

    print(predictions, scores)


if __name__ == "__main__":
    execute(main)
