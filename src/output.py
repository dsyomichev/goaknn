import sys
import csv

DELIMITER = ","
LINE_TERMINATOR = "\n"


def output(term_ids, term_scores, labels):

    writer = csv.writer(sys.stdout, delimiter=DELIMITER, lineterminator=LINE_TERMINATOR)

    for i, (ids, scores) in enumerate(zip(term_ids, term_scores)):
        for id, score in zip(ids, scores):
            writer.writerow([i, id, labels[id], score])
