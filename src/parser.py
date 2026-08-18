from argparse import ArgumentParser

parser = ArgumentParser()

sub_act = parser.add_subparsers(dest="command")

build_parser = sub_act.add_parser("build")
index_parser = sub_act.add_parser("index")
query_parser = sub_act.add_parser("query")

build_parser.add_argument("target", choices=["annotations", "corpus", "ontology"], nargs="*")

index_parser.add_argument("--method", choices=["tfidf", "bert"], default="tfidf")

query_parser.add_argument("--method", choices=["tfidf", "bert"], default="tfidf")
query_parser.add_argument("target", nargs="*")
