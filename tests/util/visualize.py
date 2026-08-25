import numpy
import json
import functools

from matplotlib import pyplot
from pathlib import Path

CACHE_DIR = Path(__file__, "../../../data/cache").resolve()

PARAMETER_KEY_MAP = {"classifier__n_neighbors": "K", "classifier__e_amplification": "Amplification"}


def plot_hyperparameter_comparison(result, n_labels=20):
    figure, axes = pyplot.subplots(1, 2)

    parameters = {}

    for group in result["params"]:
        for key, value in group.items():
            values = parameters.setdefault(key, [])
            values.append(value)

    parameters = {key: numpy.array(value) for key, value in parameters.items()}
    options = {key: numpy.unique(value) for key, value in parameters.items()}

    fixed_n_labels_idxs = numpy.argwhere(parameters["classifier__n_labels"] == n_labels).flatten()

    parameters.pop("classifier__n_labels")

    parameters = {key: value[fixed_n_labels_idxs] for key, value in parameters.items()}

    scores = numpy.array(result["mean_test_hf"])[fixed_n_labels_idxs]
    candidate = numpy.argmax(scores)
    best = {key: value[candidate] for key, value in parameters.items()}

    for i, (axis, parameter) in enumerate(zip(axes, best.keys())):
        fixed = {key: value for key, value in parameters.items() if key != parameter}
        fixed_parameters_idxs = [numpy.argwhere(parameters[key] == best[key]).flatten() for key in fixed.keys()]
        fixed_idxs = functools.reduce(numpy.intersect1d, fixed_parameters_idxs)

        x = parameters[parameter][fixed_idxs]
        y = scores[fixed_idxs]

        title = "".join([f"{PARAMETER_KEY_MAP[key]}={val} " for key, val in best.items() if key != parameter])

        axis.set_title(title, size=10)
        axis.set_xlabel(PARAMETER_KEY_MAP[parameter])
        axis.set_ylabel("HF Score" if not i else "")

        axis.set_xticks(options[parameter])
        axis.set_ylim((numpy.min(scores) - 0.025, numpy.max(scores) + 0.025))

        best_value_idx = numpy.argmax(y)

        axis.plot(x, y, label="HF Score" if not i else "", color="black", marker=".", linestyle="solid")
        axis.plot(x[best_value_idx], y[best_value_idx], color="black", marker="v")

    return figure


def plot_historical_comparison(result_tfidf, result_bert, metric="exact"):
    figure, axes = pyplot.subplots(2, 1)

    x = numpy.array([int(key) for key in result_tfidf.keys()])

    xmin, xmax = numpy.min(x) - 1, numpy.max(x) + 1
    ymin, ymax = 0.1, 0.9

    metric_keys = ["mrr", "mar"] if metric == "exact" else ["hp", "hr"]
    metric_labels = ["MRR", "R20"] if metric == "exact" else ["HP", "HR20"]

    for i, (axis, key) in enumerate(zip(axes, metric_keys)):
        y_tfidf = numpy.array([value[key] for value in result_tfidf.values()])
        y_bert = numpy.array([value[key] for value in result_bert.values()])

        axis.set_xticks(x)

        axis.set_ylabel(metric_labels[0] if key == metric_keys[0] else metric_labels[1])
        axis.set_xlabel("Year" if i else "")
        axis.set_xlim(xmin, xmax)

        axis.plot(x, y_tfidf, label="TF-IDF" if not i else "", color="black", linestyle="solid")
        axis.plot(x, y_bert, label="BERT" if not i else "", color="black", linestyle="dashed")

        axis.hlines(numpy.linspace(ymin, ymax, 10), xmin, xmax, color="gray", linestyle="dotted")

    figure.legend(loc="lower center", ncol=2)

    return figure


def plot_tfidf_versus_bert_hf(result_tfidf, result_bert):

    print(numpy.array(result_tfidf["mean_test_hf"]) < numpy.array(result_bert["mean_test_hf"]))


"""
for method in ["tfidf", "bert"]:
    with open(CACHE_DIR.joinpath(f"hyperparameters_{method}_hierarchical.json")) as file:
        result = json.load(file)

    figure = plot_hyperparameter_comparison(result)

    figure.suptitle("BERT" if method == "bert" else "TF-IDF", size=12)
    figure.set_size_inches((12, 6))

    output_path = Path.joinpath(CACHE_DIR, f"hyperparameters_{method}.png")

    figure.savefig(output_path)
"""

with open(CACHE_DIR.joinpath(f"historical_tfidf_exact.json")) as file:
    result_tfidf = json.load(file)

with open(CACHE_DIR.joinpath(f"historical_bert_exact.json")) as file:
    result_bert = json.load(file)

figure = plot_historical_comparison(result_tfidf, result_bert, metric="exact")

figure.set_size_inches((12, 8))

output_path = Path.joinpath(CACHE_DIR, f"historical_exact.png")

figure.savefig(output_path)

with open(CACHE_DIR.joinpath(f"historical_tfidf_hierarchical.json")) as file:
    result_tfidf = json.load(file)

with open(CACHE_DIR.joinpath(f"historical_bert_hierarchical.json")) as file:
    result_bert = json.load(file)

figure = plot_historical_comparison(result_tfidf, result_bert, metric="hierarchical")

figure.set_size_inches((12, 8))

output_path = Path.joinpath(CACHE_DIR, f"historical_hierarchical.png")

figure.savefig(output_path)


"""

with open(CACHE_DIR.joinpath(f"hyperparameters_tfidf_hierarchical.json")) as file:
    result_tfidf = json.load(file)

with open(CACHE_DIR.joinpath(f"hyperparameters_bert_hierarchical.json")) as file:
    result_bert = json.load(file)

figure = plot_tfidf_versus_bert_hf(result_tfidf, result_bert)

# figure.set_size_inches((12, 8))

# output_path = Path.joinpath(CACHE_DIR, f"historical.png")

# figure.savefig(output_path)
"""
