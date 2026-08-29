import numpy
import json
import functools

from matplotlib import pyplot
from matplotlib.patches import Patch
from pathlib import Path
from sklearn.preprocessing import StandardScaler

from src.db.DatabaseClient import DatabaseClient
from src.db.execute import execute
from tests.historical import train_test_split_temporal, SPLIT_TEMPORAL_START, SPLIT_TEMPORAL_STEP

CACHE_DIR = Path(__file__, "../../../data/cache").resolve()

PARAMETER_KEY_MAP = {
    "classifier__n_neighbors": "K",
    "classifier__e_amplification": "Amplification",
    "classifier__n_labels": "Limit",
}

SELECT_TERM_ID_COUNT = """
    SELECT COUNT(term_id) FROM annotations
    GROUP BY term_id
    ORDER BY COUNT(term_id) DESC
"""


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
    ymin, ymax = 0.1, 1

    metric_keys = ["mrr", "mar"] if metric == "exact" else ["hp", "hr"]
    metric_labels = ["MRR", "R20"] if metric == "exact" else ["HP", "HR20"]

    for i, (axis, key) in enumerate(zip(axes, metric_keys)):
        y_tfidf = numpy.array([value[key] for value in result_tfidf.values()])
        y_bert = numpy.array([value[key] for value in result_bert.values()])

        axis.set_xticks(x)

        axis.set_ylabel(metric_labels[0] if key == metric_keys[0] else metric_labels[1])
        axis.set_xlabel("Year" if i else "")
        axis.set_xlim(xmin, xmax)
        axis.set_ylim(ymin, ymax)

        axis.plot(x, y_tfidf, label="TF-IDF" if not i else "", color="black", linestyle="solid")
        axis.plot(x, y_bert, label="BERT" if not i else "", color="black", linestyle="dashed")

        axis.hlines(numpy.linspace(ymin, ymax, 10), xmin, xmax, color="gray", linestyle="dotted")

    figure.legend(loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.025))
    figure.suptitle(f"Psuedo-historical Comparison ({"MRR/R20" if metric == "exact" else "HP/HR20"})")

    return figure


def plot_tfidf_versus_bert_hf(result_tfidf, result_bert, metric):
    figure, axes = pyplot.subplots(1, 3)

    parameters = {}

    for group in result_tfidf["params"]:
        for key, value in group.items():
            values = parameters.setdefault(key, [])
            values.append(value)

    parameters = {key: numpy.array(value) for key, value in parameters.items()}
    options = {key: numpy.unique(value) for key, value in parameters.items()}

    for i, (axis, n_labels) in enumerate(zip(axes, options["classifier__n_labels"])):
        fixed_n_labels_idxs = numpy.argwhere(parameters["classifier__n_labels"] == n_labels).flatten()

        x = parameters["classifier__n_neighbors"][fixed_n_labels_idxs]
        y = parameters["classifier__e_amplification"][fixed_n_labels_idxs]

        X, Y = numpy.meshgrid(numpy.unique(x), numpy.unique(y))

        z_tfidf = numpy.array(result_tfidf[metric])[fixed_n_labels_idxs]
        z_bert = numpy.array(result_bert[metric])[fixed_n_labels_idxs]
        z = z_tfidf < z_bert

        Z = z.reshape(X.shape)

        if i == 0:
            axis.set_ylabel(PARAMETER_KEY_MAP["classifier__e_amplification"])
        elif i == 1:
            axis.set_xlabel(PARAMETER_KEY_MAP["classifier__n_neighbors"])

        label = PARAMETER_KEY_MAP["classifier__n_labels"]

        axis.set_title(f"{label}={n_labels}", size=10)

        axis.set_xticks(options["classifier__n_neighbors"])
        axis.set_yticks(options["classifier__e_amplification"])

        xmin, xmax = numpy.min(x) - numpy.min(x) / 2, numpy.max(x) + numpy.max(x) / 4
        ymin, ymax = numpy.min(y) - 1, numpy.max(y) + 1

        vline_xval = numpy.array([value * 1.5 for value in options["classifier__n_neighbors"]])[:-1]
        hline_yval = numpy.array([value + 1 for value in options["classifier__e_amplification"]])[:-1]

        axis.vlines(vline_xval, ymin, ymax, color="gray", linestyle="dotted")
        axis.hlines(hline_yval, xmin, xmax, color="gray", linestyle="dotted")

        axis.pcolormesh(X, Y, Z, vmin=0, vmax=1, cmap="gray")

    patches = [Patch(color="black", label=" TF-IDF > BERT")]

    figure.legend(handles=patches, loc="lower center", bbox_to_anchor=(0.5, -0.1))

    figure.suptitle("TF-IDF vs. BERT Binary Comparison (HF)")

    return figure


def plt_annotation_term_count(connection):
    figure, axis = pyplot.subplots()

    y = connection.execute(SELECT_TERM_ID_COUNT).fetchall()
    x = numpy.linspace(0, len(y) - 1, len(y))

    axis.set_xlabel("Term Index")
    axis.set_ylabel("Appearances in Annotations")

    axis.plot(x, y, color="black")

    return figure


def plt_train_test_split_temporal_size(connection):
    figure, axis = pyplot.subplots()

    client = DatabaseClient(connection)

    train_test_split_generator = train_test_split_temporal(client, start=SPLIT_TEMPORAL_START, step=SPLIT_TEMPORAL_STEP)

    x = []
    ys_train = []
    ys_test = []

    for i, (_, _, y_train, y_test) in enumerate(train_test_split_generator):
        year = SPLIT_TEMPORAL_START + i

        x.append(year)
        ys_train.append(len(y_train))
        ys_test.append(len(y_test))

    x = numpy.array(x)
    ys_train = numpy.array(ys_train)
    ys_test = numpy.array(ys_test)

    axis.set_xlabel("Year")
    axis.set_ylabel("Size")

    axis.plot(x, ys_train, color="black", linestyle="solid", label="Train")
    axis.plot(x, ys_test, color="black", linestyle="dashed", label="Test")

    figure.legend(loc="lower center", bbox_to_anchor=(0.5, -0.1))

    return figure


def plt_embedding_d1_d2(connection):
    figure, axis = pyplot.subplots()

    client = DatabaseClient(connection)

    embeddings, _ = client.get_embeddings()

    x = [embedding[0] for embedding in embeddings]
    y = [embedding[1] for embedding in embeddings]

    axis.set_xlabel("First Dimension")
    axis.set_ylabel("Second Dimension")

    lim = 0.5

    axis.set_xlim(-lim, lim)
    axis.set_ylim(-lim, lim)

    axis.vlines(0, -lim, lim, color="gray", linestyle="dotted")
    axis.hlines(0, -lim, lim, color="gray", linestyle="dotted")

    axis.scatter(x, y, color="black", marker=".")

    return figure


def plt_embedding_d1_d2_mean_center(connection):
    figure, axis = pyplot.subplots()

    client = DatabaseClient(connection)

    embeddings, _ = client.get_embeddings()

    embeddings = StandardScaler(with_std=False).fit_transform(embeddings)

    x = [embedding[0] for embedding in embeddings]
    y = [embedding[1] for embedding in embeddings]

    axis.set_xlabel("First Dimension")
    axis.set_ylabel("Second Dimension")

    lim = 0.5

    axis.set_xlim(-lim, lim)
    axis.set_ylim(-lim, lim)

    axis.vlines(0, -lim, lim, color="gray", linestyle="dotted")
    axis.hlines(0, -lim, lim, color="gray", linestyle="dotted")

    axis.scatter(x, y, color="black", marker=".")

    return figure


"""

## Hyperparameter Comparison

for method in ["tfidf", "bert"]:
    with open(CACHE_DIR.joinpath(f"hyperparameters_{method}_hierarchical.json")) as file:
        result = json.load(file)

    figure = plot_hyperparameter_comparison(result)

    figure.set_size_inches((12, 6))

    figure.suptitle(f"{"BERT" if method == "bert" else "TF-IDF"} Hyperparameters")

    output_path = Path.joinpath(CACHE_DIR, f"hyperparameters_{method}.png")

    figure.savefig(output_path)

## Historical Comparison (Exact)

with open(CACHE_DIR.joinpath(f"historical_tfidf_exact.json")) as file:
    result_tfidf = json.load(file)

with open(CACHE_DIR.joinpath(f"historical_bert_exact.json")) as file:
    result_bert = json.load(file)

figure = plot_historical_comparison(result_tfidf, result_bert, metric="exact")

figure.set_size_inches((12, 6))

output_path = Path.joinpath(CACHE_DIR, f"historical_exact.png")

figure.savefig(output_path, bbox_inches="tight")

## Historical Comparison (Hierachical)

with open(CACHE_DIR.joinpath(f"historical_tfidf_hierarchical.json")) as file:
    result_tfidf = json.load(file)

with open(CACHE_DIR.joinpath(f"historical_bert_hierarchical.json")) as file:
    result_bert = json.load(file)

figure = plot_historical_comparison(result_tfidf, result_bert, metric="hierarchical")

figure.set_size_inches((12, 6))

output_path = Path.joinpath(CACHE_DIR, f"historical_hierarchical.png")

figure.savefig(output_path, bbox_inches="tight")

## Versus (HF)

with open(CACHE_DIR.joinpath(f"hyperparameters_tfidf_hierarchical.json")) as file:
    result_tfidf = json.load(file)

with open(CACHE_DIR.joinpath(f"hyperparameters_bert_hierarchical.json")) as file:
    result_bert = json.load(file)

figure = plot_tfidf_versus_bert_hf(result_tfidf, result_bert, "mean_test_hf")

figure.set_size_inches((12, 4))

output_path = Path.joinpath(CACHE_DIR, f"tfidf_versus_bert_hf.png")

figure.savefig(output_path, bbox_inches="tight")

## Versus (MAR)

with open(CACHE_DIR.joinpath(f"hyperparameters_tfidf_exact.json")) as file:
    result_tfidf = json.load(file)

with open(CACHE_DIR.joinpath(f"hyperparameters_bert_exact.json")) as file:
    result_bert = json.load(file)

figure = plot_tfidf_versus_bert_hf(result_tfidf, result_bert, "mean_test_mar")

figure.set_size_inches((12, 4))

output_path = Path.joinpath(CACHE_DIR, f"tfidf_versus_bert_mar.png")

figure.savefig(output_path, bbox_inches="tight")

## Annotation Term Count

figure = execute(plt_annotation_term_count)

figure.set_size_inches(6, 6)

output_path = Path.joinpath(CACHE_DIR, f"annotation_term_count.png")

figure.savefig(output_path)

## Temporal Split Sizes

figure = execute(plt_train_test_split_temporal_size)

figure.set_size_inches((12, 6))

output_path = Path.joinpath(CACHE_DIR, f"historical_train_test_split_size.png")

figure.savefig(output_path, bbox_inches="tight")

"""

## Embedding Dim. 1 vs Dim. 2

figure = execute(plt_embedding_d1_d2)

figure.set_size_inches((6, 6))

output_path = Path.joinpath(CACHE_DIR, f"embedding_d1_d2.png")

figure.savefig(output_path, bbox_inches="tight")

## Scaled Embedding Dim. 1 vs Dim. 2

figure = execute(plt_embedding_d1_d2_mean_center)

figure.set_size_inches((6, 6))

output_path = Path.joinpath(CACHE_DIR, f"embeddings_d1_d2_scaled.png")

figure.savefig(output_path, bbox_inches="tight")
