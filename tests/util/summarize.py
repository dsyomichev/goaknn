import numpy
import json

from pathlib import Path
from json import encoder

CACHE_DIR = Path(__file__, "../../../data/cache").resolve()

results = {}

with open(CACHE_DIR.joinpath("hyperparameters_bert_exact.json")) as file:
    results["bert_e"] = json.load(file)

with open(CACHE_DIR.joinpath("hyperparameters_bert_hierarchical.json")) as file:
    results["bert_h"] = json.load(file)

with open(CACHE_DIR.joinpath("hyperparameters_tfidf_exact.json")) as file:
    results["tfidf_e"] = json.load(file)

with open(CACHE_DIR.joinpath("hyperparameters_tfidf_hierarchical.json")) as file:
    results["tfidf_h"] = json.load(file)


def compare(method):
    n_labels = 20

    exact = results[f"{method}_e"]
    hierarchical = results[f"{method}_h"]

    parameters = {}

    for group in exact["params"]:
        for key, value in group.items():
            values = parameters.setdefault(key, [])
            values.append(value)

    parameters = {key: numpy.array(value) for key, value in parameters.items()}

    fixed_n_labels_idxs = numpy.argwhere(parameters["classifier__n_labels"] == n_labels).flatten()

    parameters = {key: value[fixed_n_labels_idxs] for key, value in parameters.items()}

    parameters.pop("classifier__n_labels")

    best = {}

    for key in ["mean_test_mrr", "mean_test_mar"]:
        mean = numpy.array(exact[key])[fixed_n_labels_idxs]
        best[key] = (numpy.argmax(mean), mean)

    for key in ["mean_test_hp", "mean_test_hr", "mean_test_hf"]:
        mean = numpy.array(hierarchical[key])[fixed_n_labels_idxs]
        best[key] = (numpy.argmax(mean), mean)

    result = {}

    for key, (idx, _) in best.items():
        metric = {}

        for parameter, values in parameters.items():
            metric[parameter] = int(values[idx])

        for m_key in ["mean_test_mrr", "mean_test_mar"]:
            values = best[m_key][1]
            metric[m_key] = format(float(values[idx]), ".4f")

        for m_key in ["mean_test_hp", "mean_test_hr", "mean_test_hf"]:
            values = best[m_key][1]
            metric[m_key] = format(float(values[idx]), ".4f")

        result[key] = metric

    return result


result = compare("tfidf")

with open(CACHE_DIR.joinpath("best_param_score_combos_tfidf.json"), "w") as file:
    json.dump(result, file, indent=4)

result = compare("bert")

with open(CACHE_DIR.joinpath("best_param_score_combos_bert.json"), "w") as file:
    json.dump(result, file, indent=4)
