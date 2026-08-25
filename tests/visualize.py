import json
import numpy

from matplotlib import pyplot
from pathlib import Path

CACHE_DIR = Path(__file__, "../../data/cache/backup").resolve()
FILE_NAME = "tfidf_exact.json"

with open(Path.joinpath(CACHE_DIR, FILE_NAME), "r") as file:
    source = json.load(file)

n_labels = numpy.array([group["classifier__n_labels"] for group in source["params"]])
n_labels_unique = numpy.unique(n_labels)

n_neighbors = numpy.array([group["classifier__n_neighbors"] for group in source["params"]])
e_amplification = numpy.array([group["classifier__e_amplification"] for group in source["params"]])

label_map = [("MRR", "mean_test_mrr"), ("MAR{}", "mean_test_mar"), ("F", "mean_test_f")]


fig, axes = pyplot.subplots(3, 3)


for i in range(3):
    indices = numpy.argwhere(n_labels == n_labels_unique[i]).flat

    x = n_neighbors[indices]
    y = e_amplification[indices]

    X, Y = numpy.meshgrid(numpy.unique(x), numpy.unique(y))

    for j in range(3):
        axis = axes[i, j]

        label, key = label_map[j]

        label = label.format(n_labels_unique[i])

        z = numpy.array(source[key])[indices]

        Z = numpy.reshape(z, (3, 4))

        axis.set_xticks(numpy.unique(x))
        axis.set_yticks(numpy.unique(y))

        axis.set_title(label)

        if i == 2 and j == 1:
            axis.set_xlabel("Number of Neighbors")

        if j == 0 and i == 1:
            axis.set_ylabel("Amplification Exponent")

        axis.pcolormesh(X, Y, Z, cmap="grey")

pyplot.show()

"""


        axis.imshow(data)

print(pyplot.colormaps)

for n_label in numpy.unique(n_labels):
    inds = numpy.argwhere(n_labels == n_label).flat

    x = n_neighbors[inds]
    y = e_amplification[inds]

    z = mean_test_f[inds]

    ax = fig.add_subplot(projection="3d")

    ax.plot_trisurf(x, y, z, cmap="grey", antialiased=True)

    break

pyplot.show()
"""
