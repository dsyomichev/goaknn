import functools


def init_expand(relations):

    @functools.cache
    def relatives(child):
        relation = relations.get(int(child), {})

        parents = set()

        for parent, predicate in relation.items():
            if not predicate == "is a" and not predicate == "part of":
                continue

            parents.add(int(parent))

        return parents

    @functools.cache
    def trace(child):
        lineage = {int(child)}
        parents = relatives(child)

        while len(parents) > 0:
            lineage |= parents
            parents = functools.reduce(lambda acc, parent: acc.union(relatives(parent)), parents, set()) - lineage

        return lineage

    def expand(y):
        return list(map(lambda yi: list(map(lambda yii: trace(yii), yi)), y))

    return expand


def macro_average_recall(y_true, y_pred):
    result = {}

    for labels_true, labels_pred in zip(y_true, y_pred):
        labels_true_set = set(labels_true)
        labels_pred_set = set(labels_pred)

        for label in labels_true_set:
            vals = result.setdefault(label, [0, 0])

            if label in labels_pred_set:
                vals[0] += 1

            vals[1] += 1

    mar = 0
    num_class = len(result)

    for _, (true_pos, num_inst) in result.items():
        mar += 0 if not num_inst or not num_class else true_pos / num_inst / num_class

    return mar


def mean_reciprocal_rank(y_true, y_pred):
    reciprocal_rank_agg = 0
    num_samples = len(y_true)

    for labels_true, labels_pred in zip(y_true, y_pred):
        labels_true_set = set(labels_true)

        for j, label in enumerate(labels_pred):
            if label in labels_true_set:
                reciprocal_rank_agg += 1 / (j + 1)
                break

    return 0 if not num_samples else reciprocal_rank_agg / len(y_true)


def mrr_mar(y_true, y_pred):

    mrr = mean_reciprocal_rank(y_true, y_pred)
    mar = macro_average_recall(y_true, y_pred)

    return mrr, mar


def hp_hr_hf(y_true, y_pred, relations):
    expand = init_expand(relations)

    n = len(y_true)

    hp_ag = 0
    hr_ag = 0
    hf_ag = 0

    for y_true_expanded, y_pred_expanded in zip(expand(y_true), expand(y_pred)):
        expansion_precisions = []
        expansion_recalls = []

        for y_pred_lineage in y_pred_expanded:
            pool = []

            for y_true_lineage in y_true_expanded:
                intersect = y_pred_lineage & y_true_lineage

                pool.append(len(intersect) / len(y_pred_lineage))

            expansion_precisions.append(max(pool))

        for y_true_lineage in y_true_expanded:
            pool = []

            for y_pred_lineage in y_pred_expanded:
                intersect = y_pred_lineage & y_true_lineage

                pool.append(len(intersect) / len(y_true_lineage))

            expansion_recalls.append(max(pool))

        hpi = 0 if not expansion_precisions else sum(expansion_precisions) / len(y_pred_expanded)
        hri = 0 if not expansion_recalls else sum(expansion_recalls) / len(y_true_expanded)
        hfi = 0 if not hpi and not hri else 2 * (hpi * hri) / (hpi + hri)

        hp_ag += hpi
        hr_ag += hri
        hf_ag += hfi

    hp = 0 if not n else hp_ag / n
    hr = 0 if not n else hr_ag / n
    hf = 0 if not n else hf_ag / n

    return hp, hr, hf
