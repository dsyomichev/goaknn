def score_mrr(y_true, y_pred):
    reciprocal_rank_agg = 0
    num_samples = len(y_true)

    for labels_true, labels_pred in zip(y_true, y_pred):
        labels_true_set = set(labels_true)

        for j, label in enumerate(labels_pred):
            if label in labels_true_set:
                reciprocal_rank_agg += 1 / (j + 1)
                break

    return 0 if not num_samples else reciprocal_rank_agg / len(y_true)


def score_mar(y_true, y_pred):
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
