"""
P3 Validation - Competition Metrics

Independent evaluation of model predictions against ground truth.

Prediction / ground-truth representation:

{
    "S1-123": {"S2-456", "S3-789"},
    ...
}

An empty set means that the S1 entity has no matched entity.
"""

from typing import Any, Dict, Iterable, Set


def fbeta_score(
    precision: float,
    recall: float,
    beta: float = 0.5,
) -> float:
    """
    Compute F-beta score.

    F_beta = (1 + beta^2) * P * R
             ---------------------
             beta^2 * P + R
    """

    denominator = (beta ** 2) * precision + recall

    if denominator == 0:
        return 0.0

    return (
        (1 + beta ** 2)
        * precision
        * recall
        / denominator
    )


def entity_precision_recall_f05(
    predicted: Set[str],
    actual: Set[str],
) -> Dict[str, float]:
    """
    Calculate precision, recall and F0.5 for one S1 entity.

    Special no-match handling:

    actual = empty AND predicted = empty
        -> correct no-match prediction
        -> precision = 1
        -> recall = 1
        -> F0.5 = 1

    actual = empty AND predicted != empty
        -> false positive
        -> precision = 0
        -> recall = 0
        -> F0.5 = 0
    """

    predicted = set(predicted)
    actual = set(actual)

    # -----------------------------------------------------
    # Correct no-match prediction
    # -----------------------------------------------------

    if not actual and not predicted:
        return {
            "true_positive": 0,
            "false_positive": 0,
            "false_negative": 0,
            "precision": 1.0,
            "recall": 1.0,
            "f0.5": 1.0,
        }

    # -----------------------------------------------------
    # False prediction for a true no-match entity
    # -----------------------------------------------------

    if not actual and predicted:
        return {
            "true_positive": 0,
            "false_positive": len(predicted),
            "false_negative": 0,
            "precision": 0.0,
            "recall": 0.0,
            "f0.5": 0.0,
        }

    # -----------------------------------------------------
    # Normal matched-entity case
    # -----------------------------------------------------

    true_positive = len(predicted & actual)
    false_positive = len(predicted - actual)
    false_negative = len(actual - predicted)

    precision = (
        true_positive / len(predicted)
        if predicted
        else 0.0
    )

    recall = (
        true_positive / len(actual)
        if actual
        else 0.0
    )

    f05 = fbeta_score(
        precision,
        recall,
        beta=0.5,
    )

    return {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "precision": precision,
        "recall": recall,
        "f0.5": f05,
    }


def evaluate_entities(
    predictions: Dict[str, Set[str]],
    ground_truth: Dict[str, Set[str]],
    entity_ids: Iterable[str] | None = None,
) -> Dict[str, Any]:
    """
    Evaluate predictions across S1 entities.

    Every entity supplied through entity_ids is evaluated,
    including entities whose ground truth is an empty set.

    If entity_ids is not supplied, the union of prediction
    and ground-truth IDs is evaluated.
    """

    if entity_ids is None:
        entity_ids = (
            set(predictions.keys())
            | set(ground_truth.keys())
        )
    else:
        entity_ids = set(entity_ids)

    per_entity = {}

    total_tp = 0
    total_fp = 0
    total_fn = 0

    f05_values = []

    correct_no_match = 0
    false_no_match_predictions = 0

    for s1_id in entity_ids:

        predicted = predictions.get(
            s1_id,
            set(),
        )

        actual = ground_truth.get(
            s1_id,
            set(),
        )

        result = entity_precision_recall_f05(
            predicted,
            actual,
        )

        per_entity[s1_id] = result

        total_tp += result["true_positive"]
        total_fp += result["false_positive"]
        total_fn += result["false_negative"]

        f05_values.append(
            result["f0.5"]
        )

        if not actual and not predicted:
            correct_no_match += 1

        elif not actual and predicted:
            false_no_match_predictions += 1

    macro_f05 = (
        sum(f05_values) / len(f05_values)
        if f05_values
        else 0.0
    )

    micro_precision = (
        total_tp / (total_tp + total_fp)
        if total_tp + total_fp > 0
        else 0.0
    )

    micro_recall = (
        total_tp / (total_tp + total_fn)
        if total_tp + total_fn > 0
        else 0.0
    )

    micro_f05 = fbeta_score(
        micro_precision,
        micro_recall,
        beta=0.5,
    )

    return {
        "entity_count": len(entity_ids),

        "macro_f0.5": macro_f05,

        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f0.5": micro_f05,

        "total_true_positive": total_tp,
        "total_false_positive": total_fp,
        "total_false_negative": total_fn,

        "correct_no_match": correct_no_match,
        "false_no_match_predictions": false_no_match_predictions,

        "per_entity": per_entity,
    }
