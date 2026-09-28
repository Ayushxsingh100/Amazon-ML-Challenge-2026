from __future__ import annotations

from typing import Dict, Set, Iterable, Any

from validation.metrics import evaluate_entities


SCORER_VERSION = "scorer_v1"
METRIC_NAME = "macro_f0.5"


def score_predictions(
    predictions: Dict[str, Set[str]],
    ground_truth: Dict[str, Set[str]],
    entity_ids: Iterable[str] | None = None,
) -> Dict[str, Any]:
    """
    Canonical E02 scorer.

    Scoring unit:
        One Source-1 entity.

    For each S1 entity, the predicted and ground-truth S2/S3 IDs
    are treated as sets. F0.5 is calculated per entity and then
    macro-averaged.

    S2 and S3 are therefore scored jointly per S1.
    """

    return evaluate_entities(
        predictions=predictions,
        ground_truth=ground_truth,
        entity_ids=entity_ids,
    )
