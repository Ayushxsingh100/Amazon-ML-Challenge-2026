"""
Amazon ML Challenge 2026 - Entity Resolution and Record Linkage Package
"""

from src.loading import (
    load_source,
    load_ground_truth,
    parse_ground_truth_map,
    parse_ground_truth_pairs,
    get_dataset_summary,
)
from src.normalization import (
    normalize_business_name,
    normalize_address,
    normalize_country,
    normalize_dataframe,
    extract_blocking_tokens,
)
from src.blocking import (
    StandardBlocker,
    TokenInvertedIndexBlocker,
    evaluate_candidate_distribution,
)
from src.evaluation import (
    compute_blocking_metrics,
    analyze_missed_pairs,
    sample_candidate_pairs_with_labels,
)
from src.features import (
    compute_pair_features,
    extract_features_for_candidates,
    token_jaccard_similarity,
    char_ngram_jaccard,
)

__all__ = [
    "load_source",
    "load_ground_truth",
    "parse_ground_truth_map",
    "parse_ground_truth_pairs",
    "get_dataset_summary",
    "normalize_business_name",
    "normalize_address",
    "normalize_country",
    "normalize_dataframe",
    "extract_blocking_tokens",
    "StandardBlocker",
    "TokenInvertedIndexBlocker",
    "evaluate_candidate_distribution",
    "compute_blocking_metrics",
    "analyze_missed_pairs",
    "sample_candidate_pairs_with_labels",
    "compute_pair_features",
    "extract_features_for_candidates",
    "token_jaccard_similarity",
    "char_ngram_jaccard",
]
