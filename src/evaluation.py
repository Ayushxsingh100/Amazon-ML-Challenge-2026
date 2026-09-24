"""
src/evaluation.py - Evaluation metrics for candidate generation and blocking algorithms.
Calculates Pair Completeness (Candidate Recall), Reduction Ratio, Pairs Quality,
Candidate Count Distributions, and produces detailed evaluation reports.
"""

from typing import Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd


def compute_blocking_metrics(
    candidate_pairs_df: pd.DataFrame,
    ground_truth_map: Dict[str, List[str]],
    total_query_count: int,
    total_target_count: int,
) -> Dict[str, float]:
    """
    Compute standard record linkage blocking metrics:
    - Pair Completeness (Recall): |Candidates ∩ True Matches| / |True Matches|
    - Pairs Quality (Precision): |Candidates ∩ True Matches| / |Candidates|
    - Reduction Ratio (RR): 1 - |Candidates| / (|Queries| * |Targets|)
    - Query Coverage: Proportion of queries with at least one true match found
    - Strict Query Coverage: Proportion of queries with ALL true matches found
    """
    total_possible_pairs = float(total_query_count * total_target_count) if total_query_count and total_target_count else 1.0
    total_candidates = len(candidate_pairs_df)

    # Build set of candidate pairs for O(1) membership check: (s1_id, candidate_id)
    candidate_pairs_set = set(
        zip(candidate_pairs_df["source1_entity_id"], candidate_pairs_df["candidate_entity_id"])
    )

    total_true_matches = 0
    captured_true_matches = 0
    queries_with_ground_truth = 0
    queries_at_least_one_found = 0
    queries_all_found = 0

    for s1_id, true_matches in ground_truth_map.items():
        if not true_matches:
            continue
        queries_with_ground_truth += 1
        num_true = len(true_matches)
        total_true_matches += num_true

        found_count = 0
        for target_id in true_matches:
            if (s1_id, target_id) in candidate_pairs_set:
                captured_true_matches += 1
                found_count += 1

        if found_count > 0:
            queries_at_least_one_found += 1
        if found_count == num_true:
            queries_all_found += 1

    recall = captured_true_matches / total_true_matches if total_true_matches > 0 else 0.0
    precision = captured_true_matches / total_candidates if total_candidates > 0 else 0.0
    reduction_ratio = 1.0 - (total_candidates / total_possible_pairs) if total_possible_pairs > 0 else 0.0
    query_coverage = queries_at_least_one_found / queries_with_ground_truth if queries_with_ground_truth > 0 else 0.0
    strict_coverage = queries_all_found / queries_with_ground_truth if queries_with_ground_truth > 0 else 0.0

    return {
        "pair_completeness_recall": round(recall, 6),
        "pairs_quality_precision": round(precision, 6),
        "reduction_ratio": round(reduction_ratio, 8),
        "query_partial_coverage": round(query_coverage, 4),
        "query_full_coverage": round(strict_coverage, 4),
        "total_true_matches": total_true_matches,
        "captured_true_matches": captured_true_matches,
        "missed_true_matches": total_true_matches - captured_true_matches,
        "total_generated_candidates": total_candidates,
    }


def analyze_missed_pairs(
    candidate_pairs_df: pd.DataFrame,
    ground_truth_map: Dict[str, List[str]],
    s1_lookup: Dict[str, Dict[str, str]],
    target_lookup: Dict[str, Dict[str, str]],
    sample_size: int = 20,
) -> pd.DataFrame:
    """
    Extract a sample of false negatives (true matches missed by blocking)
    with their raw names and addresses for root-cause error analysis.
    """
    candidate_pairs_set = set(
        zip(candidate_pairs_df["source1_entity_id"], candidate_pairs_df["candidate_entity_id"])
    )

    missed_records = []
    for s1_id, true_matches in ground_truth_map.items():
        s1_info = s1_lookup.get(s1_id, {})
        for target_id in true_matches:
            if (s1_id, target_id) not in candidate_pairs_set:
                target_info = target_lookup.get(target_id, {})
                missed_records.append({
                    "source1_id": s1_id,
                    "target_id": target_id,
                    "s1_name": s1_info.get("business_name", ""),
                    "target_name": target_info.get("business_name", ""),
                    "s1_address": s1_info.get("business_address", ""),
                    "target_address": target_info.get("business_address", ""),
                    "s1_country": s1_info.get("country", ""),
                    "target_country": target_info.get("country", ""),
                })
                if len(missed_records) >= sample_size:
                    break
        if len(missed_records) >= sample_size:
            break

    return pd.DataFrame(missed_records)


def sample_candidate_pairs_with_labels(
    candidate_pairs_df: pd.DataFrame,
    ground_truth_map: Dict[str, List[str]],
    s1_lookup: Dict[str, Dict[str, str]],
    target_lookup: Dict[str, Dict[str, str]],
    n_positives: int = 10,
    n_negatives: int = 10,
) -> pd.DataFrame:
    """
    Produce a balanced inspection sample of candidate pairs annotated with ground truth label.
    """
    candidate_records = []
    pos_count = 0
    neg_count = 0

    for _, row in candidate_pairs_df.iterrows():
        s1_id = row["source1_entity_id"]
        cand_id = row["candidate_entity_id"]
        true_list = ground_truth_map.get(s1_id, [])
        is_true_match = cand_id in true_list

        if is_true_match and pos_count < n_positives:
            pos_count += 1
            record = _format_pair_record(s1_id, cand_id, True, row, s1_lookup, target_lookup)
            candidate_records.append(record)
        elif not is_true_match and neg_count < n_negatives:
            neg_count += 1
            record = _format_pair_record(s1_id, cand_id, False, row, s1_lookup, target_lookup)
            candidate_records.append(record)

        if pos_count >= n_positives and neg_count >= n_negatives:
            break

    return pd.DataFrame(candidate_records)


def _format_pair_record(
    s1_id: str,
    cand_id: str,
    is_match: bool,
    cand_row: pd.Series,
    s1_lookup: Dict[str, Dict[str, str]],
    target_lookup: Dict[str, Dict[str, str]],
) -> Dict[str, object]:
    s1_info = s1_lookup.get(s1_id, {})
    target_info = target_lookup.get(cand_id, {})
    return {
        "source1_id": s1_id,
        "candidate_id": cand_id,
        "ground_truth_label": int(is_match),
        "blocking_score": cand_row.get("blocking_score", np.nan),
        "s1_name": s1_info.get("business_name", ""),
        "candidate_name": target_info.get("business_name", ""),
        "s1_address": s1_info.get("business_address", ""),
        "candidate_address": target_info.get("business_address", ""),
        "country": s1_info.get("country", ""),
    }
