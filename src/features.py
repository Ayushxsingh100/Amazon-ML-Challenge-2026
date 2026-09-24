"""
src/features.py - Feature engineering for entity pair comparison.
Computes string similarity metrics, token overlap, street number concordance,
and character n-gram similarities between candidate entity pairs.
"""

from difflib import SequenceMatcher
import re
from typing import Dict, List, Optional, Set
import numpy as np
import pandas as pd
from src.normalization import normalize_address, normalize_business_name


NUMBER_REGEX = re.compile(r"\b\d+\b")


def extract_numbers(text: str) -> Set[str]:
    """Extract all numerical tokens (e.g. house number, pin code, suite number)."""
    if not text:
        return set()
    return set(NUMBER_REGEX.findall(str(text)))


def token_jaccard_similarity(tokens1: Set[str], tokens2: Set[str]) -> float:
    """Compute Jaccard similarity: |A ∩ B| / |A ∪ B|."""
    if not tokens1 or not tokens2:
        return 0.0
    union = len(tokens1 | tokens2)
    if union == 0:
        return 0.0
    return len(tokens1 & tokens2) / union


def token_overlap_coefficient(tokens1: Set[str], tokens2: Set[str]) -> float:
    """Compute Overlap coefficient: |A ∩ B| / min(|A|, |B|). Robust to sub-phrases."""
    if not tokens1 or not tokens2:
        return 0.0
    min_len = min(len(tokens1), len(tokens2))
    if min_len == 0:
        return 0.0
    return len(tokens1 & tokens2) / min_len


def char_ngram_jaccard(str1: str, str2: str, n: int = 3) -> float:
    """Compute character n-gram Jaccard similarity."""
    if not str1 or not str2:
        return 0.0
    ngrams1 = {str1[i : i + n] for i in range(len(str1) - n + 1)} if len(str1) >= n else {str1}
    ngrams2 = {str2[i : i + n] for i in range(len(str2) - n + 1)} if len(str2) >= n else {str2}
    return token_jaccard_similarity(ngrams1, ngrams2)


def string_sequence_similarity(str1: str, str2: str) -> float:
    """Normalized Ratcliff-Obershelp gesture similarity (difflib SequenceMatcher)."""
    if not str1 or not str2:
        return 0.0
    return SequenceMatcher(None, str1, str2).ratio()


def compute_pair_features(
    s1_row: Dict[str, str],
    target_row: Dict[str, str],
) -> Dict[str, float]:
    """
    Compute full feature vector for an entity pair:
    - name_exact_match, name_jaccard, name_overlap, name_sequence_ratio, name_trigram_sim
    - address_exact_match, address_jaccard, address_number_match, address_sequence_ratio
    - country_match, name_len_diff
    """
    s1_name_raw = str(s1_row.get("business_name", ""))
    s1_addr_raw = str(s1_row.get("business_address", ""))
    tgt_name_raw = str(target_row.get("business_name", ""))
    tgt_addr_raw = str(target_row.get("business_address", ""))

    s1_name = normalize_business_name(s1_name_raw)
    tgt_name = normalize_business_name(tgt_name_raw)
    s1_addr = normalize_address(s1_addr_raw)
    tgt_addr = normalize_address(tgt_addr_raw)

    s1_name_tokens = set(s1_name.split())
    tgt_name_tokens = set(tgt_name.split())
    s1_addr_tokens = set(s1_addr.split())
    tgt_addr_tokens = set(tgt_addr.split())

    # Numbers in addresses (crucial street/suite number matching)
    s1_nums = extract_numbers(s1_addr_raw)
    tgt_nums = extract_numbers(tgt_addr_raw)
    num_match = 1.0 if (s1_nums and tgt_nums and (s1_nums & tgt_nums)) else 0.0

    c1 = str(s1_row.get("country", "")).strip().lower()
    c2 = str(target_row.get("country", "")).strip().lower()
    country_match = 1.0 if c1 and c2 and c1 == c2 else 0.0

    return {
        "name_exact_match": float(s1_name == tgt_name and bool(s1_name)),
        "name_jaccard": round(token_jaccard_similarity(s1_name_tokens, tgt_name_tokens), 4),
        "name_overlap_coef": round(token_overlap_coefficient(s1_name_tokens, tgt_name_tokens), 4),
        "name_trigram_jaccard": round(char_ngram_jaccard(s1_name, tgt_name, n=3), 4),
        "name_seq_ratio": round(string_sequence_similarity(s1_name, tgt_name), 4),
        "name_len_diff": float(abs(len(s1_name) - len(tgt_name))),
        "address_exact_match": float(s1_addr == tgt_addr and bool(s1_addr)),
        "address_jaccard": round(token_jaccard_similarity(s1_addr_tokens, tgt_addr_tokens), 4),
        "address_number_match": num_match,
        "address_seq_ratio": round(string_sequence_similarity(s1_addr, tgt_addr), 4),
        "country_match": country_match,
    }


def extract_features_for_candidates(
    candidate_pairs_df: pd.DataFrame,
    s1_lookup: Dict[str, Dict[str, str]],
    target_lookup: Dict[str, Dict[str, str]],
) -> pd.DataFrame:
    """Compute feature vectors for a DataFrame of candidate pairs."""
    feature_rows = []
    for _, row in candidate_pairs_df.iterrows():
        s1_id = row["source1_entity_id"]
        cand_id = row["candidate_entity_id"]
        s1_info = s1_lookup.get(s1_id, {})
        tgt_info = target_lookup.get(cand_id, {})
        feats = compute_pair_features(s1_info, tgt_info)
        feats["source1_entity_id"] = s1_id
        feats["candidate_entity_id"] = cand_id
        feature_rows.append(feats)
    return pd.DataFrame(feature_rows)
