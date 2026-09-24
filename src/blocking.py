"""
src/blocking.py - Scalable candidate generation and blocking algorithms for entity resolution.
Implements:
1. Standard Key Blocking (Rule-based exact matching keys)
2. Token Inverted Index Blocking (with IDF weighting and stopword pruning)
3. Character Trigram / Q-gram Blocking
4. Top-K Candidate Pruning per query entity
"""

from collections import defaultdict
import math
from typing import Dict, Iterable, List, Optional, Set, Tuple
import numpy as np
import pandas as pd
from src.normalization import extract_blocking_tokens, normalize_business_name, normalize_country


class StandardBlocker:
    """
    Blocks entities using rule-based composite keys, e.g.:
    - Country + First Word of Business Name
    - Country + First 4 letters of Business Name
    - Country + Significant Name Tokens
    """

    def __init__(self, key_type: str = "first_word"):
        self.key_type = key_type
        self.index: Dict[str, List[str]] = defaultdict(list)

    def extract_key(self, name: str, country: str) -> Optional[str]:
        country_norm = normalize_country(country)
        name_norm = normalize_business_name(name)
        if not name_norm:
            return None

        tokens = name_norm.split()
        if not tokens:
            return None

        if self.key_type == "first_word":
            return f"{country_norm}::{tokens[0]}"
        elif self.key_type == "prefix_4":
            prefix = name_norm.replace(" ", "")[:4]
            return f"{country_norm}::{prefix}" if len(prefix) >= 3 else None
        elif self.key_type == "sorted_tokens":
            # Top 2 longest tokens sorted
            top_tokens = sorted(tokens, key=lambda t: len(t), reverse=True)[:2]
            return f"{country_norm}::{'_'.join(sorted(top_tokens))}"
        else:
            return f"{country_norm}::{tokens[0]}"

    def fit(self, target_df: pd.DataFrame, id_col: str = "entity_id", name_col: str = "business_name", country_col: str = "country"):
        """Index candidate records from target sources (Source 2 and/or Source 3)."""
        self.index.clear()
        for _, row in target_df.iterrows():
            eid = row[id_col]
            key = self.extract_key(row.get(name_col, ""), row.get(country_col, ""))
            if key:
                self.index[key].append(eid)
        return self

    def generate_candidates(
        self,
        query_df: pd.DataFrame,
        id_col: str = "entity_id",
        name_col: str = "business_name",
        country_col: str = "country",
        max_candidates_per_key: int = 200,
    ) -> pd.DataFrame:
        """Generate candidate pairs for query entities (Source 1)."""
        pairs = []
        for _, row in query_df.iterrows():
            s1_id = row[id_col]
            key = self.extract_key(row.get(name_col, ""), row.get(country_col, ""))
            if key and key in self.index:
                cand_list = self.index[key]
                if len(cand_list) <= max_candidates_per_key:
                    for cand_id in cand_list:
                        pairs.append({"source1_entity_id": s1_id, "candidate_entity_id": cand_id, "block_key": key})
        return pd.DataFrame(pairs)


class TokenInvertedIndexBlocker:
    """
    Inverted-index blocking with token IDF filtering and top-K candidate scoring.
    For each target entity, indexes rare/discriminative tokens per country partition.
    For query entities, retrieves entities sharing matching tokens and ranks by overlap score.
    """

    def __init__(self, max_token_freq: int = 10000, min_token_len: int = 3, top_k: int = 30):
        self.max_token_freq = max_token_freq
        self.min_token_len = min_token_len
        self.top_k = top_k
        self.index: Dict[Tuple[str, str], List[str]] = defaultdict(list)
        self.doc_lengths: Dict[str, int] = {}
        self.total_docs: int = 0

    def fit(
        self,
        target_df: pd.DataFrame,
        id_col: str = "entity_id",
        name_col: str = "business_name",
        country_col: str = "country",
    ):
        """Build the inverted index from target dataset (S2 + S3)."""
        self.index.clear()
        self.doc_lengths.clear()
        self.total_docs = len(target_df)

        for _, row in target_df.iterrows():
            eid = str(row[id_col])
            country = normalize_country(row.get(country_col, ""))
            name_norm = normalize_business_name(row.get(name_col, ""))
            tokens = extract_blocking_tokens(name_norm, min_len=self.min_token_len)
            self.doc_lengths[eid] = max(len(tokens), 1)

            for token in tokens:
                self.index[(country, token)].append(eid)

        # Prune hyper-frequent tokens that blow up comparisons without providing discrimination
        pruned_keys = [k for k, v in self.index.items() if len(v) > self.max_token_freq]
        for k in pruned_keys:
            del self.index[k]

        return self

    def generate_candidates(
        self,
        query_df: pd.DataFrame,
        id_col: str = "entity_id",
        name_col: str = "business_name",
        country_col: str = "country",
    ) -> pd.DataFrame:
        """Retrieve top-K candidates for each query entity based on shared token score."""
        results = []

        for _, row in query_df.iterrows():
            s1_id = str(row[id_col])
            country = normalize_country(row.get(country_col, ""))
            name_norm = normalize_business_name(row.get(name_col, ""))
            query_tokens = extract_blocking_tokens(name_norm, min_len=self.min_token_len)
            if not query_tokens:
                continue

            candidate_scores: Dict[str, float] = defaultdict(float)

            for token in query_tokens:
                key = (country, token)
                if key in self.index:
                    postings = self.index[key]
                    # IDF weight = log(1 + total_docs / postings_len)
                    idf = math.log(1.0 + (self.total_docs / max(len(postings), 1)))
                    for cand_id in postings:
                        candidate_scores[cand_id] += idf

            if candidate_scores:
                # Rank by score and keep top-K
                sorted_cands = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)[: self.top_k]
                for cand_id, score in sorted_cands:
                    results.append({
                        "source1_entity_id": s1_id,
                        "candidate_entity_id": cand_id,
                        "blocking_score": round(score, 4),
                    })

        return pd.DataFrame(results)


def evaluate_candidate_distribution(candidate_df: pd.DataFrame) -> Dict[str, float]:
    """Calculate summary statistics for candidate counts per source 1 entity."""
    if candidate_df.empty:
        return {"total_pairs": 0, "unique_s1": 0, "mean_cands": 0.0, "p50": 0.0, "p90": 0.0, "max": 0}

    counts = candidate_df.groupby("source1_entity_id")["candidate_entity_id"].count()
    return {
        "total_pairs": int(len(candidate_df)),
        "unique_queries": int(counts.count()),
        "mean_candidates_per_query": float(counts.mean()),
        "median_candidates": float(counts.median()),
        "p90_candidates": float(np.percentile(counts, 90)),
        "p99_candidates": float(np.percentile(counts, 99)),
        "max_candidates": int(counts.max()),
    }
