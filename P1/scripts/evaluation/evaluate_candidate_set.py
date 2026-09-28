#!/usr/bin/env python3
"""
P1/scripts/evaluation/evaluate_candidate_set.py

Canonical Candidate Set Evaluation Framework for Person 1 (P1).
Amazon ML Challenge 2026 - Entity Resolution.

Evaluates any candidate pair file (TSV or Parquet) against:
1. Canonical ground truth (train split) for exact recall, captured, missed, and true-candidate rates.
2. Canonical P1 entity datasets for schema validation, ID integrity, cross-country leakage,
   and candidate fanout per S1 entity (including zero-candidate S1 count and quantiles).
3. File provenance (SHA256, row counts, unique pairs, duplicate rows).

Usage:
  python3 evaluate_candidate_set.py --candidate-file P2/data/candidates/train_candidate_pairs_s2.tsv
  python3 evaluate_candidate_set.py --candidate-file P2/data/candidates/test_candidate_pairs_s2.tsv --split test
"""

import argparse
import hashlib
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import duckdb


def compute_sha256(filepath: str, block_size: int = 65536) -> str:
    """Compute streaming SHA256 checksum of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(block_size):
            h.update(chunk)
    return h.hexdigest()


def detect_file_source_and_split(filepath: str, explicit_source: Optional[str] = None, explicit_split: Optional[str] = None) -> tuple[str, str]:
    """Detect candidate target source (S2, S3, or COMBINED) and split (train or test)."""
    fname = os.path.basename(filepath).lower()
    
    # Split detection
    if explicit_split and explicit_split.lower() in ["train", "test"]:
        split = explicit_split.lower()
    elif "train" in fname:
        split = "train"
    elif "test" in fname:
        split = "test"
    else:
        split = "train"

    # Source detection
    if explicit_source and explicit_source.upper() in ["S2", "S3", "COMBINED"]:
        source = explicit_source.upper()
    elif "s2" in fname:
        source = "S2"
    elif "s3" in fname:
        source = "S3"
    else:
        source = "COMBINED"

    return source, split


def evaluate_candidate_set(
    candidate_file: str,
    candidate_source: Optional[str] = None,
    split: Optional[str] = None,
    ground_truth_file: Optional[str] = None,
    entities_dir: Optional[str] = None,
    entities_s1: Optional[str] = None,
    entities_target: Optional[str] = None,
    output_report: Optional[str] = None,
    output_format: str = "json",
    quiet: bool = False,
) -> Dict[str, Any]:
    """Execute complete evaluation of a candidate set."""
    t0 = time.time()
    
    # Resolve paths
    repo_root = Path(__file__).resolve().parent.parent.parent.parent
    candidate_path = Path(candidate_file).resolve()
    if not candidate_path.exists():
        raise FileNotFoundError(f"Candidate file not found: {candidate_file}")

    det_source, det_split = detect_file_source_and_split(str(candidate_path), candidate_source, split)

    # Default ground truth path
    if ground_truth_file is None:
        ground_truth_file = str(repo_root / "data" / "train" / "train_ground_truth.tsv")

    # Default entity paths
    if entities_dir is None:
        entities_base = repo_root / "P1" / "data" / "entities" / det_split
    else:
        entities_base = Path(entities_dir) / det_split

    if entities_s1 is None:
        s1_file = entities_base / "source1" / f"{det_split}_s1_entities.parquet"
    else:
        s1_file = Path(entities_s1)

    if not s1_file.exists():
        raise FileNotFoundError(f"S1 entity file not found: {s1_file}")

    target_files = {}
    if det_source in ["S2", "COMBINED"]:
        s2_file = entities_base / "source2" / f"{det_split}_s2_entities.parquet"
        if not s2_file.exists():
            raise FileNotFoundError(f"S2 entity file not found: {s2_file}")
        target_files["S2"] = str(s2_file)

    if det_source in ["S3", "COMBINED"]:
        s3_file = entities_base / "source3" / f"{det_split}_s3_entities.parquet"
        if not s3_file.exists():
            raise FileNotFoundError(f"S3 entity file not found: {s3_file}")
        target_files["S3"] = str(s3_file)

    file_size_bytes = candidate_path.stat().st_size
    file_sha256 = compute_sha256(str(candidate_path))

    if not quiet:
        print(f"=== Evaluating Candidate File: {candidate_path.name} ===", flush=True)
        print(f"Split: {det_split} | Target Source: {det_source} | Size: {file_size_bytes:,} bytes", flush=True)

    # Initialize DuckDB
    con = duckdb.connect()
    con.execute("SET threads=4")
    con.execute("SET memory_limit='8GB'")

    # Load S1 entities
    con.execute(f"""
        CREATE TEMP TABLE s1_entities AS
        SELECT entity_id, country_normalized
        FROM read_parquet('{s1_file}')
    """)
    total_s1_in_pool = con.execute("SELECT COUNT(*) FROM s1_entities").fetchone()[0]

    # Load Target entities
    if det_source == "S2":
        con.execute(f"""
            CREATE TEMP TABLE target_entities AS
            SELECT entity_id, country_normalized, 'S2' as src_type
            FROM read_parquet('{target_files["S2"]}')
        """)
    elif det_source == "S3":
        con.execute(f"""
            CREATE TEMP TABLE target_entities AS
            SELECT entity_id, country_normalized, 'S3' as src_type
            FROM read_parquet('{target_files["S3"]}')
        """)
    else:  # COMBINED
        con.execute(f"""
            CREATE TEMP TABLE target_entities AS
            SELECT entity_id, country_normalized, 'S2' as src_type
            FROM read_parquet('{target_files["S2"]}')
            UNION ALL
            SELECT entity_id, country_normalized, 'S3' as src_type
            FROM read_parquet('{target_files["S3"]}')
        """)
    total_target_in_pool = con.execute("SELECT COUNT(*) FROM target_entities").fetchone()[0]

    # Load candidates
    is_parquet = str(candidate_path).endswith(".parquet")
    if is_parquet:
        load_sql = f"SELECT * FROM read_parquet('{candidate_path}')"
    else:
        load_sql = f"SELECT * FROM read_csv('{candidate_path}', delim='\\t', header=true, all_varchar=true)"

    con.execute(f"CREATE TEMP TABLE raw_cands AS {load_sql}")

    # Inspect columns
    col_names = [c[0] for c in con.execute("DESCRIBE raw_cands").fetchall()]
    has_s1_col = "source1_entity_id" in col_names
    has_match_col = "matched_entity_id" in col_names

    if not (has_s1_col and has_match_col):
        raise ValueError(
            f"Candidate file must contain 'source1_entity_id' and 'matched_entity_id'. Found: {col_names}"
        )

    con.execute("""
        CREATE TEMP TABLE cands AS
        SELECT 
            TRIM(source1_entity_id) AS source1_entity_id,
            TRIM(matched_entity_id) AS matched_entity_id
        FROM raw_cands
    """)

    # Basic Counts
    total_candidate_rows = con.execute("SELECT COUNT(*) FROM cands").fetchone()[0]
    unique_candidate_rows = con.execute(
        "SELECT COUNT(DISTINCT (source1_entity_id, matched_entity_id)) FROM cands"
    ).fetchone()[0]
    duplicate_rows = total_candidate_rows - unique_candidate_rows
    distinct_s1_with_cands = con.execute(
        "SELECT COUNT(DISTINCT source1_entity_id) FROM cands"
    ).fetchone()[0]
    distinct_target_with_cands = con.execute(
        "SELECT COUNT(DISTINCT matched_entity_id) FROM cands"
    ).fetchone()[0]

    # Schema & Integrity Checks
    null_s1_count = con.execute(
        "SELECT COUNT(*) FROM cands WHERE source1_entity_id IS NULL OR source1_entity_id = ''"
    ).fetchone()[0]
    null_target_count = con.execute(
        "SELECT COUNT(*) FROM cands WHERE matched_entity_id IS NULL OR matched_entity_id = ''"
    ).fetchone()[0]
    self_match_count = con.execute(
        "SELECT COUNT(*) FROM cands WHERE source1_entity_id = matched_entity_id"
    ).fetchone()[0]

    # Format Regex Checks
    malformed_s1_format = con.execute(
        "SELECT COUNT(*) FROM cands WHERE regexp_matches(source1_entity_id, '^S1-[0-9]+$') = false"
    ).fetchone()[0]
    
    if det_source == "S2":
        malformed_target_format = con.execute(
            "SELECT COUNT(*) FROM cands WHERE regexp_matches(matched_entity_id, '^S2-[0-9]+$') = false"
        ).fetchone()[0]
    elif det_source == "S3":
        malformed_target_format = con.execute(
            "SELECT COUNT(*) FROM cands WHERE regexp_matches(matched_entity_id, '^S3-[0-9]+$') = false"
        ).fetchone()[0]
    else:
        malformed_target_format = con.execute(
            "SELECT COUNT(*) FROM cands WHERE regexp_matches(matched_entity_id, '^S[23]-[0-9]+$') = false"
        ).fetchone()[0]

    # ID Membership Integrity (Existence in canonical pool)
    id_integrity = con.execute("""
        SELECT 
            COUNT(CASE WHEN s1.entity_id IS NULL THEN 1 END) AS invalid_s1_ids,
            COUNT(CASE WHEN tgt.entity_id IS NULL THEN 1 END) AS invalid_target_ids,
            COUNT(CASE WHEN s1.country_normalized <> tgt.country_normalized THEN 1 END) AS cross_country_candidates
        FROM cands c
        LEFT JOIN s1_entities s1 ON c.source1_entity_id = s1.entity_id
        LEFT JOIN target_entities tgt ON c.matched_entity_id = tgt.entity_id
    """).fetchone()

    invalid_s1_ids = id_integrity[0]
    invalid_target_ids = id_integrity[1]
    cross_country_candidates = id_integrity[2]
    cross_country_rate = (cross_country_candidates / total_candidate_rows) if total_candidate_rows > 0 else 0.0

    # Fanout Statistics (across ALL entities in the split's S1 dataset)
    fanout_res = con.execute("""
        WITH s1_fanout AS (
            SELECT s1.entity_id, COUNT(c.matched_entity_id) as cand_count
            FROM s1_entities s1
            LEFT JOIN cands c ON s1.entity_id = c.source1_entity_id
            GROUP BY s1.entity_id
        )
        SELECT 
            COUNT(CASE WHEN cand_count = 0 THEN 1 END) AS zero_candidate_s1,
            MIN(cand_count) AS min_cands,
            MAX(cand_count) AS max_cands,
            AVG(cand_count) AS mean_cands,
            quantile_cont(cand_count, 0.50) AS p50_cands,
            quantile_cont(cand_count, 0.90) AS p90_cands,
            quantile_cont(cand_count, 0.95) AS p95_cands,
            quantile_cont(cand_count, 0.99) AS p99_cands,
            quantile_cont(cand_count, 0.999) AS p999_cands
        FROM s1_fanout
    """).fetchone()

    zero_candidate_s1 = fanout_res[0]
    zero_candidate_s1_rate = zero_candidate_s1 / total_s1_in_pool if total_s1_in_pool > 0 else 0.0
    min_cands_per_s1 = int(fanout_res[1])
    max_cands_per_s1 = int(fanout_res[2])
    mean_cands_per_s1 = float(fanout_res[3])
    p50_cands_per_s1 = float(fanout_res[4])
    p90_cands_per_s1 = float(fanout_res[5])
    p95_cands_per_s1 = float(fanout_res[6])
    p99_cands_per_s1 = float(fanout_res[7])
    p999_cands_per_s1 = float(fanout_res[8])

    # Ground Truth Evaluation (Train Split Only)
    gt_metrics: Dict[str, Any] = {}
    if det_split == "train":
        if not Path(ground_truth_file).exists():
            raise FileNotFoundError(f"Ground truth file not found: {ground_truth_file}")

        con.execute(f"""
            CREATE TEMP TABLE raw_gt AS
            SELECT 
                TRIM(source1_entity_id) AS source1_entity_id, 
                TRIM(UNNEST(string_split(matched_entity_ids, ','))) AS matched_entity_id
            FROM read_csv('{ground_truth_file}', delim='\\t', header=true, all_varchar=true)
            WHERE matched_entity_ids IS NOT NULL AND TRIM(matched_entity_ids) <> ''
        """)

        if det_source == "S2":
            con.execute("CREATE TEMP TABLE filtered_gt AS SELECT * FROM raw_gt WHERE matched_entity_id LIKE 'S2-%'")
        elif det_source == "S3":
            con.execute("CREATE TEMP TABLE filtered_gt AS SELECT * FROM raw_gt WHERE matched_entity_id LIKE 'S3-%'")
        else:
            con.execute("CREATE TEMP TABLE filtered_gt AS SELECT * FROM raw_gt")

        gt_total = con.execute("SELECT COUNT(*) FROM filtered_gt").fetchone()[0]
        gt_captured = con.execute("""
            SELECT COUNT(DISTINCT (c.source1_entity_id, c.matched_entity_id))
            FROM cands c
            JOIN filtered_gt gt ON c.source1_entity_id = gt.source1_entity_id AND c.matched_entity_id = gt.matched_entity_id
        """).fetchone()[0]
        gt_missed = gt_total - gt_captured
        recall = (gt_captured / gt_total) if gt_total > 0 else 0.0
        candidate_to_gt_ratio = (total_candidate_rows / gt_total) if gt_total > 0 else 0.0
        true_candidate_rate = (gt_captured / total_candidate_rows) if total_candidate_rows > 0 else 0.0
        false_candidate_count = total_candidate_rows - gt_captured

        gt_metrics = {
            "gt_pairs": gt_total,
            "gt_captured": gt_captured,
            "gt_missed": gt_missed,
            "recall": recall,
            "candidate_to_gt_ratio": candidate_to_gt_ratio,
            "true_candidate_rate": true_candidate_rate,
            "false_candidate_count": false_candidate_count,
        }
    else:
        gt_metrics = {
            "gt_pairs": "NOT_APPLICABLE",
            "gt_captured": "NOT_APPLICABLE",
            "gt_missed": "NOT_APPLICABLE",
            "recall": "NOT_APPLICABLE",
            "candidate_to_gt_ratio": "NOT_APPLICABLE",
            "true_candidate_rate": "NOT_APPLICABLE",
            "false_candidate_count": "NOT_APPLICABLE",
        }

    eval_runtime = round(time.time() - t0, 3)
    eval_timestamp = datetime.now(timezone.utc).isoformat()

    results: Dict[str, Any] = {
        "candidate_file": str(candidate_path),
        "filename": candidate_path.name,
        "split": det_split,
        "source": det_source,
        "file_size_bytes": file_size_bytes,
        "sha256": file_sha256,
        "candidate_rows": total_candidate_rows,
        "unique_candidate_rows": unique_candidate_rows,
        "duplicate_rows": duplicate_rows,
        "distinct_s1_with_cands": distinct_s1_with_cands,
        "distinct_target_with_cands": distinct_target_with_cands,
        "total_s1_entities_in_pool": total_s1_in_pool,
        "total_target_entities_in_pool": total_target_in_pool,
        "null_s1_count": null_s1_count,
        "null_target_count": null_target_count,
        "self_match_count": self_match_count,
        "malformed_s1_format": malformed_s1_format,
        "malformed_target_format": malformed_target_format,
        "invalid_s1_ids": invalid_s1_ids,
        "invalid_target_ids": invalid_target_ids,
        "cross_country_candidates": cross_country_candidates,
        "cross_country_rate": cross_country_rate,
        "zero_candidate_s1": zero_candidate_s1,
        "zero_candidate_s1_rate": zero_candidate_s1_rate,
        "min_candidates_per_s1": min_cands_per_s1,
        "max_candidates_per_s1": max_cands_per_s1,
        "mean_candidates_per_s1": mean_cands_per_s1,
        "median_candidates_per_s1": p50_cands_per_s1,
        "p90_candidates_per_s1": p90_cands_per_s1,
        "p95_candidates_per_s1": p95_cands_per_s1,
        "p99_candidates_per_s1": p99_cands_per_s1,
        "p999_candidates_per_s1": p999_cands_per_s1,
        "evaluation_script": "P1/scripts/evaluation/evaluate_candidate_set.py",
        "evaluation_runtime_seconds": eval_runtime,
        "evaluation_timestamp": eval_timestamp,
        **gt_metrics,
    }

    if not quiet:
        print(f"Results for {candidate_path.name}:", flush=True)
        print(f"  Rows: {total_candidate_rows:,} | Unique: {unique_candidate_rows:,} | Dups: {duplicate_rows}", flush=True)
        if det_split == "train":
            print(f"  GT Pairs: {gt_metrics['gt_pairs']:,} | Captured: {gt_metrics['gt_captured']:,} | Missed: {gt_metrics['gt_missed']:,} | Recall: {gt_metrics['recall']:.6%}", flush=True)
            print(f"  Cand/GT Ratio: {gt_metrics['candidate_to_gt_ratio']:.3f} | True Cand Rate: {gt_metrics['true_candidate_rate']:.4%}", flush=True)
        print(f"  Fanout per S1: zero={zero_candidate_s1:,} ({zero_candidate_s1_rate:.2%}) | median={p50_cands_per_s1:.1f} | p95={p95_cands_per_s1:.1f} | max={max_cands_per_s1:,}", flush=True)
        print(f"  Diagnostics: cross_country={cross_country_candidates} | invalid_s1={invalid_s1_ids} | invalid_tgt={invalid_target_ids}", flush=True)
        print(f"  Runtime: {eval_runtime:.2f}s | SHA256: {file_sha256[:16]}...", flush=True)

    if output_report:
        out_p = Path(output_report).resolve()
        out_p.parent.mkdir(parents=True, exist_ok=True)
        if output_format.lower() == "tsv" or out_p.suffix.lower() == ".tsv":
            # TSV single row or header + row
            header = list(results.keys())
            values = [str(results[k]) for k in header]
            with open(out_p, "w", encoding="utf-8") as f:
                f.write("\t".join(header) + "\n")
                f.write("\t".join(values) + "\n")
        else:
            with open(out_p, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)
        if not quiet:
            print(f"Report written to: {out_p}", flush=True)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="P1 Canonical Candidate Set Evaluator (Amazon ML Challenge 2026)"
    )
    parser.add_argument("--candidate-file", required=True, help="Path to candidate TSV or Parquet file")
    parser.add_argument("--candidate-source", choices=["S2", "S3", "COMBINED", "auto"], default="auto", help="Target entity source")
    parser.add_argument("--split", choices=["train", "test", "auto"], default="auto", help="Dataset split")
    parser.add_argument("--ground-truth", default=None, help="Path to train ground truth file")
    parser.add_argument("--entities-dir", default=None, help="Base directory of canonical entities")
    parser.add_argument("--entities-s1", default=None, help="Path to S1 entities parquet")
    parser.add_argument("--entities-target", default=None, help="Path to Target entities parquet")
    parser.add_argument("--output-report", default=None, help="Path to write report (JSON or TSV)")
    parser.add_argument("--format", choices=["json", "tsv"], default="json", help="Report format")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose stdout output")

    args = parser.parse_args()

    src = None if args.candidate_source == "auto" else args.candidate_source
    splt = None if args.split == "auto" else args.split

    evaluate_candidate_set(
        candidate_file=args.candidate_file,
        candidate_source=src,
        split=splt,
        ground_truth_file=args.ground_truth,
        entities_dir=args.entities_dir,
        entities_s1=args.entities_s1,
        entities_target=args.entities_target,
        output_report=args.output_report,
        output_format=args.format,
        quiet=args.quiet,
    )


if __name__ == "__main__":
    main()
