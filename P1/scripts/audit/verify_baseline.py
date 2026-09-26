#!/usr/bin/env python3
"""
Phase 0 — Baseline Verification Script
Computes exact file sizes, SHA256 hashes, row counts, unique IDs, null/empty IDs,
schema, and candidate statistics using DuckDB and hashlib.
Does not modify any repository data.
"""

import os
import sys
import json
import time
import hashlib
import duckdb

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

def compute_sha256(filepath, max_bytes=None):
    """Computes SHA256 hash of a file."""
    if not os.path.exists(filepath):
        return None
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
            if max_bytes and f.tell() >= max_bytes:
                break
    return h.hexdigest()

def get_file_info(filepath):
    if not os.path.exists(filepath):
        return {"exists": False, "size_bytes": 0, "sha256": None}
    size = os.path.getsize(filepath)
    t0 = time.time()
    sha = compute_sha256(filepath)
    dt = time.time() - t0
    return {"exists": True, "size_bytes": size, "sha256": sha, "sha_sec": dt}

def main():
    con = duckdb.connect()
    con.execute("PRAGMA threads=8;")
    con.execute("PRAGMA preserve_insertion_order=false;")

    results = {}

    print("Verifying Raw and Normalized Datasets...")
    datasets = {
        "train_source1": os.path.join(REPO_ROOT, "data", "train", "train_source1.tsv"),
        "train_source2": os.path.join(REPO_ROOT, "data", "train", "train_source2.tsv"),
        "train_source3": os.path.join(REPO_ROOT, "data", "train", "train_source3.tsv"),
        "test_source1": os.path.join(REPO_ROOT, "data", "test", "test_source1.tsv"),
        "test_source2": os.path.join(REPO_ROOT, "data", "test", "test_source2.tsv"),
        "test_source3": os.path.join(REPO_ROOT, "data", "test", "test_source3.tsv"),
        "train_ground_truth": os.path.join(REPO_ROOT, "data", "train", "train_ground_truth.tsv"),
        "train_source1_normalized": os.path.join(REPO_ROOT, "outputs", "person1_step1", "normalized", "train_source1_normalized.tsv"),
        "train_source2_normalized": os.path.join(REPO_ROOT, "outputs", "person1_step1", "normalized", "train_source2_normalized.tsv"),
        "train_source3_normalized": os.path.join(REPO_ROOT, "outputs", "person1_step1", "normalized", "train_source3_normalized.tsv"),
        "test_source1_normalized": os.path.join(REPO_ROOT, "outputs", "person1_step1", "normalized", "test_source1_normalized.tsv"),
        "test_source2_normalized": os.path.join(REPO_ROOT, "outputs", "person1_step1", "normalized", "test_source2_normalized.tsv"),
        "test_source3_normalized": os.path.join(REPO_ROOT, "outputs", "person1_step1", "normalized", "test_source3_normalized.tsv"),
        "train_ground_truth_reconstructed": os.path.join(REPO_ROOT, "outputs", "person1_step1", "train_ground_truth_reconstructed.tsv")
    }

    ds_stats = {}
    for name, path in datasets.items():
        print(f"Checking {name}...")
        finfo = get_file_info(path)
        if not finfo["exists"]:
            ds_stats[name] = {"exists": False, "path": path}
            continue

        # Inspect schema and row count using DuckDB
        if "ground_truth" in name:
            schema_info = con.execute(f"DESCRIBE SELECT * FROM read_csv('{path}', sep='\\t', header=true)").fetchall()
            cols = [col[0] for col in schema_info]
            q = f"""
            SELECT 
                COUNT(*) as rows,
                COUNT(DISTINCT source1_entity_id) as unique_s1,
                COUNT(*) - COUNT(DISTINCT source1_entity_id) as dup_s1,
                SUM(CASE WHEN source1_entity_id IS NULL THEN 1 ELSE 0 END) as null_s1,
                SUM(CASE WHEN source1_entity_id = '' THEN 1 ELSE 0 END) as empty_s1,
                SUM(CASE WHEN matched_entity_ids IS NULL OR matched_entity_ids = '' THEN 1 ELSE 0 END) as empty_matches
            FROM read_csv('{path}', sep='\\t', header=true)
            """
            r = con.execute(q).fetchone()
            ds_stats[name] = {
                "exists": True,
                "path": path,
                "file_size_bytes": finfo["size_bytes"],
                "sha256": finfo["sha256"],
                "rows": r[0],
                "columns": len(cols),
                "column_names": cols,
                "unique_id_count": r[1],
                "duplicate_id_count": r[2],
                "null_id_count": r[3],
                "empty_id_count": r[4],
                "empty_matches": r[5]
            }
        else:
            schema_info = con.execute(f"DESCRIBE SELECT * FROM read_csv('{path}', sep='\\t', header=true)").fetchall()
            cols = [col[0] for col in schema_info]
            q = f"""
            SELECT 
                COUNT(*) as rows,
                COUNT(DISTINCT entity_id) as unique_id,
                COUNT(*) - COUNT(DISTINCT entity_id) as dup_id,
                SUM(CASE WHEN entity_id IS NULL THEN 1 ELSE 0 END) as null_id,
                SUM(CASE WHEN entity_id = '' THEN 1 ELSE 0 END) as empty_id,
                SUM(CASE WHEN business_name IS NULL OR business_name = '' THEN 1 ELSE 0 END) as empty_name,
                SUM(CASE WHEN business_address IS NULL OR business_address = '' THEN 1 ELSE 0 END) as empty_addr,
                SUM(CASE WHEN country IS NULL OR country = '' THEN 1 ELSE 0 END) as empty_country
            FROM read_csv('{path}', sep='\\t', header=true)
            """
            r = con.execute(q).fetchone()
            ds_stats[name] = {
                "exists": True,
                "path": path,
                "file_size_bytes": finfo["size_bytes"],
                "sha256": finfo["sha256"],
                "rows": r[0],
                "columns": len(cols),
                "column_names": cols,
                "unique_id_count": r[1],
                "duplicate_id_count": r[2],
                "null_id_count": r[3],
                "empty_id_count": r[4],
                "empty_name_count": r[5],
                "empty_address_count": r[6],
                "empty_country_count": r[7]
            }

    results["datasets"] = ds_stats

    # Ground truth exploded pairs
    print("Checking exploded Ground Truth metrics...")
    gt_path = datasets["train_ground_truth"]
    q_gt = f"""
    WITH unnested AS (
        SELECT 
            source1_entity_id,
            UNNEST(STRING_SPLIT(matched_entity_ids, ',')) as target_id
        FROM read_csv('{gt_path}', sep='\\t', header=true)
        WHERE matched_entity_ids IS NOT NULL AND matched_entity_ids <> ''
    )
    SELECT 
        COUNT(*) as total_exploded_pairs,
        SUM(CASE WHEN target_id LIKE 'source2_%' THEN 1 ELSE 0 END) as s2_pairs,
        SUM(CASE WHEN target_id LIKE 'source3_%' THEN 1 ELSE 0 END) as s3_pairs,
        SUM(CASE WHEN target_id NOT LIKE 'source2_%' AND target_id NOT LIKE 'source3_%' THEN 1 ELSE 0 END) as other_pairs,
        COUNT(DISTINCT CASE WHEN target_id LIKE 'source2_%' THEN source1_entity_id END) as s1_with_s2,
        COUNT(DISTINCT CASE WHEN target_id LIKE 'source3_%' THEN source1_entity_id END) as s1_with_s3
    FROM unnested
    """
    gt_res = con.execute(q_gt).fetchone()
    results["ground_truth_exploded"] = {
        "total_exploded_pairs": gt_res[0],
        "s2_pairs": gt_res[1],
        "s3_pairs": gt_res[2],
        "other_pairs": gt_res[3],
        "s1_with_s2": gt_res[4],
        "s1_with_s3": gt_res[5]
    }

    # Candidate files verification
    print("Verifying Candidate Files...")
    candidates = {
        "train_candidate_pairs_s2_v1": os.path.join(REPO_ROOT, "P2", "data", "candidates", "train_candidate_pairs_s2.tsv"),
        "train_candidate_pairs_s3_v1": os.path.join(REPO_ROOT, "P2", "data", "candidates", "train_candidate_pairs_s3.tsv"),
        "test_candidate_pairs_s2_v1": os.path.join(REPO_ROOT, "P2", "data", "candidates", "test_candidate_pairs_s2.tsv"),
        "test_candidate_pairs_s3_v1": os.path.join(REPO_ROOT, "P2", "data", "candidates", "test_candidate_pairs_s3.tsv"),
        "train_candidate_pairs_s2_v2": os.path.join(REPO_ROOT, "P2", "data", "candidates", "train_candidate_pairs_s2_v2.tsv"),
        "train_candidate_pairs_s3_v2": os.path.join(REPO_ROOT, "P2", "data", "candidates", "train_candidate_pairs_s3_v2.tsv"),
        "test_candidate_pairs_s2_v2": os.path.join(REPO_ROOT, "P2", "data", "candidates", "test_candidate_pairs_s2_v2.tsv"),
        "test_candidate_pairs_s3_v2": os.path.join(REPO_ROOT, "P2", "data", "candidates", "test_candidate_pairs_s3_v2.tsv"),
        "test_candidate_pairs_s2_p1": os.path.join(REPO_ROOT, "outputs", "person1_step1", "test_candidate_pairs_s2.tsv"),
        "test_candidate_pairs_s3_p1": os.path.join(REPO_ROOT, "outputs", "person1_step1", "test_candidate_pairs_s3.tsv")
    }

    cand_stats = {}
    for name, path in candidates.items():
        print(f"Checking candidate file {name}...")
        finfo = get_file_info(path)
        if not finfo["exists"]:
            cand_stats[name] = {"exists": False, "path": path}
            continue

        schema_info = con.execute(f"DESCRIBE SELECT * FROM read_csv('{path}', sep='\\t', header=true)").fetchall()
        cols = [col[0] for col in schema_info]
        has_label = "label" in cols
        if has_label:
            q = f"""
            SELECT 
                COUNT(*) as rows,
                COUNT(DISTINCT source1_entity_id) as unique_s1,
                COUNT(DISTINCT matched_entity_id) as unique_tgt,
                SUM(CASE WHEN label = 1 THEN 1 ELSE 0 END) as pos,
                SUM(CASE WHEN label = 0 THEN 1 ELSE 0 END) as neg
            FROM read_csv('{path}', sep='\\t', header=true)
            """
            r = con.execute(q).fetchone()
            cand_stats[name] = {
                "exists": True,
                "path": path,
                "file_size_bytes": finfo["size_bytes"],
                "sha256": finfo["sha256"],
                "rows": r[0],
                "columns": len(cols),
                "column_names": cols,
                "unique_s1": r[1],
                "unique_target": r[2],
                "positives": r[3],
                "negatives": r[4]
            }
        else:
            q = f"""
            SELECT 
                COUNT(*) as rows,
                COUNT(DISTINCT source1_entity_id) as unique_s1,
                COUNT(DISTINCT matched_entity_id) as unique_tgt
            FROM read_csv('{path}', sep='\\t', header=true)
            """
            r = con.execute(q).fetchone()
            cand_stats[name] = {
                "exists": True,
                "path": path,
                "file_size_bytes": finfo["size_bytes"],
                "sha256": finfo["sha256"],
                "rows": r[0],
                "columns": len(cols),
                "column_names": cols,
                "unique_s1": r[1],
                "unique_target": r[2],
                "positives": None,
                "negatives": None
            }

    results["candidates"] = cand_stats

    out_file = os.path.join(REPO_ROOT, "P1", "reports", "verified_baseline_cache.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Done! Results written to {out_file}")

if __name__ == "__main__":
    main()
