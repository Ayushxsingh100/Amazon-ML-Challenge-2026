import os
import duckdb
import numpy as np
import pandas as pd

REPO = r"c:\NEW AMAZON\Amazon-ML-Challenge-2026"
FOLDS_PATH = os.path.join(REPO, "P3", "reports", "folds_v1_manifest.tsv")
GT_PATH = os.path.join(REPO, "outputs", "person1_step1", "train_ground_truth_reconstructed.tsv")
OOF_PATH = os.path.join(REPO, "P2", "reports", "E02_OOF_FULL_CANONICAL.tsv")

con = duckdb.connect()
con.execute("SET threads=4")
con.execute("SET memory_limit='4GB'")

print("Loading data for Candidate Recall Analysis...")
con.execute(f"CREATE TABLE s1_all AS SELECT source1_entity_id, fold FROM read_csv('{FOLDS_PATH}', delim='\\t', header=true)")
con.execute(f"""
    CREATE TABLE gt_counts AS 
    SELECT 
        source1_entity_id,
        CASE WHEN matched_entity_ids IS NULL OR matched_entity_ids = '' THEN 0 ELSE len(string_split(matched_entity_ids, ',')) END AS A,
        len(list_filter(string_split(coalesce(matched_entity_ids, ''), ','), x -> x LIKE 'S2-%')) AS A_s2,
        len(list_filter(string_split(coalesce(matched_entity_ids, ''), ','), x -> x LIKE 'S3-%')) AS A_s3
    FROM read_csv('{GT_PATH}', delim='\\t', header=true)
""")

con.execute(f"""
    CREATE TABLE cand_counts AS
    SELECT 
        source1_entity_id,
        count(*) AS total_cands,
        sum(case when matched_entity_id like 'S2-%' then 1 else 0 end) AS s2_cands,
        sum(case when matched_entity_id like 'S3-%' then 1 else 0 end) AS s3_cands,
        sum(target) AS K,
        sum(case when matched_entity_id like 'S2-%' and target=1 then 1 else 0 end) AS K_s2,
        sum(case when matched_entity_id like 'S3-%' and target=1 then 1 else 0 end) AS K_s3
    FROM read_csv('{OOF_PATH}', delim='\\t', header=true)
    GROUP BY source1_entity_id
""")

con.execute("""
    CREATE TABLE s1_recall AS
    SELECT 
        s.source1_entity_id,
        s.fold,
        coalesce(g.A, 0) AS A,
        coalesce(g.A_s2, 0) AS A_s2,
        coalesce(g.A_s3, 0) AS A_s3,
        coalesce(c.total_cands, 0) AS total_cands,
        coalesce(c.s2_cands, 0) AS s2_cands,
        coalesce(c.s3_cands, 0) AS s3_cands,
        coalesce(c.K, 0) AS K,
        coalesce(c.K_s2, 0) AS K_s2,
        coalesce(c.K_s3, 0) AS K_s3
    FROM s1_all s
    LEFT JOIN gt_counts g ON s.source1_entity_id = g.source1_entity_id
    LEFT JOIN cand_counts c ON s.source1_entity_id = c.source1_entity_id
""")

# Aggregates for S2, S3, and Joint
stats = {}
for name, col_A, col_K, col_cands in [("S2", "A_s2", "K_s2", "s2_cands"), ("S3", "A_s3", "K_s3", "s3_cands"), ("Joint", "A", "K", "total_cands")]:
    res = con.execute(f"""
        SELECT 
            sum({col_A}) AS total_truth_pairs,
            sum({col_K}) AS captured_truth_pairs,
            sum({col_K}) * 1.0 / sum({col_A}) AS recall,
            count(CASE WHEN {col_A} = 0 THEN 1 END) AS s1_zero_truth,
            count(CASE WHEN {col_A} > 0 AND {col_K} = 0 THEN 1 END) AS s1_zero_captured,
            count(CASE WHEN {col_A} > 0 AND {col_K} > 0 AND {col_K} < {col_A} THEN 1 END) AS s1_partial_capture,
            count(CASE WHEN {col_A} > 0 AND {col_K} = {col_A} THEN 1 END) AS s1_full_capture,
            count(CASE WHEN {col_A} > 0 THEN 1 END) AS s1_with_truth,
            -- candidate distribution percentiles
            min({col_cands}),
            percentile_cont(0.25) WITHIN GROUP (ORDER BY {col_cands}),
            percentile_cont(0.50) WITHIN GROUP (ORDER BY {col_cands}),
            percentile_cont(0.75) WITHIN GROUP (ORDER BY {col_cands}),
            percentile_cont(0.90) WITHIN GROUP (ORDER BY {col_cands}),
            percentile_cont(0.95) WITHIN GROUP (ORDER BY {col_cands}),
            percentile_cont(0.99) WITHIN GROUP (ORDER BY {col_cands}),
            max({col_cands}),
            avg({col_cands}),
            stddev({col_cands})
        FROM s1_recall
    """).fetchone()
    
    stats[name] = {
        "total_truth_pairs": int(res[0]),
        "captured_truth_pairs": int(res[1]),
        "recall": float(res[2]),
        "s1_zero_truth": int(res[3]),
        "s1_zero_captured": int(res[4]),
        "s1_partial_capture": int(res[5]),
        "s1_full_capture": int(res[6]),
        "s1_with_truth": int(res[7]),
        "cands_min": int(res[8]),
        "cands_p25": float(res[9]),
        "cands_p50": float(res[10]),
        "cands_p75": float(res[11]),
        "cands_p90": float(res[12]),
        "cands_p95": float(res[13]),
        "cands_p99": float(res[14]),
        "cands_max": int(res[15]),
        "cands_mean": float(res[16]),
        "cands_std": float(res[17])
    }

for k, v in stats.items():
    print(f"\n=== {k} RECALL & DISTRIBUTION ===")
    for metric, val in v.items():
        print(f"  {metric}: {val}")
