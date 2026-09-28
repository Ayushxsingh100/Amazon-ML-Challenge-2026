import os
import time
import duckdb
import numpy as np
import pandas as pd

REPO = r"c:\NEW AMAZON\Amazon-ML-Challenge-2026"
FOLDS_PATH = os.path.join(REPO, "P3", "reports", "folds_v1_manifest.tsv")
GT_PATH = os.path.join(REPO, "outputs", "person1_step1", "train_ground_truth_reconstructed.tsv")
REPORT_DIR = os.path.join(REPO, "P2", "reports")
S2_PARQUET = os.path.join(REPO, "P2", "data", "v2_s2_agg.parquet")
S3_PARQUET = os.path.join(REPO, "P2", "data", "v2_s3_agg.parquet")
OUT_TSV = os.path.join(REPORT_DIR, "C01_V2_ACCEPTANCE.tsv")

con = duckdb.connect()
con.execute("SET threads=4")
con.execute("SET memory_limit='4GB'")

t0 = time.time()
print("1. Loading folds and ground truth...")
con.execute(f"CREATE TABLE folds AS SELECT source1_entity_id, fold FROM read_csv('{FOLDS_PATH}', delim='\\t', header=true)")
con.execute(f"""
    CREATE TABLE gt_counts AS 
    SELECT 
        source1_entity_id, 
        CASE WHEN matched_entity_ids IS NULL OR matched_entity_ids = '' THEN 0 ELSE len(string_split(matched_entity_ids, ',')) END AS A,
        len(list_filter(string_split(coalesce(matched_entity_ids, ''), ','), x -> x LIKE 'S2-%')) AS A_s2,
        len(list_filter(string_split(coalesce(matched_entity_ids, ''), ','), x -> x LIKE 'S3-%')) AS A_s3
    FROM read_csv('{GT_PATH}', delim='\\t', header=true)
""")

print("2. Joining aggregated S2 and S3 Parquet tables...")
con.execute(f"""
    CREATE TABLE s1_v2_summary AS
    SELECT 
        f.source1_entity_id,
        f.fold,
        coalesce(g.A, 0) AS A,
        coalesce(g.A_s2, 0) AS A_s2,
        coalesce(g.A_s3, 0) AS A_s3,
        coalesce(s2.cands_s2, 0) AS cands_s2,
        coalesce(s3.cands_s3, 0) AS cands_s3,
        coalesce(s2.cands_s2, 0) + coalesce(s3.cands_s3, 0) AS total_cands_v2,
        coalesce(s2.k_s2, 0) AS k_s2,
        coalesce(s3.k_s3, 0) AS k_s3,
        coalesce(s2.k_s2, 0) + coalesce(s3.k_s3, 0) AS k_v2,
        -- Candidate Oracle Macro F0.5
        CASE 
            WHEN coalesce(g.A, 0) = 0 THEN 1.0
            WHEN coalesce(s2.k_s2, 0) + coalesce(s3.k_s3, 0) = 0 THEN 0.0
            ELSE (5.0 * (coalesce(s2.k_s2, 0) + coalesce(s3.k_s3, 0))) / (g.A + 4.0 * (coalesce(s2.k_s2, 0) + coalesce(s3.k_s3, 0)))
        END AS cand_oracle_v2,
        -- S2 diagnostic
        CASE 
            WHEN coalesce(g.A_s2, 0) = 0 THEN 1.0
            WHEN coalesce(s2.k_s2, 0) = 0 THEN 0.0
            ELSE (5.0 * coalesce(s2.k_s2, 0)) / (g.A_s2 + 4.0 * coalesce(s2.k_s2, 0))
        END AS cand_oracle_v2_s2,
        -- S3 diagnostic
        CASE 
            WHEN coalesce(g.A_s3, 0) = 0 THEN 1.0
            WHEN coalesce(s3.k_s3, 0) = 0 THEN 0.0
            ELSE (5.0 * coalesce(s3.k_s3, 0)) / (g.A_s3 + 4.0 * coalesce(s3.k_s3, 0))
        END AS cand_oracle_v2_s3
    FROM folds f
    LEFT JOIN gt_counts g ON f.source1_entity_id = g.source1_entity_id
    LEFT JOIN read_parquet('{S2_PARQUET}') s2 ON f.source1_entity_id = s2.source1_entity_id
    LEFT JOIN read_parquet('{S3_PARQUET}') s3 ON f.source1_entity_id = s3.source1_entity_id
""")

# Overall metrics
res = con.execute("""
    SELECT 
        count(*),
        avg(cand_oracle_v2),
        avg(cand_oracle_v2_s2),
        avg(cand_oracle_v2_s3),
        sum(total_cands_v2),
        sum(k_v2),
        sum(cands_s2),
        sum(k_s2),
        sum(cands_s3),
        sum(k_s3)
    FROM s1_v2_summary
""").fetchone()

v2_joint_oracle = res[1]
v2_s2_oracle = res[2]
v2_s3_oracle = res[3]
v2_total_cands = res[4]
v2_captured_truth = res[5]
v2_s2_cands = res[6]
v2_s2_k = res[7]
v2_s3_cands = res[8]
v2_s3_k = res[9]

print("=" * 60)
print("=== V2 CANDIDATE ORACLE RESULTS ===")
print("=" * 60)
print(f"Overall Candidate Oracle V2 (Joint): {v2_joint_oracle:.9f}")
print(f"Overall Candidate Oracle V2 (S2):    {v2_s2_oracle:.9f}")
print(f"Overall Candidate Oracle V2 (S3):    {v2_s3_oracle:.9f}")
print(f"Total V2 Candidates: {v2_total_cands:,} (S2: {v2_s2_cands:,}, S3: {v2_s3_cands:,})")
print(f"Total V2 Captured True Pairs: {v2_captured_truth:,} (S2: {v2_s2_k:,}, S3: {v2_s3_k:,})")

# Per fold
v2_folds = con.execute("""
    SELECT 
        fold,
        count(*),
        avg(cand_oracle_v2),
        avg(cand_oracle_v2_s2),
        avg(cand_oracle_v2_s3)
    FROM s1_v2_summary
    GROUP BY fold
    ORDER BY fold
""").fetchall()

fold_scores = [f[2] for f in v2_folds]
fold_mean = np.mean(fold_scores)
fold_std = np.std(fold_scores, ddof=1)

print("\nPer-Fold Candidate Oracle V2:")
for f in v2_folds:
    print(f"  Fold {f[0]}: Joint={f[2]:.9f}, S2={f[3]:.9f}, S3={f[4]:.9f} (n={f[1]:,})")
print(f"Fold Mean +/- Std: {fold_mean:.9f} +/- {fold_std:.9f}")

# Capture buckets
buckets = con.execute("""
    SELECT 
        CASE 
            WHEN A = 0 THEN 'TRUE_NO_MATCH'
            WHEN k_v2 = 0 THEN 'ZERO_CAPTURED'
            WHEN k_v2 < A THEN 'PARTIAL_CAPTURED'
            ELSE 'FULL_CAPTURED'
        END AS bucket,
        count(*) AS cnt,
        count(*) * 100.0 / 2206821 AS pct
    FROM s1_v2_summary
    GROUP BY 1
    ORDER BY cnt DESC
""").fetchall()

print("\nV2 S1 Capture Buckets:")
v2_bucket_dict = {}
for b in buckets:
    v2_bucket_dict[b[0]] = (b[1], b[2])
    print(f"  {b[0]}: {b[1]:,} ({b[2]:.2f}%)")

# Distribution
dist = con.execute("""
    SELECT 
        min(total_cands_v2),
        percentile_cont(0.25) WITHIN GROUP (ORDER BY total_cands_v2),
        percentile_cont(0.50) WITHIN GROUP (ORDER BY total_cands_v2),
        percentile_cont(0.75) WITHIN GROUP (ORDER BY total_cands_v2),
        percentile_cont(0.90) WITHIN GROUP (ORDER BY total_cands_v2),
        percentile_cont(0.95) WITHIN GROUP (ORDER BY total_cands_v2),
        percentile_cont(0.99) WITHIN GROUP (ORDER BY total_cands_v2),
        max(total_cands_v2),
        avg(total_cands_v2),
        stddev(total_cands_v2)
    FROM s1_v2_summary
""").fetchone()

print("\nV2 Candidates Per S1 Distribution:")
print(f"  Min: {dist[0]}")
print(f"  P25: {dist[1]}")
print(f"  P50 (Median): {dist[2]}")
print(f"  P75: {dist[3]}")
print(f"  P90: {dist[4]}")
print(f"  P95: {dist[5]}")
print(f"  P99: {dist[6]}")
print(f"  Max: {dist[7]}")
print(f"  Mean +/- Std: {dist[8]:.2f} +/- {dist[9]:.2f}")

# Comparison with V1
v1_oracle = 0.784522591
oracle_gain = v2_joint_oracle - v1_oracle

print("\n" + "=" * 60)
print(f"V1 Candidate Oracle: {v1_oracle:.9f}")
print(f"V2 Candidate Oracle: {v2_joint_oracle:.9f}")
print(f"ORACLE GAIN:        {oracle_gain:+.9f} ({oracle_gain*100/v1_oracle:+.2f}%)")
print("=" * 60)

# Build and save comparison TSV
comp_rows = [
    # Candidate Oracle Metrics
    {"category": "candidate_oracle", "metric": "joint_macro_f05", "v1_value": v1_oracle, "v2_value": v2_joint_oracle, "delta": oracle_gain, "pct_change": (oracle_gain/v1_oracle)*100},
    {"category": "candidate_oracle", "metric": "s2_diagnostic_f05", "v1_value": 0.714624767, "v2_value": v2_s2_oracle, "delta": v2_s2_oracle - 0.714624767, "pct_change": ((v2_s2_oracle - 0.714624767)/0.714624767)*100},
    {"category": "candidate_oracle", "metric": "s3_diagnostic_f05", "v1_value": 0.724138181, "v2_value": v2_s3_oracle, "delta": v2_s3_oracle - 0.724138181, "pct_change": ((v2_s3_oracle - 0.724138181)/0.724138181)*100},
    
    # Candidate Volumes
    {"category": "candidate_volume", "metric": "total_candidates", "v1_value": 54592725, "v2_value": v2_total_cands, "delta": v2_total_cands - 54592725, "pct_change": ((v2_total_cands - 54592725)/54592725)*100},
    {"category": "candidate_volume", "metric": "s2_candidates", "v1_value": 24594064, "v2_value": v2_s2_cands, "delta": v2_s2_cands - 24594064, "pct_change": ((v2_s2_cands - 24594064)/24594064)*100},
    {"category": "candidate_volume", "metric": "s3_candidates", "v1_value": 29998661, "v2_value": v2_s3_cands, "delta": v2_s3_cands - 29998661, "pct_change": ((v2_s3_cands - 29998661)/29998661)*100},
    
    # Candidate Recall
    {"category": "truth_recall", "metric": "captured_truth_pairs", "v1_value": 4560579, "v2_value": v2_captured_truth, "delta": v2_captured_truth - 4560579, "pct_change": ((v2_captured_truth - 4560579)/4560579)*100},
    {"category": "truth_recall", "metric": "candidate_recall_pct", "v1_value": 59.7062, "v2_value": (v2_captured_truth * 100.0 / 7638365), "delta": (v2_captured_truth * 100.0 / 7638365) - 59.7062, "pct_change": (((v2_captured_truth * 100.0 / 7638365) - 59.7062)/59.7062)*100},
    {"category": "truth_recall", "metric": "missed_truth_pairs", "v1_value": 3077786, "v2_value": (7638365 - v2_captured_truth), "delta": (7638365 - v2_captured_truth) - 3077786, "pct_change": (((7638365 - v2_captured_truth) - 3077786)/3077786)*100},
    
    # S1 Capture Buckets
    {"category": "s1_capture_buckets", "metric": "true_no_match", "v1_value": 123247, "v2_value": v2_bucket_dict['TRUE_NO_MATCH'][0], "delta": v2_bucket_dict['TRUE_NO_MATCH'][0] - 123247, "pct_change": 0.0},
    {"category": "s1_capture_buckets", "metric": "zero_captured_truth", "v1_value": 258666, "v2_value": v2_bucket_dict['ZERO_CAPTURED'][0], "delta": v2_bucket_dict['ZERO_CAPTURED'][0] - 258666, "pct_change": ((v2_bucket_dict['ZERO_CAPTURED'][0] - 258666)/258666)*100},
    {"category": "s1_capture_buckets", "metric": "partial_captured_truth", "v1_value": 1279869, "v2_value": v2_bucket_dict['PARTIAL_CAPTURED'][0], "delta": v2_bucket_dict['PARTIAL_CAPTURED'][0] - 1279869, "pct_change": ((v2_bucket_dict['PARTIAL_CAPTURED'][0] - 1279869)/1279869)*100},
    {"category": "s1_capture_buckets", "metric": "full_captured_truth", "v1_value": 545039, "v2_value": v2_bucket_dict['FULL_CAPTURED'][0], "delta": v2_bucket_dict['FULL_CAPTURED'][0] - 545039, "pct_change": ((v2_bucket_dict['FULL_CAPTURED'][0] - 545039)/545039)*100},
    
    # Per S1 Candidate Volume Distribution
    {"category": "candidate_distribution", "metric": "cands_min", "v1_value": 0, "v2_value": dist[0], "delta": dist[0] - 0, "pct_change": 0.0},
    {"category": "candidate_distribution", "metric": "cands_p25", "v1_value": 2.0, "v2_value": dist[1], "delta": dist[1] - 2.0, "pct_change": ((dist[1] - 2.0)/2.0)*100},
    {"category": "candidate_distribution", "metric": "cands_p50", "v1_value": 5.0, "v2_value": dist[2], "delta": dist[2] - 5.0, "pct_change": ((dist[2] - 5.0)/5.0)*100},
    {"category": "candidate_distribution", "metric": "cands_p75", "v1_value": 16.0, "v2_value": dist[3], "delta": dist[3] - 16.0, "pct_change": ((dist[3] - 16.0)/16.0)*100},
    {"category": "candidate_distribution", "metric": "cands_p90", "v1_value": 63.0, "v2_value": dist[4], "delta": dist[4] - 63.0, "pct_change": ((dist[4] - 63.0)/63.0)*100},
    {"category": "candidate_distribution", "metric": "cands_p95", "v1_value": 123.0, "v2_value": dist[5], "delta": dist[5] - 123.0, "pct_change": ((dist[5] - 123.0)/123.0)*100},
    {"category": "candidate_distribution", "metric": "cands_p99", "v1_value": 284.0, "v2_value": dist[6], "delta": dist[6] - 284.0, "pct_change": ((dist[6] - 284.0)/284.0)*100},
    {"category": "candidate_distribution", "metric": "cands_max", "v1_value": 2972, "v2_value": dist[7], "delta": dist[7] - 2972, "pct_change": ((dist[7] - 2972)/2972)*100},
    {"category": "candidate_distribution", "metric": "cands_mean", "v1_value": 24.738175, "v2_value": dist[8], "delta": dist[8] - 24.738175, "pct_change": ((dist[8] - 24.738175)/24.738175)*100},
]

pd.DataFrame(comp_rows).to_csv(OUT_TSV, sep='\t', index=False)
print(f"Saved {OUT_TSV}")
print(f"Total script runtime: {time.time() - t0:.2f}s")
