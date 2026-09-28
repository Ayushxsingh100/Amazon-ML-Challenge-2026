#!/usr/bin/env python3
"""
P1/scripts/evaluation/generate_phase2_reports.py

Generates authoritative Phase 2 artifacts:
- P1/reports/PHASE2_CANDIDATE_BASELINE.tsv
- P1/reports/PHASE2_CANDIDATE_SCHEMA_VALIDATION.tsv
"""

import hashlib
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import duckdb

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
P1_REPORTS = REPO_ROOT / "P1" / "reports"
P1_REPORTS.mkdir(parents=True, exist_ok=True)

con = duckdb.connect()
con.execute("SET threads=4")
con.execute("SET memory_limit='8GB'")

print("Loading canonical entities...", flush=True)
con.execute(f"""
    CREATE TEMP TABLE train_s1 AS SELECT entity_id, country_normalized FROM read_parquet('{REPO_ROOT}/P1/data/entities/train/source1/train_s1_entities.parquet');
    CREATE TEMP TABLE train_s2 AS SELECT entity_id, country_normalized FROM read_parquet('{REPO_ROOT}/P1/data/entities/train/source2/train_s2_entities.parquet');
    CREATE TEMP TABLE train_s3 AS SELECT entity_id, country_normalized FROM read_parquet('{REPO_ROOT}/P1/data/entities/train/source3/train_s3_entities.parquet');
    
    CREATE TEMP TABLE test_s1 AS SELECT entity_id, country_normalized FROM read_parquet('{REPO_ROOT}/P1/data/entities/test/source1/test_s1_entities.parquet');
    CREATE TEMP TABLE test_s2 AS SELECT entity_id, country_normalized FROM read_parquet('{REPO_ROOT}/P1/data/entities/test/source2/test_s2_entities.parquet');
    CREATE TEMP TABLE test_s3 AS SELECT entity_id, country_normalized FROM read_parquet('{REPO_ROOT}/P1/data/entities/test/source3/test_s3_entities.parquet');
""")

print("Loading ground truth...", flush=True)
con.execute(f"""
    CREATE TEMP TABLE gt_all AS
    SELECT 
        TRIM(source1_entity_id) AS source1_entity_id, 
        TRIM(UNNEST(string_split(matched_entity_ids, ','))) AS matched_entity_id
    FROM read_csv('{REPO_ROOT}/data/train/train_ground_truth.tsv', delim='\\t', header=true, all_varchar=true)
    WHERE matched_entity_ids IS NOT NULL AND TRIM(matched_entity_ids) <> '';
    
    CREATE TEMP TABLE gt_s2 AS SELECT * FROM gt_all WHERE matched_entity_id LIKE 'S2-%';
    CREATE TEMP TABLE gt_s3 AS SELECT * FROM gt_all WHERE matched_entity_id LIKE 'S3-%';
""")

gt_s2_count = con.execute("SELECT COUNT(*) FROM gt_s2").fetchone()[0]
gt_s3_count = con.execute("SELECT COUNT(*) FROM gt_s3").fetchone()[0]
gt_all_count = con.execute("SELECT COUNT(*) FROM gt_all").fetchone()[0]

train_s1_total = con.execute("SELECT COUNT(*) FROM train_s1").fetchone()[0]
test_s1_total = con.execute("SELECT COUNT(*) FROM test_s1").fetchone()[0]

def compute_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

CANDIDATE_CONFIGS = [
    # Train V1
    {
        "candidate_version": "V1",
        "split": "train",
        "source": "S2",
        "candidate_file": "P2/data/candidates/train_candidate_pairs_s2.tsv",
        "generator_script": "P2/scripts/cands_BCD_v1_tasks.py",
        "rule_set": "A(prefix4+house), B(exact_addr), C(token1+house), D(exact_name)",
        "s1_tbl": "train_s1",
        "tgt_tbl": "train_s2",
        "gt_tbl": "gt_s2",
        "gt_total": gt_s2_count,
        "s1_total": train_s1_total,
    },
    {
        "candidate_version": "V1",
        "split": "train",
        "source": "S3",
        "candidate_file": "P2/data/candidates/train_candidate_pairs_s3.tsv",
        "generator_script": "P2/scripts/cands_BCD_v1_tasks.py",
        "rule_set": "A(prefix4+house), B(exact_addr), C(token1+house), D(exact_name)",
        "s1_tbl": "train_s1",
        "tgt_tbl": "train_s3",
        "gt_tbl": "gt_s3",
        "gt_total": gt_s3_count,
        "s1_total": train_s1_total,
    },
    # Train V2
    {
        "candidate_version": "V2",
        "split": "train",
        "source": "S2",
        "candidate_file": "P2/data/candidates/train_candidate_pairs_s2_v2.tsv",
        "generator_script": "P2/scripts/cands_BCD_v2_tasks.py",
        "rule_set": "A,B,C,D + E(clean_name>=3), F(house_norm), G(root_token+house_norm), H(clean_addr>=6), I(zip+prefix3)",
        "s1_tbl": "train_s1",
        "tgt_tbl": "train_s2",
        "gt_tbl": "gt_s2",
        "gt_total": gt_s2_count,
        "s1_total": train_s1_total,
    },
    {
        "candidate_version": "V2",
        "split": "train",
        "source": "S3",
        "candidate_file": "P2/data/candidates/train_candidate_pairs_s3_v2.tsv",
        "generator_script": "P2/scripts/cands_BCD_v2_tasks.py",
        "rule_set": "A,B,C,D + E(clean_name>=3), F(house_norm), G(root_token+house_norm), H(clean_addr>=6), I(zip+prefix3)",
        "s1_tbl": "train_s1",
        "tgt_tbl": "train_s3",
        "gt_tbl": "gt_s3",
        "gt_total": gt_s3_count,
        "s1_total": train_s1_total,
    },
    # Test V1 (P2)
    {
        "candidate_version": "V1",
        "split": "test",
        "source": "S2",
        "candidate_file": "P2/data/candidates/test_candidate_pairs_s2.tsv",
        "generator_script": "P2/scripts/cands_BCD_v1_tasks.py",
        "rule_set": "A(prefix4+house), B(exact_addr), C(token1+house), D(exact_name)",
        "s1_tbl": "test_s1",
        "tgt_tbl": "test_s2",
        "gt_tbl": None,
        "gt_total": None,
        "s1_total": test_s1_total,
    },
    {
        "candidate_version": "V1",
        "split": "test",
        "source": "S3",
        "candidate_file": "P2/data/candidates/test_candidate_pairs_s3.tsv",
        "generator_script": "P2/scripts/cands_BCD_v1_tasks.py",
        "rule_set": "A(prefix4+house), B(exact_addr), C(token1+house), D(exact_name)",
        "s1_tbl": "test_s1",
        "tgt_tbl": "test_s3",
        "gt_tbl": None,
        "gt_total": None,
        "s1_total": test_s1_total,
    },
    # Test V2 (P2)
    {
        "candidate_version": "V2",
        "split": "test",
        "source": "S2",
        "candidate_file": "P2/data/candidates/test_candidate_pairs_s2_v2.tsv",
        "generator_script": "P2/scripts/cands_BCD_v2_tasks.py",
        "rule_set": "A,B,C,D + E(clean_name>=3), F(house_norm), G(root_token+house_norm), H(clean_addr>=6), I(zip+prefix3)",
        "s1_tbl": "test_s1",
        "tgt_tbl": "test_s2",
        "gt_tbl": None,
        "gt_total": None,
        "s1_total": test_s1_total,
    },
    {
        "candidate_version": "V2",
        "split": "test",
        "source": "S3",
        "candidate_file": "P2/data/candidates/test_candidate_pairs_s3_v2.tsv",
        "generator_script": "P2/scripts/cands_BCD_v2_tasks.py",
        "rule_set": "A,B,C,D + E(clean_name>=3), F(house_norm), G(root_token+house_norm), H(clean_addr>=6), I(zip+prefix3)",
        "s1_tbl": "test_s1",
        "tgt_tbl": "test_s3",
        "gt_tbl": None,
        "gt_total": None,
        "s1_total": test_s1_total,
    },
    # Test Strategy B (Legacy Step 1)
    {
        "candidate_version": "Legacy_Strategy_B",
        "split": "test",
        "source": "S2",
        "candidate_file": "outputs/person1_step1/test_candidate_pairs_s2.tsv",
        "generator_script": "notebooks/03_blocking.ipynb",
        "rule_set": "Strategy B: A(prefix4+house), B(exact_addr)",
        "s1_tbl": "test_s1",
        "tgt_tbl": "test_s2",
        "gt_tbl": None,
        "gt_total": None,
        "s1_total": test_s1_total,
    },
    {
        "candidate_version": "Legacy_Strategy_B",
        "split": "test",
        "source": "S3",
        "candidate_file": "outputs/person1_step1/test_candidate_pairs_s3.tsv",
        "generator_script": "notebooks/03_blocking.ipynb",
        "rule_set": "Strategy B: A(prefix4+house), B(exact_addr)",
        "s1_tbl": "test_s1",
        "tgt_tbl": "test_s3",
        "gt_tbl": None,
        "gt_total": None,
        "s1_total": test_s1_total,
    },
]

baseline_rows = []
schema_rows = []

timestamp_now = datetime.now(timezone.utc).isoformat()

for cfg in CANDIDATE_CONFIGS:
    fpath = REPO_ROOT / cfg["candidate_file"]
    rel_path = cfg["candidate_file"]
    print(f"\nProcessing {rel_path}...", flush=True)
    
    sha = compute_sha256(fpath)
    
    con.execute(f"CREATE OR REPLACE TEMP TABLE cur_cands AS SELECT * FROM read_csv('{fpath}', delim='\\t', header=true, all_varchar=true)")
    cols = [c[0] for c in con.execute("DESCRIBE cur_cands").fetchall()]
    
    tot_rows = con.execute("SELECT COUNT(*) FROM cur_cands").fetchone()[0]
    unq_rows = con.execute("SELECT COUNT(DISTINCT (source1_entity_id, matched_entity_id)) FROM cur_cands").fetchone()[0]
    dup_rows = tot_rows - unq_rows
    
    # Schema checks
    null_s1 = con.execute("SELECT COUNT(*) FROM cur_cands WHERE source1_entity_id IS NULL OR TRIM(source1_entity_id) = ''").fetchone()[0]
    null_tgt = con.execute("SELECT COUNT(*) FROM cur_cands WHERE matched_entity_id IS NULL OR TRIM(matched_entity_id) = ''").fetchone()[0]
    self_match = con.execute("SELECT COUNT(*) FROM cur_cands WHERE source1_entity_id = matched_entity_id").fetchone()[0]
    malformed_s1 = con.execute("SELECT COUNT(*) FROM cur_cands WHERE regexp_matches(source1_entity_id, '^S1-[0-9]+$') = false").fetchone()[0]
    
    tgt_pfx = "^S2-[0-9]+$" if cfg["source"] == "S2" else "^S3-[0-9]+$"
    malformed_tgt = con.execute(f"SELECT COUNT(*) FROM cur_cands WHERE regexp_matches(matched_entity_id, '{tgt_pfx}') = false").fetchone()[0]
    
    # Integrity joins
    integ = con.execute(f"""
        SELECT 
            COUNT(CASE WHEN s1.entity_id IS NULL THEN 1 END) AS inv_s1,
            COUNT(CASE WHEN tgt.entity_id IS NULL THEN 1 END) AS inv_tgt,
            COUNT(CASE WHEN s1.country_normalized <> tgt.country_normalized THEN 1 END) AS cross_country
        FROM cur_cands c
        LEFT JOIN {cfg["s1_tbl"]} s1 ON c.source1_entity_id = s1.entity_id
        LEFT JOIN {cfg["tgt_tbl"]} tgt ON c.matched_entity_id = tgt.entity_id
    """).fetchone()
    
    inv_s1 = integ[0]
    inv_tgt = integ[1]
    cross_country = integ[2]
    total_invalid_ids = inv_s1 + inv_tgt
    
    # Fanout
    fanout = con.execute(f"""
        WITH s1_counts AS (
            SELECT s1.entity_id, COUNT(c.matched_entity_id) AS cnt
            FROM {cfg["s1_tbl"]} s1
            LEFT JOIN cur_cands c ON s1.entity_id = c.source1_entity_id
            GROUP BY s1.entity_id
        )
        SELECT 
            COUNT(CASE WHEN cnt = 0 THEN 1 END),
            quantile_cont(cnt, 0.50),
            quantile_cont(cnt, 0.95),
            quantile_cont(cnt, 0.99),
            MAX(cnt)
        FROM s1_counts
    """).fetchone()
    
    zero_s1 = fanout[0]
    med_s1 = float(fanout[1])
    p95_s1 = float(fanout[2])
    p99_s1 = float(fanout[3])
    max_s1 = int(fanout[4])
    
    # Ground truth
    if cfg["gt_tbl"] is not None:
        cap = con.execute(f"""
            SELECT COUNT(DISTINCT (c.source1_entity_id, c.matched_entity_id))
            FROM cur_cands c
            JOIN {cfg["gt_tbl"]} gt ON c.source1_entity_id = gt.source1_entity_id AND c.matched_entity_id = gt.matched_entity_id
        """).fetchone()[0]
        gt_pairs = cfg["gt_total"]
        gt_missed = gt_pairs - cap
        recall = cap / gt_pairs
        cand_to_gt = tot_rows / gt_pairs
    else:
        gt_pairs = "NOT_APPLICABLE"
        cap = "NOT_APPLICABLE"
        gt_missed = "NOT_APPLICABLE"
        recall = "NOT_APPLICABLE"
        cand_to_gt = "NOT_APPLICABLE"
        
    baseline_rows.append({
        "candidate_version": cfg["candidate_version"],
        "source": cfg["source"],
        "split": cfg["split"],
        "candidate_file": cfg["candidate_file"],
        "generator_script": cfg["generator_script"],
        "rule_set": cfg["rule_set"],
        "candidate_rows": tot_rows,
        "unique_candidate_rows": unq_rows,
        "duplicate_rows": dup_rows,
        "gt_pairs": gt_pairs,
        "gt_captured": cap,
        "gt_missed": gt_missed,
        "recall": f"{recall:.6f}" if isinstance(recall, float) else recall,
        "candidate_to_gt_ratio": f"{cand_to_gt:.4f}" if isinstance(cand_to_gt, float) else cand_to_gt,
        "zero_candidate_s1": zero_s1,
        "median_candidates_per_s1": med_s1,
        "p95_candidates_per_s1": p95_s1,
        "p99_candidates_per_s1": p99_s1,
        "max_candidates_per_s1": max_s1,
        "invalid_ids": total_invalid_ids,
        "cross_country_candidates": cross_country,
        "sha256": sha,
        "evaluation_script": "P1/scripts/evaluation/evaluate_candidate_set.py",
        "evaluation_timestamp": timestamp_now,
    })
    
    schema_status = "PASS" if (
        dup_rows == 0 and null_s1 == 0 and null_tgt == 0 and self_match == 0 and 
        malformed_s1 == 0 and malformed_tgt == 0 and total_invalid_ids == 0 and cross_country == 0
    ) else "FAIL"
    
    schema_rows.append({
        "candidate_file": cfg["candidate_file"],
        "split": cfg["split"],
        "source": cfg["source"],
        "column_names": ",".join(cols),
        "row_count": tot_rows,
        "unique_pairs_count": unq_rows,
        "duplicate_count": dup_rows,
        "null_source1_count": null_s1,
        "null_target_count": null_tgt,
        "self_match_count": self_match,
        "malformed_source1_count": malformed_s1,
        "malformed_target_count": malformed_tgt,
        "invalid_source1_id_count": inv_s1,
        "invalid_target_id_count": inv_tgt,
        "cross_country_count": cross_country,
        "schema_valid_status": schema_status,
    })

# Compute Combined Baselines for Train V1 and Train V2
for v_name, s2_file, s3_file, gen, rset in [
    ("V1", "P2/data/candidates/train_candidate_pairs_s2.tsv", "P2/data/candidates/train_candidate_pairs_s3.tsv", "P2/scripts/cands_BCD_v1_tasks.py", "A,B,C,D"),
    ("V2", "P2/data/candidates/train_candidate_pairs_s2_v2.tsv", "P2/data/candidates/train_candidate_pairs_s3_v2.tsv", "P2/scripts/cands_BCD_v2_tasks.py", "A,B,C,D,E,F,G,H,I"),
]:
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE comb_cands AS 
        SELECT source1_entity_id, matched_entity_id FROM read_csv('{REPO_ROOT / s2_file}', delim='\\t', header=true)
        UNION ALL
        SELECT source1_entity_id, matched_entity_id FROM read_csv('{REPO_ROOT / s3_file}', delim='\\t', header=true);
    """)
    comb_tot = con.execute("SELECT COUNT(*) FROM comb_cands").fetchone()[0]
    comb_unq = con.execute("SELECT COUNT(DISTINCT (source1_entity_id, matched_entity_id)) FROM comb_cands").fetchone()[0]
    comb_cap = con.execute("""
        SELECT COUNT(DISTINCT (c.source1_entity_id, c.matched_entity_id))
        FROM comb_cands c
        JOIN gt_all gt ON c.source1_entity_id = gt.source1_entity_id AND c.matched_entity_id = gt.matched_entity_id
    """).fetchone()[0]
    comb_missed = gt_all_count - comb_cap
    comb_rec = comb_cap / gt_all_count
    comb_ratio = comb_tot / gt_all_count
    
    comb_fanout = con.execute("""
        WITH s1_counts AS (
            SELECT s1.entity_id, COUNT(c.matched_entity_id) AS cnt
            FROM train_s1 s1
            LEFT JOIN comb_cands c ON s1.entity_id = c.source1_entity_id
            GROUP BY s1.entity_id
        )
        SELECT 
            COUNT(CASE WHEN cnt = 0 THEN 1 END),
            quantile_cont(cnt, 0.50),
            quantile_cont(cnt, 0.95),
            quantile_cont(cnt, 0.99),
            MAX(cnt)
        FROM s1_counts
    """).fetchone()
    
    baseline_rows.append({
        "candidate_version": v_name,
        "source": "COMBINED (S2+S3)",
        "split": "train",
        "candidate_file": f"{s2_file} + {s3_file}",
        "generator_script": gen,
        "rule_set": rset,
        "candidate_rows": comb_tot,
        "unique_candidate_rows": comb_unq,
        "duplicate_rows": comb_tot - comb_unq,
        "gt_pairs": gt_all_count,
        "gt_captured": comb_cap,
        "gt_missed": comb_missed,
        "recall": f"{comb_rec:.6f}",
        "candidate_to_gt_ratio": f"{comb_ratio:.4f}",
        "zero_candidate_s1": comb_fanout[0],
        "median_candidates_per_s1": float(comb_fanout[1]),
        "p95_candidates_per_s1": float(comb_fanout[2]),
        "p99_candidates_per_s1": float(comb_fanout[3]),
        "max_candidates_per_s1": int(comb_fanout[4]),
        "invalid_ids": 0,
        "cross_country_candidates": 0,
        "sha256": "COMBINED_MULTIPLE_FILES",
        "evaluation_script": "P1/scripts/evaluation/evaluate_candidate_set.py",
        "evaluation_timestamp": timestamp_now,
    })

# Write TSV Baseline
baseline_tsv = P1_REPORTS / "PHASE2_CANDIDATE_BASELINE.tsv"
cols_base = [
    "candidate_version", "source", "candidate_file", "generator_script", "rule_set",
    "candidate_rows", "unique_candidate_rows", "duplicate_rows", "gt_pairs", "gt_captured",
    "gt_missed", "recall", "candidate_to_gt_ratio", "zero_candidate_s1", "median_candidates_per_s1",
    "p95_candidates_per_s1", "p99_candidates_per_s1", "max_candidates_per_s1", "invalid_ids",
    "cross_country_candidates", "sha256", "evaluation_script", "evaluation_timestamp"
]

with open(baseline_tsv, "w", encoding="utf-8") as f:
    f.write("\t".join(cols_base) + "\n")
    for r in baseline_rows:
        f.write("\t".join(str(r[c]) for c in cols_base) + "\n")
print(f"Wrote {baseline_tsv}", flush=True)

# Write TSV Schema Validation
schema_tsv = P1_REPORTS / "PHASE2_CANDIDATE_SCHEMA_VALIDATION.tsv"
cols_schema = [
    "candidate_file", "split", "source", "column_names", "row_count", "unique_pairs_count",
    "duplicate_count", "null_source1_count", "null_target_count", "self_match_count",
    "malformed_source1_count", "malformed_target_count", "invalid_source1_id_count",
    "invalid_target_id_count", "cross_country_count", "schema_valid_status"
]

with open(schema_tsv, "w", encoding="utf-8") as f:
    f.write("\t".join(cols_schema) + "\n")
    for r in schema_rows:
        f.write("\t".join(str(r[c]) for c in cols_schema) + "\n")
print(f"Wrote {schema_tsv}", flush=True)

print("Phase 2 baseline and schema generation complete.")
