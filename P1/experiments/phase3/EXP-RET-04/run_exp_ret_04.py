#!/usr/bin/env python3
"""
P1/experiments/phase3/EXP-RET-04/run_exp_ret_04.py

EXP-RET-04: Locality + Name Retrieval.
Targets entities where house numbers are absent or unaligned using:
1. Address prefix shingle (10 chars) + name_prefix_3 with strict country blocking.
2. Postal code (len >= 5) + root_token (len >= 3) with strict country blocking.

Evaluates incrementally on top of V2 baseline against train ground truth.
"""

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import duckdb

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
EXP_DIR = REPO_ROOT / "P1" / "experiments" / "phase3" / "EXP-RET-04"
EXP_DIR.mkdir(parents=True, exist_ok=True)

con = duckdb.connect()
con.execute("SET threads=4")
con.execute("SET memory_limit='8GB'")

t0 = time.time()
print("1. Loading canonical entities for EXP-RET-04...", flush=True)

for src in ["1", "2", "3"]:
    p = REPO_ROOT / "P1" / "data" / "entities" / "train" / f"source{src}" / f"train_s{src}_entities.parquet"
    con.execute(f"""
        CREATE TEMP TABLE s{src} AS
        SELECT entity_id, country_normalized as country,
               LEFT(address_clean, 10) as addr_p10,
               name_prefix_3 as p3,
               postal_code,
               root_token
        FROM read_parquet('{p}');
    """)

print("2. Loading V2 candidates...", flush=True)
for src in ["s2", "s3"]:
    p = REPO_ROOT / "P2" / "data" / "candidates" / f"train_candidate_pairs_{src}_v2.tsv"
    con.execute(f"""
        CREATE TEMP TABLE v2_{src} AS
        SELECT source1_entity_id, matched_entity_id
        FROM read_csv('{p}', delim='\\t', header=true);
    """)

print("3. Loading Ground Truth...", flush=True)
gt_path = REPO_ROOT / "data" / "train" / "train_ground_truth.tsv"
con.execute(f"""
    CREATE TEMP TABLE gt AS
    SELECT TRIM(source1_entity_id) AS source1_entity_id,
           TRIM(UNNEST(string_split(matched_entity_ids, ','))) AS matched_id
    FROM read_csv('{gt_path}', delim='\\t', header=true)
    WHERE matched_entity_ids IS NOT NULL;

    CREATE TEMP TABLE gt_s2 AS SELECT * FROM gt WHERE matched_id LIKE 'S2-%';
    CREATE TEMP TABLE gt_s3 AS SELECT * FROM gt WHERE matched_id LIKE 'S3-%';
""")

gt_s2_tot = con.execute("SELECT COUNT(*) FROM gt_s2").fetchone()[0]
gt_s3_tot = con.execute("SELECT COUNT(*) FROM gt_s3").fetchone()[0]
gt_all_tot = gt_s2_tot + gt_s3_tot

print("4. Generating EXP-RET-04 candidates...", flush=True)
metrics = {}

for target_src, target_tbl, gt_tbl, gt_tot in [
    ("s2", "s2", "gt_s2", gt_s2_tot),
    ("s3", "s3", "gt_s3", gt_s3_tot),
]:
    t_start = time.time()
    print(f"Processing {target_src.upper()}...", flush=True)
    
    # Standalone candidates
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE ret04_{target_src}_raw AS
        -- Rule 1: Address 10-char shingle + name prefix3
        SELECT s1.entity_id AS source1_entity_id, tgt.entity_id AS matched_entity_id
        FROM s1 JOIN {target_tbl} tgt
        ON s1.country <> '' AND s1.country = tgt.country 
           AND s1.addr_p10 <> '' AND LENGTH(s1.addr_p10) >= 10 AND s1.addr_p10 = tgt.addr_p10
           AND s1.p3 <> '' AND LENGTH(s1.p3) = 3 AND s1.p3 = tgt.p3
           
        UNION
        
        -- Rule 2: Postal code + root token
        SELECT s1.entity_id AS source1_entity_id, tgt.entity_id AS matched_entity_id
        FROM s1 JOIN {target_tbl} tgt
        ON s1.country <> '' AND s1.country = tgt.country 
           AND s1.postal_code <> '' AND LENGTH(s1.postal_code) >= 5 AND s1.postal_code = tgt.postal_code
           AND s1.root_token <> '' AND LENGTH(s1.root_token) >= 3 AND s1.root_token = tgt.root_token;
    """)
    raw_cnt = con.execute(f"SELECT COUNT(*) FROM ret04_{target_src}_raw").fetchone()[0]
    
    out_file = EXP_DIR / f"train_candidates_{target_src}_exp_ret_04.tsv"
    con.execute(f"""
        COPY (SELECT DISTINCT source1_entity_id, matched_entity_id FROM ret04_{target_src}_raw)
        TO '{out_file}' WITH (FORMAT CSV, DELIMITER '\\t', HEADER)
    """)
    
    # Union with V2
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE comb_{target_src} AS
        SELECT source1_entity_id, matched_entity_id FROM v2_{target_src}
        UNION
        SELECT source1_entity_id, matched_entity_id FROM ret04_{target_src}_raw;
    """)
    
    tot_comb = con.execute(f"SELECT COUNT(*) FROM comb_{target_src}").fetchone()[0]
    cap = con.execute(f"""
        SELECT COUNT(DISTINCT (c.source1_entity_id, c.matched_entity_id))
        FROM comb_{target_src} c
        JOIN {gt_tbl} gt ON c.source1_entity_id = gt.source1_entity_id AND c.matched_entity_id = gt.matched_id
    """).fetchone()[0]
    
    new_cap = con.execute(f"""
        SELECT COUNT(DISTINCT (c.source1_entity_id, c.matched_entity_id))
        FROM ret04_{target_src}_raw c
        JOIN {gt_tbl} gt ON c.source1_entity_id = gt.source1_entity_id AND c.matched_entity_id = gt.matched_id
        ANTI JOIN v2_{target_src} v2 ON c.source1_entity_id = v2.source1_entity_id AND c.matched_entity_id = v2.matched_entity_id
    """).fetchone()[0]
    
    rec = cap / gt_tot
    print(f"{target_src.upper()} Total Cands: {tot_comb:,} | Captured: {cap:,} (+{new_cap:,} new) | Recall: {rec:.6%} ({time.time()-t_start:.1f}s)", flush=True)
    
    metrics[target_src] = {
        "raw_candidates": raw_cnt,
        "combined_candidates": tot_comb,
        "captured": cap,
        "new_captured": new_cap,
        "gt_total": gt_tot,
        "recall": rec,
        "artifact": str(out_file),
    }

# Combined S2+S3
con.execute("""
    CREATE OR REPLACE TEMP TABLE comb_all AS
    SELECT source1_entity_id, matched_entity_id FROM comb_s2
    UNION ALL
    SELECT source1_entity_id, matched_entity_id FROM comb_s3;
""")
tot_all = con.execute("SELECT COUNT(*) FROM comb_all").fetchone()[0]
cap_all = con.execute("""
    SELECT COUNT(DISTINCT (c.source1_entity_id, c.matched_entity_id))
    FROM comb_all c
    JOIN gt ON c.source1_entity_id = gt.source1_entity_id AND c.matched_entity_id = gt.matched_id
""").fetchone()[0]
rec_all = cap_all / gt_all_tot
missed_all = gt_all_tot - cap_all

# Fanout
fanout = con.execute("""
    WITH s1_counts AS (
        SELECT s1.entity_id, COUNT(c.matched_entity_id) AS cnt
        FROM (SELECT entity_id FROM read_parquet('P1/data/entities/train/source1/train_s1_entities.parquet')) s1
        LEFT JOIN comb_all c ON s1.entity_id = c.source1_entity_id
        GROUP BY s1.entity_id
    )
    SELECT 
        COUNT(CASE WHEN cnt = 0 THEN 1 END) AS zero_cands,
        quantile_cont(cnt, 0.50) AS p50,
        quantile_cont(cnt, 0.90) AS p90,
        quantile_cont(cnt, 0.95) AS p95,
        quantile_cont(cnt, 0.99) AS p99,
        quantile_cont(cnt, 0.999) AS p999,
        MAX(cnt) AS max_cnt
    FROM s1_counts
""").fetchone()

summary = {
    "experiment": "EXP-RET-04",
    "description": "Locality + Name Retrieval (Address 10-char shingle + name_prefix_3 + Postal Code / Root Token)",
    "v2_baseline_recall": 0.657623,
    "combined_recall": rec_all,
    "recall_delta": rec_all - 0.657623,
    "total_candidates": tot_all,
    "candidate_delta": tot_all - 67332524,
    "captured_pairs": cap_all,
    "new_captured_pairs": cap_all - 5023168,
    "missed_pairs": missed_all,
    "zero_candidate_s1": fanout[0],
    "zero_candidate_delta": fanout[0] - 94043,
    "fanout_p50": float(fanout[1]),
    "fanout_p90": float(fanout[2]),
    "fanout_p95": float(fanout[3]),
    "fanout_p99": float(fanout[4]),
    "fanout_p999": float(fanout[5]),
    "fanout_max": int(fanout[6]),
    "fanout_constraint_p95_le_300": "PASS" if fanout[3] <= 300 else "FAIL",
    "status": "ACCEPTED" if (rec_all > 0.657623 and fanout[3] <= 300) else "REJECTED",
    "s2_metrics": metrics["s2"],
    "s3_metrics": metrics["s3"],
    "total_runtime_seconds": round(time.time() - t0, 2),
    "timestamp": datetime.now(timezone.utc).isoformat(),
}

with open(EXP_DIR / "metrics.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\n=== EXP-RET-04 Summary ===")
print(f"Combined Recall: {rec_all:.6%} (Delta: {summary['recall_delta']:+.6%})")
print(f"Total Candidates: {tot_all:,} (Delta: {summary['candidate_delta']:+,})")
print(f"Captured Pairs: {cap_all:,} (New True Captures: {summary['new_captured_pairs']:+,})")
print(f"Zero-candidate S1: {fanout[0]:,} (Delta: {summary['zero_candidate_delta']:+,})")
print(f"Fanout: p50={fanout[1]} | p95={fanout[2]} | p99={fanout[3]} | max={fanout[6]}")
print(f"Decision: {summary['status']}")
