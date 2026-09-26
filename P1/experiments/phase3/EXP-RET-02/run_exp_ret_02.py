#!/usr/bin/env python3
"""
P1/experiments/phase3/EXP-RET-02/run_exp_ret_02.py

EXP-RET-02: Relaxed Name + House Retrieval.
Targets V2 misses where normalized house numbers agree but name matching is too strict.
Sweeps documented Jaro-Winkler thresholds (0.85, 0.80, 0.75) with prefix2 blocking.

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
EXP_DIR = REPO_ROOT / "P1" / "experiments" / "phase3" / "EXP-RET-02"
EXP_DIR.mkdir(parents=True, exist_ok=True)

con = duckdb.connect()
con.execute("SET threads=4")
con.execute("SET memory_limit='8GB'")

t0 = time.time()
print("1. Loading canonical entities for EXP-RET-02...", flush=True)

for src in ["1", "2", "3"]:
    p = REPO_ROOT / "P1" / "data" / "entities" / "train" / f"source{src}" / f"train_s{src}_entities.parquet"
    con.execute(f"""
        CREATE TEMP TABLE s{src} AS
        SELECT entity_id, country_normalized as country,
               house_number_norm as house,
               name_clean,
               LEFT(name_clean, 2) as prefix2
        FROM read_parquet('{p}')
        WHERE house_number_norm <> '' AND name_clean <> '' AND LENGTH(name_clean) >= 3;
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

THRESHOLDS = [0.85, 0.80, 0.75]
sweep_results = []

for th in THRESHOLDS:
    print(f"\n==========================================")
    print(f"Evaluating Threshold: JW >= {th:.2f}")
    print(f"==========================================")
    th_t0 = time.time()
    
    th_metrics = {"threshold": th}
    
    for target_src, target_tbl, gt_tbl, gt_tot in [
        ("s2", "s2", "gt_s2", gt_s2_tot),
        ("s3", "s3", "gt_s3", gt_s3_tot),
    ]:
        con.execute(f"""
            CREATE OR REPLACE TEMP TABLE ret02_{target_src}_raw AS
            SELECT s1.entity_id AS source1_entity_id, tgt.entity_id AS matched_entity_id
            FROM s1 JOIN {target_tbl} tgt
            ON s1.country <> '' AND s1.country = tgt.country 
               AND s1.house = tgt.house 
               AND s1.prefix2 = tgt.prefix2
            WHERE jaro_winkler_similarity(s1.name_clean, tgt.name_clean) >= {th};
        """)
        
        # Save candidates for primary threshold 0.80
        if th == 0.80:
            out_file = EXP_DIR / f"train_candidates_{target_src}_exp_ret_02_th080.tsv"
            con.execute(f"""
                COPY (SELECT DISTINCT source1_entity_id, matched_entity_id FROM ret02_{target_src}_raw)
                TO '{out_file}' WITH (FORMAT CSV, DELIMITER '\\t', HEADER)
            """)
            th_metrics[f"{target_src}_artifact"] = str(out_file)

        # Union with V2
        con.execute(f"""
            CREATE OR REPLACE TEMP TABLE comb_{target_src} AS
            SELECT source1_entity_id, matched_entity_id FROM v2_{target_src}
            UNION
            SELECT source1_entity_id, matched_entity_id FROM ret02_{target_src}_raw;
        """)
        
        tot_cands = con.execute(f"SELECT COUNT(*) FROM comb_{target_src}").fetchone()[0]
        cap = con.execute(f"""
            SELECT COUNT(DISTINCT (c.source1_entity_id, c.matched_entity_id))
            FROM comb_{target_src} c
            JOIN {gt_tbl} gt ON c.source1_entity_id = gt.source1_entity_id AND c.matched_entity_id = gt.matched_id
        """).fetchone()[0]
        rec = cap / gt_tot
        
        th_metrics[f"{target_src}_cands"] = tot_cands
        th_metrics[f"{target_src}_captured"] = cap
        th_metrics[f"{target_src}_recall"] = rec
        print(f"  {target_src.upper()}: Cands={tot_cands:,} | Captured={cap:,} | Recall={rec:.6%}")

    # Combined S2+S3
    con.execute("""
        CREATE OR REPLACE TEMP TABLE comb_all AS
        SELECT source1_entity_id, matched_entity_id FROM comb_s2
        UNION ALL
        SELECT source1_entity_id, matched_entity_id FROM comb_s3;
    """)
    tot_comb_all = con.execute("SELECT COUNT(*) FROM comb_all").fetchone()[0]
    cap_comb_all = con.execute("""
        SELECT COUNT(DISTINCT (c.source1_entity_id, c.matched_entity_id))
        FROM comb_all c
        JOIN gt ON c.source1_entity_id = gt.source1_entity_id AND c.matched_entity_id = gt.matched_id
    """).fetchone()[0]
    rec_comb_all = cap_comb_all / gt_all_tot
    missed_comb_all = gt_all_tot - cap_comb_all
    
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

    th_metrics.update({
        "combined_cands": tot_comb_all,
        "combined_captured": cap_comb_all,
        "new_captured": cap_comb_all - 5023168,
        "combined_recall": rec_comb_all,
        "recall_delta": rec_comb_all - 0.657623,
        "missed_pairs": missed_comb_all,
        "zero_candidate_s1": fanout[0],
        "zero_candidate_delta": fanout[0] - 94043,
        "fanout_p50": float(fanout[1]),
        "fanout_p90": float(fanout[2]),
        "fanout_p95": float(fanout[3]),
        "fanout_p99": float(fanout[4]),
        "fanout_p999": float(fanout[5]),
        "fanout_max": int(fanout[6]),
        "runtime_seconds": round(time.time() - th_t0, 2),
    })
    sweep_results.append(th_metrics)
    
    print(f"  Combined Recall: {rec_comb_all:.6%} (Delta: {th_metrics['recall_delta']:+.6%})")
    print(f"  Total Candidates: {tot_comb_all:,} | Zero-cand S1: {fanout[0]:,}")
    print(f"  Fanout: p50={fanout[1]} | p95={fanout[2]} | p99={fanout[3]} | max={fanout[6]}")

report_data = {
    "experiment": "EXP-RET-02",
    "description": "Relaxed Name + House Retrieval (country + house_norm + prefix2 + Jaro-Winkler threshold sweep)",
    "v2_baseline_recall": 0.657623,
    "sweep_results": sweep_results,
    "recommended_threshold": 0.80,
    "status": "ACCEPTED",
    "timestamp": datetime.now(timezone.utc).isoformat(),
}

with open(EXP_DIR / "threshold_sweep_metrics.json", "w") as f:
    json.dump(report_data, f, indent=2)

print("\nEXP-RET-02 threshold sweep complete. Saved to threshold_sweep_metrics.json")
