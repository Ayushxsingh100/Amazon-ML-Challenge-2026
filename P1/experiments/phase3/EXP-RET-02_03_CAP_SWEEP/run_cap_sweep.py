#!/usr/bin/env python3
"""
P1/experiments/phase3/EXP-RET-02_03_CAP_SWEEP/run_cap_sweep.py

Controlled experiment sweep testing per-S1 fanout pruning on EXP-RET-02 and EXP-RET-03:
- Config A: V2 + EXP-01 + EXP-04 (Reference control)
- Config B: V2 + EXP-01 + EXP-02 (cap=100) + EXP-04
- Config C: V2 + EXP-01 + EXP-03 (cap=100) + EXP-04
- Config D: V2 + EXP-01 + EXP-02 (cap=50) + EXP-03 (cap=50) + EXP-04
- Config E: V2 + EXP-01 + EXP-02 (cap=30) + EXP-03 (cap=30) + EXP-04
- Config F: V2 + EXP-01 + EXP-02 (cap=25) + EXP-03 (cap=25) + EXP-04
- Config G: V2 + EXP-01 + EXP-02 (cap=20) + EXP-03 (cap=20) + EXP-04

Memory-optimized two-phase execution (Train phase, then Test phase).
Uses disjoint target aggregation (S2 and S3 are disjoint target ID namespaces),
eliminating massive cross-source unions and disk spill.
"""

import os
import sys
import time
import json
import hashlib
from pathlib import Path
import duckdb

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
EXP_DIR = REPO_ROOT / "P1" / "experiments" / "phase3" / "EXP-RET-02_03_CAP_SWEEP"
EXP_DIR.mkdir(parents=True, exist_ok=True)
CANDIDATE_OUT_DIR = EXP_DIR / "candidates"
CANDIDATE_OUT_DIR.mkdir(parents=True, exist_ok=True)

COMMON_PREFIXES = "('the', 'shri', 'sri', 'dr', 'm/s', 'hotel', 'new', 'om', 'sai', 'jai', 'a', 'an')"

DEVANAGARI_MAP = {
    'अ': 'a', 'आ': 'a', 'इ': 'i', 'ई': 'i', 'उ': 'u', 'ऊ': 'u',
    'ऋ': 'r', 'ए': 'e', 'ऐ': 'ai', 'ओ': 'o', 'औ': 'au',
    'क': 'k', 'ख': 'kh', 'ग': 'g', 'घ': 'gh', 'ङ': 'ng',
    'च': 'ch', 'छ': 'chh', 'ज': 'j', 'झ': 'jh', 'ञ': 'ny',
    'ट': 't', 'ठ': 'th', 'ड': 'd', 'ढ': 'dh', 'ण': 'n',
    'त': 't', 'थ': 'th', 'द': 'd', 'ध': 'dh', 'न': 'n',
    'प': 'p', 'फ': 'ph', 'ब': 'b', 'भ': 'bh', 'म': 'm',
    'य': 'y', 'र': 'r', 'ल': 'l', 'व': 'v', 'श': 'sh',
    'ष': 'sh', 'स': 's', 'ह': 'h',
    'ा': 'a', 'ि': 'i', 'ी': 'i', 'ु': 'u', 'ू': 'u',
    'ृ': 'r', 'े': 'e', 'ै': 'ai', 'ो': 'o', 'ौ': 'au',
    '्': '', 'ं': 'n', 'ँ': 'n', 'ः': 'h',
    '०': '0', '१': '1', '२': '2', '३': '3', '४': '4',
    '५': '5', '६': '6', '७': '7', '८': '8', '९': '9'
}

def transliterate_text(text: str) -> str:
    if not text:
        return ""
    res = []
    for char in text:
        res.append(DEVANAGARI_MAP.get(char, char))
    return "".join(res)

def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

CONFIG_DEFS = [
    {"name": "EXPERIMENT A", "desc": "V2 + EXP-01 + EXP-04 (Control)", "cap_e2": None, "cap_e3": None},
    {"name": "EXPERIMENT B", "desc": "V2 + EXP-01 + EXP-02 (cap=100) + EXP-04", "cap_e2": 100, "cap_e3": None},
    {"name": "EXPERIMENT C", "desc": "V2 + EXP-01 + EXP-03 (cap=100) + EXP-04", "cap_e2": None, "cap_e3": 100},
    {"name": "EXPERIMENT D", "desc": "EXP-02 cap=50, EXP-03 cap=50", "cap_e2": 50, "cap_e3": 50},
    {"name": "EXPERIMENT E", "desc": "EXP-02 cap=30, EXP-03 cap=30", "cap_e2": 30, "cap_e3": 30},
    {"name": "EXPERIMENT F", "desc": "EXP-02 cap=25, EXP-03 cap=25", "cap_e2": 25, "cap_e3": 25},
    {"name": "EXPERIMENT G", "desc": "EXP-02 cap=20, EXP-03 cap=20", "cap_e2": 20, "cap_e3": 20},
]

def run_train_phase():
    """Execute TRAIN phase and return metrics for all configurations."""
    print("\n=======================================================")
    print("PHASE 1: TRAIN EVALUATION")
    print("=======================================================")
    con = duckdb.connect()
    con.execute("PRAGMA threads=8;")
    con.execute("PRAGMA memory_limit='10GB';")

    gt_path = REPO_ROOT / "data" / "train" / "train_ground_truth.tsv"
    con.execute(f"""
        CREATE TABLE gt AS
        SELECT TRIM(source1_entity_id) AS s1_id, TRIM(UNNEST(string_split(matched_entity_ids, ','))) AS matched_id
        FROM read_csv_auto('{gt_path}', delim='\t', header=true)
        WHERE matched_entity_ids IS NOT NULL;
    """)
    gt_s2_tot = con.execute("SELECT count(*) FROM gt WHERE matched_id LIKE 'S2-%'").fetchone()[0]
    gt_s3_tot = con.execute("SELECT count(*) FROM gt WHERE matched_id LIKE 'S3-%'").fetchone()[0]
    gt_tot = gt_s2_tot + gt_s3_tot
    print(f"Ground Truth loaded: S2={gt_s2_tot:,}, S3={gt_s3_tot:,}, Combined={gt_tot:,}")

    train_s1_path = REPO_ROOT / "P1/data/entities/train/source1/train_s1_entities.parquet"
    con.execute(f"CREATE TABLE train_s1 AS SELECT entity_id FROM read_parquet('{train_s1_path}');")
    n_s1 = con.execute("SELECT count(*) FROM train_s1").fetchone()[0]
    print(f"Train S1 Entities: {n_s1:,}")

    # Load component tables
    for target in ['s2', 's3']:
        v2_path = REPO_ROOT / f"P2/data/candidates/train_candidate_pairs_{target}_v2.tsv"
        exp1_path = REPO_ROOT / f"P1/experiments/phase3/EXP-RET-01/train_candidates_{target}_exp_ret_01.tsv"
        exp2_path = REPO_ROOT / f"P1/experiments/phase3/EXP-RET-02/train_candidates_{target}_exp_ret_02_th080.tsv"
        exp3_path = REPO_ROOT / f"P1/experiments/phase3/EXP-RET-03/train_candidates_{target}_exp_ret_03.tsv"
        exp4_path = REPO_ROOT / f"P1/experiments/phase3/EXP-RET-04/train_candidates_{target}_exp_ret_04.tsv"

        con.execute(f"CREATE TABLE tr_v2_{target} AS SELECT source1_entity_id, matched_entity_id FROM read_csv_auto('{v2_path}', delim='\t', header=true);")
        
        # EXP-01 capped at 100
        con.execute(f"""
            CREATE TEMP TABLE tr_e1_raw_{target} AS SELECT source1_entity_id, matched_entity_id FROM read_csv_auto('{exp1_path}', delim='\t', header=true);
            CREATE TABLE tr_e1_{target} AS
            SELECT r.source1_entity_id, r.matched_entity_id
            FROM tr_e1_raw_{target} r
            JOIN (SELECT source1_entity_id FROM tr_e1_raw_{target} GROUP BY source1_entity_id HAVING count(*) <= 100) k
              ON r.source1_entity_id = k.source1_entity_id;
            DROP TABLE tr_e1_raw_{target};
        """)

        con.execute(f"CREATE TABLE tr_e4_{target} AS SELECT source1_entity_id, matched_entity_id FROM read_csv_auto('{exp4_path}', delim='\t', header=true);")
        con.execute(f"CREATE TABLE tr_e2_raw_{target} AS SELECT source1_entity_id, matched_entity_id FROM read_csv_auto('{exp2_path}', delim='\t', header=true);")
        con.execute(f"CREATE TABLE tr_e3_raw_{target} AS SELECT source1_entity_id, matched_entity_id FROM read_csv_auto('{exp3_path}', delim='\t', header=true);")
        print(f"Loaded train components for {target.upper()}")

    train_results = {}
    for cfg in CONFIG_DEFS:
        t0 = time.time()
        c_name = cfg["name"]
        cap2 = cfg["cap_e2"]
        cap3 = cfg["cap_e3"]
        print(f"Evaluating {c_name} on TRAIN (cap2={cap2}, cap3={cap3})...")

        for target in ['s2', 's3']:
            tbls = [f"tr_v2_{target}", f"tr_e1_{target}", f"tr_e4_{target}"]
            if cap2 is not None:
                con.execute(f"""
                    CREATE OR REPLACE TEMP TABLE tr_e2_c_{target} AS
                    SELECT r.source1_entity_id, r.matched_entity_id
                    FROM tr_e2_raw_{target} r
                    JOIN (SELECT source1_entity_id FROM tr_e2_raw_{target} GROUP BY source1_entity_id HAVING count(*) <= {cap2}) k
                      ON r.source1_entity_id = k.source1_entity_id;
                """)
                tbls.append(f"tr_e2_c_{target}")

            if cap3 is not None:
                con.execute(f"""
                    CREATE OR REPLACE TEMP TABLE tr_e3_c_{target} AS
                    SELECT r.source1_entity_id, r.matched_entity_id
                    FROM tr_e3_raw_{target} r
                    JOIN (SELECT source1_entity_id FROM tr_e3_raw_{target} GROUP BY source1_entity_id HAVING count(*) <= {cap3}) k
                      ON r.source1_entity_id = k.source1_entity_id;
                """)
                tbls.append(f"tr_e3_c_{target}")

            union_sql = " UNION ALL ".join([f"SELECT source1_entity_id, matched_entity_id FROM {t}" for t in tbls])
            con.execute(f"""
                CREATE OR REPLACE TEMP TABLE tr_curr_{target} AS
                SELECT DISTINCT source1_entity_id, matched_entity_id FROM ({union_sql});
            """)

        # Calculate metrics for S2 and S3
        cands_s2 = con.execute("SELECT count(*) FROM tr_curr_s2").fetchone()[0]
        cands_s3 = con.execute("SELECT count(*) FROM tr_curr_s3").fetchone()[0]
        tot_cands = cands_s2 + cands_s3

        cap_s2 = con.execute("SELECT count(*) FROM tr_curr_s2 c JOIN gt ON c.source1_entity_id = gt.s1_id AND c.matched_entity_id = gt.matched_id;").fetchone()[0]
        cap_s3 = con.execute("SELECT count(*) FROM tr_curr_s3 c JOIN gt ON c.source1_entity_id = gt.s1_id AND c.matched_entity_id = gt.matched_id;").fetchone()[0]
        cap_comb = cap_s2 + cap_s3

        rec_s2 = cap_s2 / gt_s2_tot
        rec_s3 = cap_s3 / gt_s3_tot
        rec_comb = cap_comb / gt_tot
        missed_comb = gt_tot - cap_comb

        # Fanout: compute per-source counts, then join on train_s1
        con.execute("""
            CREATE OR REPLACE TEMP TABLE tr_f2 AS SELECT source1_entity_id, count(*) as cnt FROM tr_curr_s2 GROUP BY source1_entity_id;
            CREATE OR REPLACE TEMP TABLE tr_f3 AS SELECT source1_entity_id, count(*) as cnt FROM tr_curr_s3 GROUP BY source1_entity_id;
        """)
        stats = con.execute("""
            SELECT 
                count(CASE WHEN cnt = 0 THEN 1 END),
                quantile_cont(cnt, 0.50),
                quantile_cont(cnt, 0.90),
                quantile_cont(cnt, 0.95),
                quantile_cont(cnt, 0.99),
                quantile_cont(cnt, 0.999),
                max(cnt)
            FROM (
                SELECT s.entity_id, COALESCE(f2.cnt, 0) + COALESCE(f3.cnt, 0) as cnt
                FROM train_s1 s
                LEFT JOIN tr_f2 f2 ON s.entity_id = f2.source1_entity_id
                LEFT JOIN tr_f3 f3 ON s.entity_id = f3.source1_entity_id
            );
        """).fetchone()

        train_results[c_name] = {
            "s2_candidates": cands_s2,
            "s3_candidates": cands_s3,
            "total_candidates": tot_cands,
            "s2_captured": cap_s2,
            "s3_captured": cap_s3,
            "combined_captured": cap_comb,
            "s2_recall": rec_s2,
            "s3_recall": rec_s3,
            "combined_recall": rec_comb,
            "missed_pairs": missed_comb,
            "zero_s1": stats[0],
            "p50": stats[1],
            "p90": stats[2],
            "p95": stats[3],
            "p99": stats[4],
            "p999": stats[5],
            "max": stats[6],
            "elapsed_s": round(time.time() - t0, 2)
        }
        print(f"  -> Train Recall: {rec_comb:.6%} ({cap_comb:,}/{gt_tot:,}) | Train p95: {stats[3]:.1f} | Zero-S1: {stats[0]:,} | ({time.time()-t0:.2f}s)")

    con.close()
    return train_results


def run_test_phase():
    """Execute TEST phase and return metrics for all configurations."""
    print("\n=======================================================")
    print("PHASE 2: TEST EVALUATION")
    print("=======================================================")
    con = duckdb.connect()
    con.execute("PRAGMA threads=8;")
    con.execute("PRAGMA memory_limit='10GB';")
    con.create_function("transliterate_devanagari", transliterate_text, [str], str)

    test_s1_path = REPO_ROOT / "P1/data/entities/test/source1/test_s1_entities.parquet"
    con.execute(f"""
        CREATE TABLE test_s1 AS
        SELECT 
            entity_id,
            country_raw AS country,
            business_name_raw AS name_raw,
            business_address_raw AS addr_raw,
            regexp_replace(lower(trim(business_name_raw)), '[^a-z0-9]', '', 'g') AS clean_name,
            regexp_replace(
                regexp_replace(lower(trim(business_name_raw)), '[^a-z0-9]', '', 'g'),
                '(privatelimited|pvtlimited|pvtltd|pvtld|limited|ltd|llc|inc|corp|corporation|enterprises?|services?)$',
                ''
            ) AS core_name,
            LEFT(regexp_replace(lower(trim(business_name_raw)), '[^a-z0-9]', '', 'g'), 2) AS name_prefix_2,
            LEFT(regexp_replace(lower(trim(business_name_raw)), '[^a-z0-9]', '', 'g'), 3) AS name_prefix_3,
            regexp_replace(regexp_extract(trim(business_address_raw), '[0-9]+[A-Za-z]?', 0), '^0+', '') AS house_norm,
            regexp_replace(lower(trim(business_address_raw)), '[^a-z0-9]', '', 'g') AS addr_clean,
            regexp_extract(trim(business_address_raw), '(?:^|[^0-9])([0-9]{{5,6}})(?:[^0-9]|$)', 1) AS postal_code,
            CASE 
                WHEN lower(split_part(trim(business_name_raw), ' ', 1)) IN {COMMON_PREFIXES} 
                     AND split_part(trim(business_name_raw), ' ', 2) <> '' 
                THEN split_part(trim(business_name_raw), ' ', 2)
                ELSE split_part(trim(business_name_raw), ' ', 1)
            END AS root_token,
            LEFT(regexp_replace(lower(trim(transliterate_devanagari(business_name_raw))), '[^a-z0-9]', '', 'g'), 2) AS translit_prefix_2,
            regexp_replace(lower(trim(transliterate_devanagari(business_name_raw))), '[^a-z0-9]', '', 'g') AS translit_name
        FROM read_parquet('{test_s1_path}');
    """)
    n_s1 = con.execute("SELECT count(*) FROM test_s1").fetchone()[0]
    print(f"Loaded Test S1 Entities: {n_s1:,}")

    # Generate test components
    for target in ['s2', 's3']:
        tgt_parquet = REPO_ROOT / f"P1/data/entities/test/{'source2' if target == 's2' else 'source3'}/test_{target}_entities.parquet"
        con.execute(f"""
            CREATE TABLE test_{target} AS
            SELECT 
                entity_id,
                country_raw AS country,
                business_name_raw AS name_raw,
                business_address_raw AS addr_raw,
                regexp_replace(lower(trim(business_name_raw)), '[^a-z0-9]', '', 'g') AS clean_name,
                regexp_replace(
                    regexp_replace(lower(trim(business_name_raw)), '[^a-z0-9]', '', 'g'),
                    '(privatelimited|pvtlimited|pvtltd|pvtld|limited|ltd|llc|inc|corp|corporation|enterprises?|services?)$',
                    ''
                ) AS core_name,
                LEFT(regexp_replace(lower(trim(business_name_raw)), '[^a-z0-9]', '', 'g'), 2) AS name_prefix_2,
                LEFT(regexp_replace(lower(trim(business_name_raw)), '[^a-z0-9]', '', 'g'), 3) AS name_prefix_3,
                regexp_replace(regexp_extract(trim(business_address_raw), '[0-9]+[A-Za-z]?', 0), '^0+', '') AS house_norm,
                regexp_replace(lower(trim(business_address_raw)), '[^a-z0-9]', '', 'g') AS addr_clean,
                regexp_extract(trim(business_address_raw), '(?:^|[^0-9])([0-9]{{5,6}})(?:[^0-9]|$)', 1) AS postal_code,
                CASE 
                    WHEN lower(split_part(trim(business_name_raw), ' ', 1)) IN {COMMON_PREFIXES} 
                         AND split_part(trim(business_name_raw), ' ', 2) <> '' 
                    THEN split_part(trim(business_name_raw), ' ', 2)
                    ELSE split_part(trim(business_name_raw), ' ', 1)
                END AS root_token,
                LEFT(regexp_replace(lower(trim(transliterate_devanagari(business_name_raw))), '[^a-z0-9]', '', 'g'), 2) AS translit_prefix_2,
                regexp_replace(lower(trim(transliterate_devanagari(business_name_raw))), '[^a-z0-9]', '', 'g') AS translit_name
            FROM read_parquet('{tgt_parquet}');
        """)

        # V2 test
        v2_path = REPO_ROOT / f"P2/data/candidates/test_candidate_pairs_{target}_v2.tsv"
        con.execute(f"CREATE TABLE te_v2_{target} AS SELECT source1_entity_id, matched_entity_id FROM read_csv_auto('{v2_path}', delim='\t', header=True);")

        # EXP-01 test (cap <= 100)
        con.execute(f"""
            CREATE TEMP TABLE te_e1_raw_{target} AS
            SELECT s1.entity_id AS source1_entity_id, tgt.entity_id AS matched_entity_id
            FROM test_s1 s1 JOIN test_{target} tgt ON s1.country = tgt.country AND s1.core_name = tgt.core_name
            WHERE LENGTH(s1.core_name) >= 5;

            CREATE TABLE te_e1_{target} AS
            SELECT r.source1_entity_id, r.matched_entity_id
            FROM te_e1_raw_{target} r
            JOIN (SELECT source1_entity_id FROM te_e1_raw_{target} GROUP BY source1_entity_id HAVING count(*) <= 100) k
              ON r.source1_entity_id = k.source1_entity_id;
            DROP TABLE te_e1_raw_{target};
        """)

        # EXP-04 test
        con.execute(f"""
            CREATE TABLE te_e4_{target} AS
            SELECT DISTINCT source1_entity_id, matched_entity_id FROM (
                SELECT s1.entity_id AS source1_entity_id, tgt.entity_id AS matched_entity_id
                FROM test_s1 s1 JOIN test_{target} tgt
                  ON s1.country = tgt.country AND LEFT(s1.addr_clean, 10) = LEFT(tgt.addr_clean, 10) AND s1.name_prefix_3 = tgt.name_prefix_3
                WHERE LENGTH(s1.addr_clean) >= 10 AND LENGTH(tgt.addr_clean) >= 10 AND LENGTH(s1.name_prefix_3) = 3
                UNION ALL
                SELECT s1.entity_id AS source1_entity_id, tgt.entity_id AS matched_entity_id
                FROM test_s1 s1 JOIN test_{target} tgt
                  ON s1.country = tgt.country AND s1.postal_code = tgt.postal_code AND s1.root_token = tgt.root_token
                WHERE s1.postal_code IS NOT NULL AND LENGTH(s1.postal_code) >= 5 AND s1.root_token IS NOT NULL AND LENGTH(s1.root_token) >= 4
            );
        """)

        # EXP-02 raw test
        con.execute(f"""
            CREATE TABLE te_e2_raw_{target} AS
            SELECT s1.entity_id AS source1_entity_id, tgt.entity_id AS matched_entity_id
            FROM test_s1 s1 JOIN test_{target} tgt
              ON s1.country = tgt.country AND s1.house_norm = tgt.house_norm AND s1.name_prefix_2 = tgt.name_prefix_2
            WHERE s1.house_norm IS NOT NULL AND s1.house_norm <> ''
              AND LENGTH(s1.clean_name) >= 2 AND LENGTH(tgt.clean_name) >= 2
              AND jaro_winkler_similarity(s1.clean_name, tgt.clean_name) >= 0.80;
        """)

        # EXP-03 raw test
        con.execute(f"""
            CREATE TABLE te_e3_raw_{target} AS
            SELECT DISTINCT source1_entity_id, matched_entity_id FROM (
                SELECT s1.entity_id AS source1_entity_id, tgt.entity_id AS matched_entity_id
                FROM test_s1 s1 JOIN test_{target} tgt
                  ON s1.country = tgt.country AND s1.house_norm = tgt.house_norm AND s1.translit_prefix_2 = tgt.translit_prefix_2
                WHERE s1.house_norm IS NOT NULL AND s1.house_norm <> ''
                  AND LENGTH(s1.translit_name) >= 2 AND LENGTH(tgt.translit_name) >= 2
                  AND jaro_winkler_similarity(s1.translit_name, tgt.translit_name) >= 0.80
                UNION ALL
                SELECT s1.entity_id AS source1_entity_id, tgt.entity_id AS matched_entity_id
                FROM test_s1 s1 JOIN test_{target} tgt
                  ON s1.country = tgt.country AND s1.house_norm = tgt.house_norm AND s1.postal_code = tgt.postal_code
                WHERE s1.country = 'India' AND s1.house_norm IS NOT NULL AND s1.house_norm <> ''
                  AND s1.postal_code IS NOT NULL AND LENGTH(s1.postal_code) >= 5
            );
        """)
        print(f"Generated test components for {target.upper()}")

    test_results = {}
    for cfg in CONFIG_DEFS:
        t0 = time.time()
        c_name = cfg["name"]
        cap2 = cfg["cap_e2"]
        cap3 = cfg["cap_e3"]
        print(f"Evaluating {c_name} on TEST (cap2={cap2}, cap3={cap3})...")

        for target in ['s2', 's3']:
            tbls = [f"te_v2_{target}", f"te_e1_{target}", f"te_e4_{target}"]
            if cap2 is not None:
                con.execute(f"""
                    CREATE OR REPLACE TEMP TABLE te_e2_c_{target} AS
                    SELECT r.source1_entity_id, r.matched_entity_id
                    FROM te_e2_raw_{target} r
                    JOIN (SELECT source1_entity_id FROM te_e2_raw_{target} GROUP BY source1_entity_id HAVING count(*) <= {cap2}) k
                      ON r.source1_entity_id = k.source1_entity_id;
                """)
                tbls.append(f"te_e2_c_{target}")

            if cap3 is not None:
                con.execute(f"""
                    CREATE OR REPLACE TEMP TABLE te_e3_c_{target} AS
                    SELECT r.source1_entity_id, r.matched_entity_id
                    FROM te_e3_raw_{target} r
                    JOIN (SELECT source1_entity_id FROM te_e3_raw_{target} GROUP BY source1_entity_id HAVING count(*) <= {cap3}) k
                      ON r.source1_entity_id = k.source1_entity_id;
                """)
                tbls.append(f"te_e3_c_{target}")

            union_sql = " UNION ALL ".join([f"SELECT source1_entity_id, matched_entity_id FROM {t}" for t in tbls])
            con.execute(f"""
                CREATE OR REPLACE TEMP TABLE te_curr_{target} AS
                SELECT DISTINCT source1_entity_id, matched_entity_id FROM ({union_sql});
            """)

        cands_s2 = con.execute("SELECT count(*) FROM te_curr_s2").fetchone()[0]
        cands_s3 = con.execute("SELECT count(*) FROM te_curr_s3").fetchone()[0]
        tot_cands = cands_s2 + cands_s3

        # Test fanout
        con.execute("""
            CREATE OR REPLACE TEMP TABLE te_f2 AS SELECT source1_entity_id, count(*) as cnt FROM te_curr_s2 GROUP BY source1_entity_id;
            CREATE OR REPLACE TEMP TABLE te_f3 AS SELECT source1_entity_id, count(*) as cnt FROM te_curr_s3 GROUP BY source1_entity_id;
        """)
        stats = con.execute("""
            SELECT 
                count(CASE WHEN cnt = 0 THEN 1 END),
                quantile_cont(cnt, 0.50),
                quantile_cont(cnt, 0.90),
                quantile_cont(cnt, 0.95),
                quantile_cont(cnt, 0.99),
                quantile_cont(cnt, 0.999),
                max(cnt),
                count(CASE WHEN cnt >= 300 THEN 1 END),
                count(CASE WHEN cnt >= 500 THEN 1 END),
                count(CASE WHEN cnt >= 1000 THEN 1 END)
            FROM (
                SELECT s.entity_id, COALESCE(f2.cnt, 0) + COALESCE(f3.cnt, 0) as cnt
                FROM test_s1 s
                LEFT JOIN te_f2 f2 ON s.entity_id = f2.source1_entity_id
                LEFT JOIN te_f3 f3 ON s.entity_id = f3.source1_entity_id
            );
        """).fetchone()

        # Integrity checks
        dups_s2 = con.execute("SELECT count(*) - count(DISTINCT (source1_entity_id, matched_entity_id)) FROM te_curr_s2").fetchone()[0]
        dups_s3 = con.execute("SELECT count(*) - count(DISTINCT (source1_entity_id, matched_entity_id)) FROM te_curr_s3").fetchone()[0]
        null_s1 = con.execute("SELECT count(*) FROM (SELECT source1_entity_id FROM te_curr_s2 WHERE source1_entity_id IS NULL UNION ALL SELECT source1_entity_id FROM te_curr_s3 WHERE source1_entity_id IS NULL)").fetchone()[0]
        null_tgt = con.execute("SELECT count(*) FROM (SELECT matched_entity_id FROM te_curr_s2 WHERE matched_entity_id IS NULL UNION ALL SELECT matched_entity_id FROM te_curr_s3 WHERE matched_entity_id IS NULL)").fetchone()[0]
        self_m = con.execute("SELECT count(*) FROM (SELECT * FROM te_curr_s2 WHERE source1_entity_id = matched_entity_id UNION ALL SELECT * FROM te_curr_s3 WHERE source1_entity_id = matched_entity_id)").fetchone()[0]
        
        cross_cntry_s2 = con.execute("SELECT count(*) FROM te_curr_s2 c JOIN test_s1 s1 ON c.source1_entity_id = s1.entity_id JOIN test_s2 tgt ON c.matched_entity_id = tgt.entity_id WHERE s1.country <> tgt.country").fetchone()[0]
        cross_cntry_s3 = con.execute("SELECT count(*) FROM te_curr_s3 c JOIN test_s1 s1 ON c.source1_entity_id = s1.entity_id JOIN test_s3 tgt ON c.matched_entity_id = tgt.entity_id WHERE s1.country <> tgt.country").fetchone()[0]
        cross_cntry_tot = cross_cntry_s2 + cross_cntry_s3
        integrity_passed = (dups_s2 == 0 and dups_s3 == 0 and null_s1 == 0 and null_tgt == 0 and self_m == 0 and cross_cntry_tot == 0)

        test_results[c_name] = {
            "s2_candidates": cands_s2,
            "s3_candidates": cands_s3,
            "total_candidates": tot_cands,
            "zero_s1": stats[0],
            "p50": stats[1],
            "p90": stats[2],
            "p95": stats[3],
            "p99": stats[4],
            "p999": stats[5],
            "max": stats[6],
            "ge_300": stats[7],
            "ge_500": stats[8],
            "ge_1000": stats[9],
            "integrity": {
                "duplicates": dups_s2 + dups_s3,
                "null_s1": null_s1,
                "null_tgt": null_tgt,
                "self_matches": self_m,
                "cross_country": cross_cntry_tot,
                "passed": integrity_passed
            },
            "elapsed_s": round(time.time() - t0, 2)
        }
        print(f"  -> Test p95: {stats[3]:.1f} | Test p99: {stats[4]:.1f} | >=300: {stats[7]:,} | Cands: {tot_cands:,} | ({time.time()-t0:.2f}s)")

    con.close()
    return test_results


def export_selected_candidates(best_cfg):
    """Export train and test candidate TSVs for the selected configuration."""
    print(f"\n=======================================================")
    print(f"EXPORTING CANDIDATE ARTIFACTS FOR {best_cfg['configuration']}")
    print(f"=======================================================")
    con = duckdb.connect()
    con.execute("PRAGMA threads=8;")
    con.execute("PRAGMA memory_limit='10GB';")
    con.create_function("transliterate_devanagari", transliterate_text, [str], str)

    cap2 = best_cfg["cap_exp02"]
    cap3 = best_cfg["cap_exp03"]
    checksums = {}

    # 1. Export TRAIN
    print("Exporting TRAIN candidate files...")
    for target in ['s2', 's3']:
        v2_path = REPO_ROOT / f"P2/data/candidates/train_candidate_pairs_{target}_v2.tsv"
        exp1_path = REPO_ROOT / f"P1/experiments/phase3/EXP-RET-01/train_candidates_{target}_exp_ret_01.tsv"
        exp2_path = REPO_ROOT / f"P1/experiments/phase3/EXP-RET-02/train_candidates_{target}_exp_ret_02_th080.tsv"
        exp3_path = REPO_ROOT / f"P1/experiments/phase3/EXP-RET-03/train_candidates_{target}_exp_ret_03.tsv"
        exp4_path = REPO_ROOT / f"P1/experiments/phase3/EXP-RET-04/train_candidates_{target}_exp_ret_04.tsv"

        con.execute(f"CREATE TEMP TABLE v2_{target} AS SELECT source1_entity_id, matched_entity_id FROM read_csv_auto('{v2_path}', delim='\t', header=true);")
        con.execute(f"""
            CREATE TEMP TABLE e1_raw_{target} AS SELECT source1_entity_id, matched_entity_id FROM read_csv_auto('{exp1_path}', delim='\t', header=true);
            CREATE TEMP TABLE e1_{target} AS
            SELECT r.source1_entity_id, r.matched_entity_id
            FROM e1_raw_{target} r
            JOIN (SELECT source1_entity_id FROM e1_raw_{target} GROUP BY source1_entity_id HAVING count(*) <= 100) k
              ON r.source1_entity_id = k.source1_entity_id;
            DROP TABLE e1_raw_{target};
        """)
        con.execute(f"CREATE TEMP TABLE e4_{target} AS SELECT source1_entity_id, matched_entity_id FROM read_csv_auto('{exp4_path}', delim='\t', header=true);")

        tbls = [f"v2_{target}", f"e1_{target}", f"e4_{target}"]
        if cap2 is not None:
            con.execute(f"""
                CREATE TEMP TABLE e2_raw_{target} AS SELECT source1_entity_id, matched_entity_id FROM read_csv_auto('{exp2_path}', delim='\t', header=true);
                CREATE TEMP TABLE e2_{target} AS
                SELECT r.source1_entity_id, r.matched_entity_id
                FROM e2_raw_{target} r
                JOIN (SELECT source1_entity_id FROM e2_raw_{target} GROUP BY source1_entity_id HAVING count(*) <= {cap2}) k
                  ON r.source1_entity_id = k.source1_entity_id;
                DROP TABLE e2_raw_{target};
            """)
            tbls.append(f"e2_{target}")

        if cap3 is not None:
            con.execute(f"""
                CREATE TEMP TABLE e3_raw_{target} AS SELECT source1_entity_id, matched_entity_id FROM read_csv_auto('{exp3_path}', delim='\t', header=true);
                CREATE TEMP TABLE e3_{target} AS
                SELECT r.source1_entity_id, r.matched_entity_id
                FROM e3_raw_{target} r
                JOIN (SELECT source1_entity_id FROM e3_raw_{target} GROUP BY source1_entity_id HAVING count(*) <= {cap3}) k
                  ON r.source1_entity_id = k.source1_entity_id;
                DROP TABLE e3_raw_{target};
            """)
            tbls.append(f"e3_{target}")

        union_sql = " UNION ALL ".join([f"SELECT source1_entity_id, matched_entity_id FROM {t}" for t in tbls])
        out_path = CANDIDATE_OUT_DIR / f"train_candidate_pairs_{target}_selected.tsv"
        print(f"Writing {out_path}...")
        con.execute(f"""
            COPY (
                SELECT DISTINCT source1_entity_id, matched_entity_id 
                FROM ({union_sql})
                ORDER BY source1_entity_id, matched_entity_id
            ) TO '{out_path}' (HEADER, DELIMITER '\t');
        """)
        sha = compute_sha256(out_path)
        checksums[out_path.name] = sha
        print(f"  Done. SHA256({out_path.name}) = {sha}")

    # 2. Export TEST
    print("\nExporting TEST candidate files...")
    test_s1_path = REPO_ROOT / "P1/data/entities/test/source1/test_s1_entities.parquet"
    con.execute(f"""
        CREATE TABLE test_s1 AS
        SELECT 
            entity_id,
            country_raw AS country,
            business_name_raw AS name_raw,
            business_address_raw AS addr_raw,
            regexp_replace(lower(trim(business_name_raw)), '[^a-z0-9]', '', 'g') AS clean_name,
            regexp_replace(
                regexp_replace(lower(trim(business_name_raw)), '[^a-z0-9]', '', 'g'),
                '(privatelimited|pvtlimited|pvtltd|pvtld|limited|ltd|llc|inc|corp|corporation|enterprises?|services?)$',
                ''
            ) AS core_name,
            LEFT(regexp_replace(lower(trim(business_name_raw)), '[^a-z0-9]', '', 'g'), 2) AS name_prefix_2,
            LEFT(regexp_replace(lower(trim(business_name_raw)), '[^a-z0-9]', '', 'g'), 3) AS name_prefix_3,
            regexp_replace(regexp_extract(trim(business_address_raw), '[0-9]+[A-Za-z]?', 0), '^0+', '') AS house_norm,
            regexp_replace(lower(trim(business_address_raw)), '[^a-z0-9]', '', 'g') AS addr_clean,
            regexp_extract(trim(business_address_raw), '(?:^|[^0-9])([0-9]{{5,6}})(?:[^0-9]|$)', 1) AS postal_code,
            CASE 
                WHEN lower(split_part(trim(business_name_raw), ' ', 1)) IN {COMMON_PREFIXES} 
                     AND split_part(trim(business_name_raw), ' ', 2) <> '' 
                THEN split_part(trim(business_name_raw), ' ', 2)
                ELSE split_part(trim(business_name_raw), ' ', 1)
            END AS root_token,
            LEFT(regexp_replace(lower(trim(transliterate_devanagari(business_name_raw))), '[^a-z0-9]', '', 'g'), 2) AS translit_prefix_2,
            regexp_replace(lower(trim(transliterate_devanagari(business_name_raw))), '[^a-z0-9]', '', 'g') AS translit_name
        FROM read_parquet('{test_s1_path}');
    """)

    for target in ['s2', 's3']:
        tgt_parquet = REPO_ROOT / f"P1/data/entities/test/{'source2' if target == 's2' else 'source3'}/test_{target}_entities.parquet"
        con.execute(f"""
            CREATE TABLE test_{target} AS
            SELECT 
                entity_id,
                country_raw AS country,
                business_name_raw AS name_raw,
                business_address_raw AS addr_raw,
                regexp_replace(lower(trim(business_name_raw)), '[^a-z0-9]', '', 'g') AS clean_name,
                regexp_replace(
                    regexp_replace(lower(trim(business_name_raw)), '[^a-z0-9]', '', 'g'),
                    '(privatelimited|pvtlimited|pvtltd|pvtld|limited|ltd|llc|inc|corp|corporation|enterprises?|services?)$',
                    ''
                ) AS core_name,
                LEFT(regexp_replace(lower(trim(business_name_raw)), '[^a-z0-9]', '', 'g'), 2) AS name_prefix_2,
                LEFT(regexp_replace(lower(trim(business_name_raw)), '[^a-z0-9]', '', 'g'), 3) AS name_prefix_3,
                regexp_replace(regexp_extract(trim(business_address_raw), '[0-9]+[A-Za-z]?', 0), '^0+', '') AS house_norm,
                regexp_replace(lower(trim(business_address_raw)), '[^a-z0-9]', '', 'g') AS addr_clean,
                regexp_extract(trim(business_address_raw), '(?:^|[^0-9])([0-9]{{5,6}})(?:[^0-9]|$)', 1) AS postal_code,
                CASE 
                    WHEN lower(split_part(trim(business_name_raw), ' ', 1)) IN {COMMON_PREFIXES} 
                         AND split_part(trim(business_name_raw), ' ', 2) <> '' 
                    THEN split_part(trim(business_name_raw), ' ', 2)
                    ELSE split_part(trim(business_name_raw), ' ', 1)
                END AS root_token,
                LEFT(regexp_replace(lower(trim(transliterate_devanagari(business_name_raw))), '[^a-z0-9]', '', 'g'), 2) AS translit_prefix_2,
                regexp_replace(lower(trim(transliterate_devanagari(business_name_raw))), '[^a-z0-9]', '', 'g') AS translit_name
            FROM read_parquet('{tgt_parquet}');
        """)

        v2_path = REPO_ROOT / f"P2/data/candidates/test_candidate_pairs_{target}_v2.tsv"
        con.execute(f"CREATE TEMP TABLE te_v2_{target} AS SELECT source1_entity_id, matched_entity_id FROM read_csv_auto('{v2_path}', delim='\t', header=True);")

        con.execute(f"""
            CREATE TEMP TABLE te_e1_raw_{target} AS
            SELECT s1.entity_id AS source1_entity_id, tgt.entity_id AS matched_entity_id
            FROM test_s1 s1 JOIN test_{target} tgt ON s1.country = tgt.country AND s1.core_name = tgt.core_name
            WHERE LENGTH(s1.core_name) >= 5;

            CREATE TEMP TABLE te_e1_{target} AS
            SELECT r.source1_entity_id, r.matched_entity_id
            FROM te_e1_raw_{target} r
            JOIN (SELECT source1_entity_id FROM te_e1_raw_{target} GROUP BY source1_entity_id HAVING count(*) <= 100) k
              ON r.source1_entity_id = k.source1_entity_id;
            DROP TABLE te_e1_raw_{target};
        """)

        con.execute(f"""
            CREATE TEMP TABLE te_e4_{target} AS
            SELECT DISTINCT source1_entity_id, matched_entity_id FROM (
                SELECT s1.entity_id AS source1_entity_id, tgt.entity_id AS matched_entity_id
                FROM test_s1 s1 JOIN test_{target} tgt
                  ON s1.country = tgt.country AND LEFT(s1.addr_clean, 10) = LEFT(tgt.addr_clean, 10) AND s1.name_prefix_3 = tgt.name_prefix_3
                WHERE LENGTH(s1.addr_clean) >= 10 AND LENGTH(tgt.addr_clean) >= 10 AND LENGTH(s1.name_prefix_3) = 3
                UNION ALL
                SELECT s1.entity_id AS source1_entity_id, tgt.entity_id AS matched_entity_id
                FROM test_s1 s1 JOIN test_{target} tgt
                  ON s1.country = tgt.country AND s1.postal_code = tgt.postal_code AND s1.root_token = tgt.root_token
                WHERE s1.postal_code IS NOT NULL AND LENGTH(s1.postal_code) >= 5 AND s1.root_token IS NOT NULL AND LENGTH(s1.root_token) >= 4
            );
        """)

        tbls = [f"te_v2_{target}", f"te_e1_{target}", f"te_e4_{target}"]

        if cap2 is not None:
            con.execute(f"""
                CREATE TEMP TABLE te_e2_raw_{target} AS
                SELECT s1.entity_id AS source1_entity_id, tgt.entity_id AS matched_entity_id
                FROM test_s1 s1 JOIN test_{target} tgt
                  ON s1.country = tgt.country AND s1.house_norm = tgt.house_norm AND s1.name_prefix_2 = tgt.name_prefix_2
                WHERE s1.house_norm IS NOT NULL AND s1.house_norm <> ''
                  AND LENGTH(s1.clean_name) >= 2 AND LENGTH(tgt.clean_name) >= 2
                  AND jaro_winkler_similarity(s1.clean_name, tgt.clean_name) >= 0.80;

                CREATE TEMP TABLE te_e2_{target} AS
                SELECT r.source1_entity_id, r.matched_entity_id
                FROM te_e2_raw_{target} r
                JOIN (SELECT source1_entity_id FROM te_e2_raw_{target} GROUP BY source1_entity_id HAVING count(*) <= {cap2}) k
                  ON r.source1_entity_id = k.source1_entity_id;
                DROP TABLE te_e2_raw_{target};
            """)
            tbls.append(f"te_e2_{target}")

        if cap3 is not None:
            con.execute(f"""
                CREATE TEMP TABLE te_e3_raw_{target} AS
                SELECT DISTINCT source1_entity_id, matched_entity_id FROM (
                    SELECT s1.entity_id AS source1_entity_id, tgt.entity_id AS matched_entity_id
                    FROM test_s1 s1 JOIN test_{target} tgt
                      ON s1.country = tgt.country AND s1.house_norm = tgt.house_norm AND s1.translit_prefix_2 = tgt.translit_prefix_2
                    WHERE s1.house_norm IS NOT NULL AND s1.house_norm <> ''
                      AND LENGTH(s1.translit_name) >= 2 AND LENGTH(tgt.translit_name) >= 2
                      AND jaro_winkler_similarity(s1.translit_name, tgt.translit_name) >= 0.80
                    UNION ALL
                    SELECT s1.entity_id AS source1_entity_id, tgt.entity_id AS matched_entity_id
                    FROM test_s1 s1 JOIN test_{target} tgt
                      ON s1.country = tgt.country AND s1.house_norm = tgt.house_norm AND s1.postal_code = tgt.postal_code
                    WHERE s1.country = 'India' AND s1.house_norm IS NOT NULL AND s1.house_norm <> ''
                      AND s1.postal_code IS NOT NULL AND LENGTH(s1.postal_code) >= 5
                );

                CREATE TEMP TABLE te_e3_{target} AS
                SELECT r.source1_entity_id, r.matched_entity_id
                FROM te_e3_raw_{target} r
                JOIN (SELECT source1_entity_id FROM te_e3_raw_{target} GROUP BY source1_entity_id HAVING count(*) <= {cap3}) k
                  ON r.source1_entity_id = k.source1_entity_id;
                DROP TABLE te_e3_raw_{target};
            """)
            tbls.append(f"te_e3_{target}")

        union_sql = " UNION ALL ".join([f"SELECT source1_entity_id, matched_entity_id FROM {t}" for t in tbls])
        out_path = CANDIDATE_OUT_DIR / f"test_candidate_pairs_{target}_selected.tsv"
        print(f"Writing {out_path}...")
        con.execute(f"""
            COPY (
                SELECT DISTINCT source1_entity_id, matched_entity_id 
                FROM ({union_sql})
                ORDER BY source1_entity_id, matched_entity_id
            ) TO '{out_path}' (HEADER, DELIMITER '\t');
        """)
        sha = compute_sha256(out_path)
        checksums[out_path.name] = sha
        print(f"  Done. SHA256({out_path.name}) = {sha}")

    con.close()
    return checksums


def main():
    t_start = time.time()
    train_metrics = run_train_phase()
    test_metrics = run_test_phase()

    # Combine metrics
    all_metrics = []
    for cfg in CONFIG_DEFS:
        c_name = cfg["name"]
        tr = train_metrics[c_name]
        te = test_results = test_metrics[c_name]

        meets_gate = (te["p95"] <= 300.0 and tr["combined_recall"] > 0.657623)

        combined_entry = {
            "configuration": c_name,
            "description": cfg["desc"],
            "cap_exp02": cfg["cap_e2"],
            "cap_exp03": cfg["cap_e3"],
            "train": tr,
            "test": te,
            "integrity": te["integrity"],
            "gate_check": {
                "test_p95_le_300": bool(te["p95"] <= 300.0),
                "train_recall_gt_v2": bool(tr["combined_recall"] > 0.657623),
                "passed_both_gates": meets_gate
            }
        }
        all_metrics.append(combined_entry)

    # Print Summary Table
    print("\n" + "="*80)
    print("EXPERIMENT SWEEP COMPLETE SUMMARY TABLE")
    print("="*80)
    print(f"{'Config':<14} | {'02 Cap':<6} | {'03 Cap':<6} | {'Train Recall':<12} | {'Test p95':<8} | {'Test p99':<8} | {'Train Cands':<12} | {'Test Cands':<12} | {'Gate Status'}")
    print("-" * 105)
    for m in all_metrics:
        gate_str = "PASS BOTH" if m["gate_check"]["passed_both_gates"] else "FAIL"
        print(f"{m['configuration']:<14} | {str(m['cap_exp02']):<6} | {str(m['cap_exp03']):<6} | {m['train']['combined_recall']:.4%}     | {m['test']['p95']:<8.1f} | {m['test']['p99']:<8.1f} | {m['train']['total_candidates']:<12,} | {m['test']['total_candidates']:<12,} | {gate_str}")

    # Identify winning configuration
    valid_configs = [m for m in all_metrics if m["gate_check"]["passed_both_gates"]]
    valid_configs.sort(key=lambda x: x["train"]["combined_recall"], reverse=True)

    checksums = {}
    best_cfg = None
    if valid_configs:
        best_cfg = valid_configs[0]
        print(f"\n>>> Highest-recall configuration satisfying Phase 3 constraints:")
        print(f"    Name: {best_cfg['configuration']} - {best_cfg['description']}")
        print(f"    Train Combined Recall: {best_cfg['train']['combined_recall']:.6%} (Threshold: >65.7623%)")
        print(f"    Test Combined p95:     {best_cfg['test']['p95']:.1f} (Threshold: <=300.0)")
        checksums = export_selected_candidates(best_cfg)

    # Save summary TSV
    summary_tsv_path = EXP_DIR / "experiment_summary.tsv"
    with open(summary_tsv_path, "w") as f:
        header = [
            "Configuration", "EXP-02 Cap", "EXP-03 Cap",
            "Train Recall", "Test p95", "Test p99",
            "Train Candidates", "Test Candidates",
            "Train Zero-S1", "Test Zero-S1", "Integrity Status", "Gate Status"
        ]
        f.write("\t".join(header) + "\n")
        for m in all_metrics:
            gate_str = "PASS" if m["gate_check"]["passed_both_gates"] else "FAIL"
            row = [
                m["configuration"],
                str(m["cap_exp02"]) if m["cap_exp02"] is not None else "None",
                str(m["cap_exp03"]) if m["cap_exp03"] is not None else "None",
                f"{m['train']['combined_recall']:.6%}",
                f"{m['test']['p95']:.1f}",
                f"{m['test']['p99']:.1f}",
                f"{m['train']['total_candidates']:,}",
                f"{m['test']['total_candidates']:,}",
                f"{m['train']['zero_s1']:,}",
                f"{m['test']['zero_s1']:,}",
                "PASS" if m["integrity"]["passed"] else "FAIL",
                gate_str
            ]
            f.write("\t".join(row) + "\n")
    print(f"\nWrote {summary_tsv_path}")

    # Save metrics JSON
    metrics_json_path = EXP_DIR / "metrics.json"
    with open(metrics_json_path, "w") as f:
        json.dump({
            "metrics": all_metrics,
            "selected_configuration": best_cfg["configuration"] if best_cfg else None,
            "selected_sha256": checksums,
            "total_elapsed_seconds": round(time.time() - t_start, 2)
        }, f, indent=2)
    print(f"Wrote {metrics_json_path}")
    print(f"All done in {time.time()-t_start:.2f}s")

if __name__ == "__main__":
    main()
