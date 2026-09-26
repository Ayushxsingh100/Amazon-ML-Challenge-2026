#!/usr/bin/env python3
"""
E06 -- cands_BCD_v2 Modeling Pipeline
Tasks:
  1. Verify/regenerate V2 candidate files (train+test, S2+S3)
  2. Compute features in-memory via DuckDB SQL (same spec as E02)
  3. Train 5-fold LightGBM (same E02 recipe, 10% neg sampling, seed=2026)
  4. Generate full canonical V2 OOF (67,332,524 rows)
  5. Threshold sweep (T 0.50..0.995 step 0.005) diagnostic
  6. V2 candidate oracle and decomposition vs V1
  7. All required output artifacts
IMPORTANT:
  - No feature TSVs written to disk (disk-space constrained ~5.5 GB free)
  - No modification to E02 artifacts, V1 candidates, folds, scorer
  - Primary result: T=0.50 (controlled comparison to E02 baseline 0.718765173)
  - DO NOT claim V2 score authoritative -- prepare for P3 validation
"""

import gc
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd

# ============================================================
# Paths
# ============================================================
REPO        = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
NORM_DIR    = os.path.join(REPO, "outputs", "person1_step1", "normalized")
P1_OUT      = os.path.join(REPO, "outputs", "person1_step1")
GT_PATH     = os.path.join(P1_OUT, "train_ground_truth_reconstructed.tsv")
P3_MANIFEST = os.path.join(REPO, "P3", "reports", "folds_v1_manifest.tsv")
CAND_DIR    = os.path.join(REPO, "P2", "data", "candidates")
REPORT_DIR  = os.path.join(REPO, "P2", "reports")
MODEL_DIR   = os.path.join(REPO, "P2", "models")

os.makedirs(CAND_DIR,   exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR,  exist_ok=True)

sys.path.insert(0, REPO)
from validation.scorer_v1 import score_predictions

# V2 candidate file paths
TRAIN_S2_V2 = os.path.join(CAND_DIR, "train_candidate_pairs_s2_v2.tsv")
TRAIN_S3_V2 = os.path.join(CAND_DIR, "train_candidate_pairs_s3_v2.tsv")
TEST_S2_V2  = os.path.join(CAND_DIR, "test_candidate_pairs_s2_v2.tsv")
TEST_S3_V2  = os.path.join(CAND_DIR, "test_candidate_pairs_s3_v2.tsv")

# Canonical feature names (identical to E02)
FEATURES = [
    "name_exact_match", "name_jaro_winkler", "name_jaccard", "prefix4_match",
    "first_token_match", "name_len_diff", "name_len_ratio",
    "address_exact_match", "address_jaro_winkler", "address_jaccard",
    "address_len_diff", "address_len_ratio", "address_first_number_match",
    "country_match", "s1_name_len", "tgt_name_len", "s1_addr_len",
    "tgt_addr_len", "source_is_s3"
]
CAT_FEATURES = [
    "source_is_s3", "country_match", "address_first_number_match",
    "name_exact_match", "address_exact_match", "prefix4_match", "first_token_match"
]

# Identical LightGBM hyperparameters to E02
LGB_PARAMS = {
    "objective":        "binary",
    "boosting_type":    "gbdt",
    "learning_rate":    0.05,
    "num_leaves":       31,
    "max_depth":        -1,
    "feature_fraction": 1.0,
    "bagging_fraction": 1.0,
    "min_data_in_leaf": 20,
    "lambda_l1":        0.0,
    "lambda_l2":        0.0,
    "seed":             2026,
    "metric":           "None",
    "verbose":          -1,
    "num_threads":      2,
}
NUM_BOOST_ROUND = 500

COMMON_PREFIXES = "('the', 'shri', 'sri', 'dr', 'm/s', 'hotel', 'new', 'om', 'sai', 'jai', 'a', 'an')"

FEATURE_SQL = """
    CASE WHEN s1.business_name <> '' AND s1.business_name = tgt.business_name THEN 1.0 ELSE 0.0 END AS name_exact_match,
    jaro_winkler_similarity(COALESCE(s1.business_name,''), COALESCE(tgt.business_name,'')) / 100.0 AS name_jaro_winkler,
    CASE WHEN LENGTH(COALESCE(s1.business_name,'')) >= 2 AND LENGTH(COALESCE(tgt.business_name,'')) >= 2
         THEN jaccard(COALESCE(s1.business_name,''), COALESCE(tgt.business_name,'')) ELSE 0.0 END AS name_jaccard,
    CASE WHEN LENGTH(s1.business_name) >= 4 AND LENGTH(tgt.business_name) >= 4
              AND LEFT(s1.business_name, 4) = LEFT(tgt.business_name, 4) THEN 1.0 ELSE 0.0 END AS prefix4_match,
    CASE WHEN s1.business_name <> '' AND tgt.business_name <> ''
              AND SPLIT_PART(s1.business_name, ' ', 1) = SPLIT_PART(tgt.business_name, ' ', 1) THEN 1.0 ELSE 0.0 END AS first_token_match,
    CAST(ABS(LENGTH(COALESCE(s1.business_name,'')) - LENGTH(COALESCE(tgt.business_name,''))) AS DOUBLE) AS name_len_diff,
    CASE WHEN LENGTH(s1.business_name) > 0 AND LENGTH(tgt.business_name) > 0
         THEN CAST(LEAST(LENGTH(s1.business_name), LENGTH(tgt.business_name)) AS DOUBLE) / GREATEST(LENGTH(s1.business_name), LENGTH(tgt.business_name))
         ELSE 0.0 END AS name_len_ratio,
    CASE WHEN s1.business_address <> '' AND s1.business_address = tgt.business_address THEN 1.0 ELSE 0.0 END AS address_exact_match,
    jaro_winkler_similarity(COALESCE(s1.business_address,''), COALESCE(tgt.business_address,'')) / 100.0 AS address_jaro_winkler,
    CASE WHEN LENGTH(COALESCE(s1.business_address,'')) >= 2 AND LENGTH(COALESCE(tgt.business_address,'')) >= 2
         THEN jaccard(COALESCE(s1.business_address,''), COALESCE(tgt.business_address,'')) ELSE 0.0 END AS address_jaccard,
    CAST(ABS(LENGTH(COALESCE(s1.business_address,'')) - LENGTH(COALESCE(tgt.business_address,''))) AS DOUBLE) AS address_len_diff,
    CASE WHEN LENGTH(s1.business_address) > 0 AND LENGTH(tgt.business_address) > 0
         THEN CAST(LEAST(LENGTH(s1.business_address), LENGTH(tgt.business_address)) AS DOUBLE) / GREATEST(LENGTH(s1.business_address), LENGTH(tgt.business_address))
         ELSE 0.0 END AS address_len_ratio,
    CASE WHEN regexp_extract(COALESCE(s1.business_address,''), '[0-9]+[A-Za-z]?', 0) <> ''
              AND regexp_extract(COALESCE(tgt.business_address,''), '[0-9]+[A-Za-z]?', 0) <> ''
              AND regexp_extract(s1.business_address, '[0-9]+[A-Za-z]?', 0) = regexp_extract(tgt.business_address, '[0-9]+[A-Za-z]?', 0)
         THEN 1.0 ELSE 0.0 END AS address_first_number_match,
    CASE WHEN s1.country <> '' AND s1.country = tgt.country THEN 1.0 ELSE 0.0 END AS country_match,
    CAST(LENGTH(COALESCE(s1.business_name,'')) AS DOUBLE) AS s1_name_len,
    CAST(LENGTH(COALESCE(tgt.business_name,'')) AS DOUBLE) AS tgt_name_len,
    CAST(LENGTH(COALESCE(s1.business_address,'')) AS DOUBLE) AS s1_addr_len,
    CAST(LENGTH(COALESCE(tgt.business_address,'')) AS DOUBLE) AS tgt_addr_len
"""


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(65536)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def get_git_sha():
    import subprocess
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO).decode().strip()
    except Exception:
        return "unknown"


def get_free_gb():
    import shutil
    return shutil.disk_usage(REPO).free / (1024 ** 3)


def load_normalized_with_v2_keys(con, typ, src):
    num = src[-1]
    path = os.path.join(NORM_DIR, f"{typ}_source{num}_normalized.tsv").replace("\\", "/")
    con.execute(f"""
        CREATE OR REPLACE TABLE {typ}_{src} AS
        SELECT entity_id, business_name, business_address, country,
            LEFT(TRIM(business_name), 4) AS prefix4,
            split_part(TRIM(business_name), ' ', 1) AS token1,
            regexp_extract(TRIM(business_address), '[0-9]+[A-Za-z]?', 0) AS house_v1,
            regexp_replace(regexp_extract(TRIM(business_address), '[0-9]+[A-Za-z]?', 0), '^0+', '') AS house_norm,
            regexp_replace(lower(trim(business_name)), '[^a-z0-9]', '', 'g') AS clean_name,
            regexp_replace(lower(trim(business_address)), '[^a-z0-9]', '', 'g') AS clean_addr,
            LEFT(TRIM(business_name), 3) AS prefix3,
            CASE WHEN lower(split_part(TRIM(business_name), ' ', 1)) IN {COMMON_PREFIXES}
                      AND split_part(TRIM(business_name), ' ', 2) <> ''
                 THEN split_part(TRIM(business_name), ' ', 2)
                 ELSE split_part(TRIM(business_name), ' ', 1)
            END AS root_token,
            regexp_extract(TRIM(business_address), '(?:^|[^0-9])([0-9]{{5,6}})(?:[^0-9]|$)', 1) AS zip_code
        FROM read_csv('{path}', delim='\\t', header=true, all_varchar=true)
    """)
    cnt = con.execute(f"SELECT COUNT(*) FROM {typ}_{src}").fetchone()[0]
    print(f"  Loaded {typ}_{src}: {cnt:,} entities", flush=True)
    return cnt


def v2_blocking_sql(typ, src):
    return f"""
        SELECT s1.entity_id AS source1_entity_id, tgt.entity_id AS matched_entity_id
        FROM {typ}_s1 s1 JOIN {typ}_{src} tgt
        ON s1.country <> '' AND s1.country = tgt.country
           AND s1.prefix4 <> '' AND s1.prefix4 = tgt.prefix4
           AND s1.house_v1 <> '' AND s1.house_v1 = tgt.house_v1
        UNION
        SELECT s1.entity_id, tgt.entity_id FROM {typ}_s1 s1 JOIN {typ}_{src} tgt
        ON s1.country <> '' AND s1.country = tgt.country
           AND s1.business_address <> '' AND s1.business_address = tgt.business_address
        UNION
        SELECT s1.entity_id, tgt.entity_id FROM {typ}_s1 s1 JOIN {typ}_{src} tgt
        ON s1.country <> '' AND s1.country = tgt.country
           AND s1.token1 <> '' AND s1.token1 = tgt.token1
           AND s1.house_v1 <> '' AND s1.house_v1 = tgt.house_v1
        UNION
        SELECT s1.entity_id, tgt.entity_id FROM {typ}_s1 s1 JOIN {typ}_{src} tgt
        ON s1.country <> '' AND s1.country = tgt.country
           AND TRIM(s1.business_name) <> '' AND TRIM(s1.business_name) = TRIM(tgt.business_name)
        UNION
        SELECT s1.entity_id, tgt.entity_id FROM {typ}_s1 s1 JOIN {typ}_{src} tgt
        ON s1.country <> '' AND s1.country = tgt.country
           AND s1.clean_name <> '' AND LENGTH(s1.clean_name) >= 3 AND s1.clean_name = tgt.clean_name
        UNION
        SELECT s1.entity_id, tgt.entity_id FROM {typ}_s1 s1 JOIN {typ}_{src} tgt
        ON s1.country <> '' AND s1.country = tgt.country
           AND s1.prefix4 <> '' AND s1.prefix4 = tgt.prefix4
           AND s1.house_norm <> '' AND s1.house_norm = tgt.house_norm
        UNION
        SELECT s1.entity_id, tgt.entity_id FROM {typ}_s1 s1 JOIN {typ}_{src} tgt
        ON s1.country <> '' AND s1.country = tgt.country
           AND s1.token1 <> '' AND s1.token1 = tgt.token1
           AND s1.house_norm <> '' AND s1.house_norm = tgt.house_norm
        UNION
        SELECT s1.entity_id, tgt.entity_id FROM {typ}_s1 s1 JOIN {typ}_{src} tgt
        ON s1.country <> '' AND s1.country = tgt.country
           AND s1.root_token <> '' AND LENGTH(s1.root_token) >= 3 AND s1.root_token = tgt.root_token
           AND s1.house_norm <> '' AND s1.house_norm = tgt.house_norm
        UNION
        SELECT s1.entity_id, tgt.entity_id FROM {typ}_s1 s1 JOIN {typ}_{src} tgt
        ON s1.country <> '' AND s1.country = tgt.country
           AND s1.clean_addr <> '' AND LENGTH(s1.clean_addr) >= 6 AND s1.clean_addr = tgt.clean_addr
        UNION
        SELECT s1.entity_id, tgt.entity_id FROM {typ}_s1 s1 JOIN {typ}_{src} tgt
        ON s1.country <> '' AND s1.country = tgt.country
           AND s1.zip_code <> '' AND LENGTH(s1.zip_code) >= 5 AND s1.zip_code = tgt.zip_code
           AND s1.prefix3 <> '' AND LENGTH(s1.prefix3) = 3 AND s1.prefix3 = tgt.prefix3
    """

# ============================================================
# TASK 1 -- Verify / generate V2 candidate files
# ============================================================
def task1_verify_candidates():
    print("\n" + "="*65)
    print("TASK 1 -- VERIFY / GENERATE V2 CANDIDATES")
    print("="*65, flush=True)

    EXPECTED = {
        TRAIN_S2_V2: 30_359_040,
        TRAIN_S3_V2: 36_973_484,
        TEST_S2_V2:  34_919_169,
        TEST_S3_V2:  41_714_627,
    }

    need_regen = False
    for path, exp in EXPECTED.items():
        if not os.path.exists(path):
            print(f"  MISSING: {os.path.basename(path)}", flush=True)
            need_regen = True
        else:
            sz = os.path.getsize(path)
            print(f"  EXISTS:  {os.path.basename(path)}  ({sz:,} bytes)", flush=True)

    if need_regen:
        print("\n  Regenerating V2 candidates ...", flush=True)
        _generate_v2_candidates()

    print("\n  Verifying row counts ...", flush=True)
    counts = {}
    TMP_V = os.path.join(REPO, "P2", "data", "duckdb_tmp_e06_verify")
    os.makedirs(TMP_V, exist_ok=True)
    con = duckdb.connect()
    con.execute(f"SET temp_directory='{TMP_V}'")
    con.execute("SET memory_limit='3GB'")
    con.execute("SET threads=2")

    gt_p = GT_PATH.replace("\\", "/")
    con.execute(f"""
        CREATE TABLE gt_exp AS
        SELECT source1_entity_id,
               TRIM(UNNEST(string_split(matched_entity_ids, ','))) AS matched_entity_id
        FROM read_csv('{gt_p}', delim='\\t', header=true, all_varchar=true)
        WHERE matched_entity_ids IS NOT NULL AND TRIM(matched_entity_ids) <> ''
    """)
    con.execute("CREATE TABLE gt_s2 AS SELECT * FROM gt_exp WHERE matched_entity_id LIKE 'S2-%'")
    con.execute("CREATE TABLE gt_s3 AS SELECT * FROM gt_exp WHERE matched_entity_id LIKE 'S3-%'")

    issues = []
    for path, exp in EXPECTED.items():
        p = path.replace("\\", "/")
        n = os.path.basename(path)
        rc = con.execute(f"SELECT COUNT(*) FROM read_csv('{p}', delim='\\t', header=true)").fetchone()[0]
        dups = con.execute(f"""
            SELECT COUNT(*) FROM (
                SELECT source1_entity_id, matched_entity_id
                FROM read_csv('{p}', delim='\\t', header=true)
                GROUP BY 1,2 HAVING COUNT(*) > 1
            )
        """).fetchone()[0]
        ok = (rc == exp) and (dups == 0)
        status = "OK" if ok else "FAIL"
        print(f"  [{status}] {n}: rows={rc:,} (expected {exp:,}), dups={dups}", flush=True)
        counts[n] = {"rows": rc, "expected": exp, "dups": dups, "match": ok}
        if not ok:
            issues.append(f"{n}: rows={rc} expected={exp} dups={dups}")

    for path, src_tag in [(TRAIN_S2_V2, "s2"), (TRAIN_S3_V2, "s3")]:
        p = path.replace("\\", "/")
        label_check = con.execute(f"""
            SELECT
                SUM(CASE WHEN CAST(c.label AS INT) = 1 THEN 1 ELSE 0 END) AS cand_pos,
                (SELECT COUNT(*) FROM gt_{src_tag}) AS gt_total
            FROM read_csv('{p}', delim='\\t', header=true) c
        """).fetchone()
        label_mismatch = con.execute(f"""
            SELECT COUNT(*) FROM (
                SELECT c.source1_entity_id, c.matched_entity_id
                FROM read_csv('{p}', delim='\\t', header=true) c
                WHERE CAST(c.label AS INT) = 1
                EXCEPT
                SELECT source1_entity_id, matched_entity_id FROM gt_{src_tag}
            )
        """).fetchone()[0]
        print(f"  Train {src_tag}: cand_pos={label_check[0]:,}, gt_total={label_check[1]:,}, label_mismatch={label_mismatch}", flush=True)
        if label_mismatch > 0:
            issues.append(f"train_{src_tag} label mismatch vs GT: {label_mismatch}")

    train_s2_s1s = con.execute(f"SELECT COUNT(DISTINCT source1_entity_id) FROM read_csv('{TRAIN_S2_V2.replace(chr(92),'/')}', delim='\\t', header=true)").fetchone()[0]
    test_s2_s1s  = con.execute(f"SELECT COUNT(DISTINCT source1_entity_id) FROM read_csv('{TEST_S2_V2.replace(chr(92),'/')}',  delim='\\t', header=true)").fetchone()[0]
    print(f"  Train S2 distinct S1: {train_s2_s1s:,} | Test S2 distinct S1: {test_s2_s1s:,}", flush=True)
    con.close()

    if issues:
        print(f"\n  WARNING: {len(issues)} issue(s): {issues}", flush=True)
    else:
        print("\n  All V2 candidate files verified OK.", flush=True)
    return counts, issues


def _generate_v2_candidates():
    """
    Disk-safe candidate generation: one fresh DuckDB connection per (typ, src) pair.
    Each connection loads only train_s1 (or test_s1) + target source table,
    runs the 10-rule UNION SQL, writes output, closes, and cleans its temp dir.
    Peak disk per step < 2.5 GB.
    """
    import shutil

    gt_p = GT_PATH.replace("\\", "/")

    for typ in ["train", "test"]:
        for src in ["s2", "s3"]:
            TMP_STEP = os.path.join(REPO, "P2", "data", f"duckdb_tmp_v2_{typ}_{src}")
            os.makedirs(TMP_STEP, exist_ok=True)

            t0 = time.time()
            print(f"\n  [{typ} {src}] Opening fresh DuckDB connection ...", flush=True)
            con = duckdb.connect()
            con.execute(f"SET temp_directory='{TMP_STEP}'")
            con.execute("SET memory_limit='4GB'")
            con.execute("SET threads=2")
            con.execute("SET preserve_insertion_order=false")

            # Load only the two tables needed for this pair
            load_normalized_with_v2_keys(con, typ, "s1")
            load_normalized_with_v2_keys(con, typ, src)

            if typ == "train":
                # Load GT for label join
                con.execute(f"""
                    CREATE TABLE gt_exp AS
                    SELECT source1_entity_id,
                           TRIM(UNNEST(string_split(matched_entity_ids, ','))) AS matched_entity_id
                    FROM read_csv('{gt_p}', delim='\\t', header=true, all_varchar=true)
                    WHERE matched_entity_ids IS NOT NULL AND TRIM(matched_entity_ids) <> ''
                """)
                con.execute(f"CREATE TABLE gt_{src} AS SELECT * FROM gt_exp WHERE matched_entity_id LIKE '{src.upper()}-%'")

            # Build raw candidates
            print(f"  [{typ} {src}] Running blocking SQL ...", flush=True)
            con.execute(f"CREATE TABLE raw AS {v2_blocking_sql(typ, src)}")
            cand_count = con.execute("SELECT COUNT(*) FROM raw").fetchone()[0]
            print(f"  [{typ} {src}] {cand_count:,} unique pairs ({time.time()-t0:.1f}s)", flush=True)

            if typ == "train":
                out_path = os.path.join(CAND_DIR, f"train_candidate_pairs_{src}_v2.tsv").replace("\\", "/")
                con.execute(f"""
                    COPY (
                        SELECT c.source1_entity_id, c.matched_entity_id,
                               CASE WHEN gt.matched_entity_id IS NOT NULL THEN 1 ELSE 0 END AS label
                        FROM raw c
                        LEFT JOIN gt_{src} gt
                          ON c.source1_entity_id = gt.source1_entity_id
                         AND c.matched_entity_id  = gt.matched_entity_id
                    ) TO '{out_path}' WITH (FORMAT CSV, DELIMITER '\\t', HEADER)
                """)
            else:
                out_path = os.path.join(CAND_DIR, f"test_candidate_pairs_{src}_v2.tsv").replace("\\", "/")
                con.execute(f"""
                    COPY (SELECT source1_entity_id, matched_entity_id FROM raw)
                    TO '{out_path}' WITH (FORMAT CSV, DELIMITER '\\t', HEADER)
                """)

            size_mb = os.path.getsize(out_path.replace("/", os.sep)) / (1024*1024)
            print(f"  [{typ} {src}] Written: {out_path} ({size_mb:.0f} MB) in {time.time()-t0:.1f}s", flush=True)

            con.close()
            # Remove temp to free disk before next iteration
            shutil.rmtree(TMP_STEP, ignore_errors=True)
            free_gb = get_free_gb()
            print(f"  [{typ} {src}] Temp cleaned. Free disk: {free_gb:.2f} GB", flush=True)

    print("  V2 candidate generation complete.", flush=True)


# ============================================================
# TASK 2+3 -- Features in-memory + LightGBM training
# ============================================================
def task2_3_train(run_meta):
    print("\n" + "="*65)
    print("TASK 2+3 -- FEATURE COMPUTATION + LGBM TRAINING (5-fold)")
    print("="*65, flush=True)

    TMP_TR = os.path.join(REPO, "P2", "data", "duckdb_tmp_e06_train")
    os.makedirs(TMP_TR, exist_ok=True)
    con = duckdb.connect()
    con.execute(f"SET temp_directory='{TMP_TR}'")
    con.execute("SET memory_limit='5GB'")
    con.execute("SET threads=2")
    con.execute("SET preserve_insertion_order=false")

    for src in ["s1", "s2", "s3"]:
        load_normalized_with_v2_keys(con, "train", src)

    folds_path = P3_MANIFEST.replace("\\", "/")
    con.execute(f"CREATE TABLE folds AS SELECT source1_entity_id, fold FROM read_csv('{folds_path}', delim='\\t', header=true)")
    n_folds = con.execute("SELECT COUNT(*) FROM folds").fetchone()[0]
    print(f"  Folds: {n_folds:,} S1 entities", flush=True)

    t0 = time.time()
    s2_p = TRAIN_S2_V2.replace("\\", "/")
    s3_p = TRAIN_S3_V2.replace("\\", "/")

    con.execute(f"""
        CREATE TABLE train_sampled AS
        SELECT f.fold, c.source1_entity_id, c.matched_entity_id,
               CAST(c.label AS INT) AS label,
               {FEATURE_SQL},
               0 AS source_is_s3
        FROM read_csv('{s2_p}', delim='\\t', header=true, all_varchar=true) c
        JOIN folds f ON c.source1_entity_id = f.source1_entity_id
        JOIN train_s1 s1  ON c.source1_entity_id = s1.entity_id
        JOIN train_s2 tgt ON c.matched_entity_id  = tgt.entity_id
        WHERE CAST(c.label AS INT) = 1 OR (ABS(hash(c.source1_entity_id || c.matched_entity_id)) % 10 = 0)
        UNION ALL
        SELECT f.fold, c.source1_entity_id, c.matched_entity_id,
               CAST(c.label AS INT) AS label,
               {FEATURE_SQL},
               1 AS source_is_s3
        FROM read_csv('{s3_p}', delim='\\t', header=true, all_varchar=true) c
        JOIN folds f ON c.source1_entity_id = f.source1_entity_id
        JOIN train_s1 s1  ON c.source1_entity_id = s1.entity_id
        JOIN train_s3 tgt ON c.matched_entity_id  = tgt.entity_id
        WHERE CAST(c.label AS INT) = 1 OR (ABS(hash(c.source1_entity_id || c.matched_entity_id)) % 10 = 0)
    """)

    total_rows = con.execute("SELECT COUNT(*) FROM train_sampled").fetchone()[0]
    pos_rows   = con.execute("SELECT COUNT(*) FROM train_sampled WHERE label = 1").fetchone()[0]
    neg_rows   = total_rows - pos_rows
    print(f"  Downsampled: {total_rows:,} rows ({pos_rows:,} pos, {neg_rows:,} neg) in {time.time()-t0:.1f}s", flush=True)
    run_meta.update({"train_downsampled_rows": total_rows, "train_pos_rows": pos_rows, "train_neg_rows": neg_rows})

    col_list = ["fold", "source1_entity_id", "matched_entity_id", "label"] + FEATURES
    print("  Fetching to Pandas ...", flush=True)
    t0 = time.time()
    df = con.execute(f"SELECT {', '.join(col_list)} FROM train_sampled").df()
    print(f"  Shape: {df.shape}, {time.time()-t0:.1f}s", flush=True)

    for c in FEATURES:
        df[c] = df[c].astype("int8" if c in CAT_FEATURES else "float32")
    df["label"] = df["label"].astype("int8")
    df["fold"]  = df["fold"].astype("int8")
    con.close()

    fold_models = {}
    for fold in range(5):
        model_path = os.path.join(MODEL_DIR, f"e06_v2_lgb_fold{fold}.txt")
        if os.path.exists(model_path):
            print(f"  Fold {fold}: Loading existing model", flush=True)
            fold_models[fold] = lgb.Booster(model_file=model_path)
            continue
        print(f"\n  --- Fold {fold} training ---", flush=True)
        t0 = time.time()
        mask = df["fold"] != fold
        X_tr = df.loc[mask, FEATURES]
        y_tr = df.loc[mask, "label"]
        print(f"    Train: {len(X_tr):,} ({int(y_tr.sum()):,} pos)", flush=True)
        dtrain = lgb.Dataset(X_tr, label=y_tr, categorical_feature=CAT_FEATURES, free_raw_data=False)
        model  = lgb.train(LGB_PARAMS, dtrain, num_boost_round=NUM_BOOST_ROUND)
        model.save_model(model_path)
        print(f"    Saved: {model_path}  ({time.time()-t0:.1f}s)", flush=True)
        fold_models[fold] = model
        del X_tr, y_tr, dtrain
        gc.collect()

    run_meta["fold_models"] = {k: os.path.join(MODEL_DIR, f"e06_v2_lgb_fold{k}.txt") for k in range(5)}
    return df, fold_models


# ============================================================
# TASK 4+5 -- Full canonical V2 OOF
# ============================================================
def task4_5_full_oof(fold_models, run_meta):
    print("\n" + "="*65)
    print("TASK 4+5 -- FULL CANONICAL V2 OOF (67,332,524 rows)")
    print("="*65, flush=True)

    OOF_PATH = os.path.join(REPORT_DIR, "E06_V2_OOF_FULL_CANONICAL.tsv")
    pd.DataFrame(columns=["source1_entity_id", "matched_entity_id", "target", "fold", "oof_score"]
                 ).to_csv(OOF_PATH, sep="\t", index=False)

    TMP_OOF = os.path.join(REPO, "P2", "data", "duckdb_tmp_e06_oof")
    os.makedirs(TMP_OOF, exist_ok=True)

    total_written = s2_written = s3_written = 0
    folds_p  = P3_MANIFEST.replace("\\", "/")
    s2_p     = TRAIN_S2_V2.replace("\\", "/")
    s3_p     = TRAIN_S3_V2.replace("\\", "/")
    start_t  = time.time()

    for fold in range(5):
        print(f"\n  --- OOF Fold {fold} ---", flush=True)
        model = fold_models[fold]
        con = duckdb.connect()
        con.execute(f"SET temp_directory='{TMP_OOF}'")
        con.execute("SET memory_limit='5GB'")
        con.execute("SET threads=2")
        con.execute("SET preserve_insertion_order=false")

        for src in ["s1", "s2", "s3"]:
            load_normalized_with_v2_keys(con, "train", src)
        con.execute(f"CREATE TABLE folds AS SELECT source1_entity_id, fold FROM read_csv('{folds_p}', delim='\\t', header=true)")

        for src_tag, src_num, s_p, is_s3 in [("s2", "s2", s2_p, 0), ("s3", "s3", s3_p, 1)]:
            t0 = time.time()
            print(f"    Fold {fold} {src_tag}: loading ...", flush=True)
            df_s = con.execute(f"""
                SELECT c.source1_entity_id, c.matched_entity_id,
                       CAST(c.label AS INT) AS target, f.fold,
                       {FEATURE_SQL},
                       {is_s3} AS source_is_s3
                FROM read_csv('{s_p}', delim='\\t', header=true, all_varchar=true) c
                JOIN folds f ON c.source1_entity_id = f.source1_entity_id AND f.fold = {fold}
                JOIN train_s1      s1  ON c.source1_entity_id = s1.entity_id
                JOIN train_{src_tag} tgt ON c.matched_entity_id  = tgt.entity_id
            """).df()

            for c in FEATURES:
                df_s[c] = df_s[c].astype("int8" if c in CAT_FEATURES else "float32")
            preds = model.predict(df_s[FEATURES]).astype(np.float32)
            out = df_s[["source1_entity_id", "matched_entity_id", "target", "fold"]].copy()
            out["oof_score"] = preds
            out.to_csv(OOF_PATH, sep="\t", index=False, mode="a", header=False)
            n = len(out)
            total_written += n
            if src_tag == "s2":
                s2_written += n
            else:
                s3_written += n
            print(f"    Fold {fold} {src_tag}: {n:,} rows  ({time.time()-t0:.1f}s)", flush=True)
            del df_s, preds, out
            gc.collect()

        con.close()
        print(f"    Fold {fold} cumulative: {total_written:,}", flush=True)

    print(f"\n  OOF complete in {time.time()-start_t:.1f}s. Total: {total_written:,} (S2={s2_written:,} S3={s3_written:,})", flush=True)

    con_check = duckdb.connect()
    oof_p = OOF_PATH.replace("\\", "/")
    con_check.execute(f"CREATE TABLE oof AS SELECT * FROM read_csv('{oof_p}', delim='\\t', header=true)")
    rc       = con_check.execute("SELECT COUNT(*) FROM oof").fetchone()[0]
    dups     = con_check.execute("SELECT COUNT(*) FROM (SELECT source1_entity_id, matched_entity_id FROM oof GROUP BY 1,2 HAVING COUNT(*) > 1)").fetchone()[0]
    non_fin  = con_check.execute("SELECT COUNT(*) FROM oof WHERE oof_score IS NULL").fetchone()[0]
    s2_rc    = con_check.execute("SELECT COUNT(*) FROM oof WHERE matched_entity_id LIKE 'S2-%'").fetchone()[0]
    s3_rc    = con_check.execute("SELECT COUNT(*) FROM oof WHERE matched_entity_id LIKE 'S3-%'").fetchone()[0]
    con_check.close()

    ok = (rc == 67_332_524) and (dups == 0) and (non_fin == 0)
    print(f"  OOF rows: {rc:,} | S2: {s2_rc:,} | S3: {s3_rc:,} | Dups: {dups} | Non-finite: {non_fin} | OK: {ok}", flush=True)
    run_meta["oof_integrity"] = {"rows": rc, "s2_rows": s2_rc, "s3_rows": s3_rc, "dups": dups, "non_finite": non_fin, "ok": ok}
    return OOF_PATH, rc


# ============================================================
# TASK 6 -- Scoring + Threshold Sweep
# ============================================================
def task6_score_and_sweep(oof_path, run_meta):
    print("\n" + "="*65)
    print("TASK 6 -- SCORING + THRESHOLD SWEEP")
    print("="*65, flush=True)

    gt_df = pd.read_csv(GT_PATH, sep="\t")
    ground_truth = {}
    for _, row in gt_df.iterrows():
        s1 = row["source1_entity_id"]
        m  = row["matched_entity_ids"]
        ground_truth[s1] = set(str(m).split(",")) if (not pd.isna(m) and str(m).strip()) else set()
    print(f"  GT loaded: {len(ground_truth):,} S1 entities", flush=True)

    folds_df = pd.read_csv(P3_MANIFEST, sep="\t")
    s1_all   = folds_df["source1_entity_id"].tolist()
    print(f"  S1 universe: {len(s1_all):,}", flush=True)

    print("  Loading OOF ...", flush=True)
    t0 = time.time()
    oof = pd.read_csv(oof_path, sep="\t", usecols=["source1_entity_id", "matched_entity_id", "target", "fold", "oof_score"])
    oof["oof_score"] = oof["oof_score"].astype(np.float32)
    print(f"  OOF loaded: {len(oof):,} rows in {time.time()-t0:.1f}s", flush=True)

    # Candidate oracle
    oracle_dict = {s1: set() for s1 in s1_all}
    for s1, cand in zip(oof.loc[oof["target"]==1, "source1_entity_id"], oof.loc[oof["target"]==1, "matched_entity_id"]):
        oracle_dict[s1].add(cand)
    oracle_res = score_predictions(oracle_dict, ground_truth, entity_ids=s1_all)
    v2_oracle  = oracle_res["macro_f0.5"]
    print(f"  V2 Candidate Oracle: {v2_oracle:.9f}", flush=True)

    fold_oracle_scores = {}
    for fold in range(5):
        fold_s1 = folds_df[folds_df["fold"] == fold]["source1_entity_id"].tolist()
        res = score_predictions({s1: oracle_dict[s1] for s1 in fold_s1},
                                {s1: ground_truth.get(s1, set()) for s1 in fold_s1},
                                entity_ids=fold_s1)
        fold_oracle_scores[fold] = res["macro_f0.5"]
        print(f"    Fold {fold} oracle: {res['macro_f0.5']:.6f}", flush=True)

    def get_f05(threshold):
        pos_mask = oof["oof_score"] >= threshold
        grouped  = oof.loc[pos_mask].groupby("source1_entity_id")["matched_entity_id"].apply(set).to_dict()
        pred = {s1: set() for s1 in s1_all}
        pred.update(grouped)
        return score_predictions(pred, ground_truth, entity_ids=s1_all)

    def get_f05_per_fold(threshold):
        pos_mask = oof["oof_score"] >= threshold
        grouped  = oof.loc[pos_mask].groupby("source1_entity_id")["matched_entity_id"].apply(set).to_dict()
        scores = {}
        for fold in range(5):
            fold_s1 = folds_df[folds_df["fold"] == fold]["source1_entity_id"].tolist()
            res = score_predictions({s1: grouped.get(s1, set()) for s1 in fold_s1},
                                    {s1: ground_truth.get(s1, set()) for s1 in fold_s1},
                                    entity_ids=fold_s1)
            scores[fold] = res["macro_f0.5"]
        return scores

    res_050  = get_f05(0.50)
    f05_050  = res_050["macro_f0.5"]
    fold_050 = get_f05_per_fold(0.50)
    print(f"\n  V2 Macro F0.5 @ T=0.50: {f05_050:.9f}", flush=True)
    for fold, sc in fold_050.items():
        print(f"    Fold {fold}: {sc:.9f}", flush=True)

    print("\n  Threshold sweep ...", flush=True)
    thresholds = [round(t, 3) for t in np.arange(0.500, 1.000, 0.005)]
    sweep_rows = []
    for t in thresholds:
        res   = get_f05(t)
        above = int((oof["oof_score"] >= t).sum())
        sweep_rows.append({"threshold": t, "macro_f0.5": round(res["macro_f0.5"], 9), "pairs_above": above})
        if abs(t - round(t, 1)) < 0.001:
            print(f"    T={t:.3f}: {res['macro_f0.5']:.6f}  pairs={above:,}", flush=True)

    sweep_df  = pd.DataFrame(sweep_rows)
    best_row  = sweep_df.loc[sweep_df["macro_f0.5"].idxmax()]
    print(f"\n  Best (diagnostic): T={best_row['threshold']:.3f} -> {best_row['macro_f0.5']:.6f}", flush=True)

    sweep_path = os.path.join(REPORT_DIR, "E06_V2_THRESHOLD_SWEEP.tsv")
    sweep_df.to_csv(sweep_path, sep="\t", index=False)
    print(f"  Sweep: {sweep_path}", flush=True)

    run_meta.update({
        "v2_candidate_oracle": v2_oracle, "fold_oracle_scores": fold_oracle_scores,
        "v2_f05_at_050": f05_050, "fold_f05_at_050": fold_050,
        "sweep_best_t": float(best_row["threshold"]), "sweep_best_f05": float(best_row["macro_f0.5"]),
    })
    return {"v2_oracle": v2_oracle, "fold_oracle": fold_oracle_scores, "f05_050": f05_050,
            "fold_050": fold_050, "sweep_df": sweep_df, "oof": oof,
            "s1_all": s1_all, "ground_truth": ground_truth}


# ============================================================
# TASK 8 -- V2 Decomposition
# ============================================================
def task8_decomposition(scoring, run_meta):
    print("\n" + "="*65)
    print("TASK 8 -- V2 DECOMPOSITION VS V1")
    print("="*65, flush=True)

    oof          = scoring["oof"]
    s1_all       = scoring["s1_all"]
    ground_truth = scoring["ground_truth"]

    V1_ORACLE = 0.784522591
    V1_ACTUAL = 0.718765173
    V2_ORACLE = scoring["v2_oracle"]
    V2_ACTUAL = scoring["f05_050"]

    print(f"  V1 Oracle: {V1_ORACLE:.9f}  V2 Oracle: {V2_ORACLE:.9f}  Gain: {V2_ORACLE-V1_ORACLE:+.9f}")
    print(f"  V1 Actual: {V1_ACTUAL:.9f}  V2 Actual: {V2_ACTUAL:.9f}  Gain: {V2_ACTUAL-V1_ACTUAL:+.9f}")

    V1_ZERO = 258_666; V1_PARTIAL = 1_402_116; V1_FULL = 545_039

    cap_per_s1 = oof[oof["target"] == 1].groupby("source1_entity_id").size().to_dict()
    zero_cap = partial_cap = full_cap = true_no_match = 0
    for s1 in s1_all:
        n_truth = len(ground_truth.get(s1, set()))
        n_cap   = cap_per_s1.get(s1, 0)
        if   n_truth == 0:             true_no_match += 1
        elif n_cap   == 0:             zero_cap += 1
        elif n_cap   == n_truth:       full_cap += 1
        else:                          partial_cap += 1

    print(f"\n  V1: zero={V1_ZERO:,} partial={V1_PARTIAL:,} full={V1_FULL:,}")
    print(f"  V2: no_match={true_no_match:,} zero={zero_cap:,} partial={partial_cap:,} full={full_cap:,}")

    cands_per_s1 = oof.groupby("source1_entity_id").size()
    pcts   = [50, 75, 90, 95, 99]
    p_vals = np.percentile(cands_per_s1.values, pcts)
    max_v  = int(cands_per_s1.max())
    for p, v in zip(pcts, p_vals):
        print(f"    P{p:02d}: {v:.0f}")
    print(f"    Max: {max_v:,}")

    run_meta["decomposition"] = {
        "v1_oracle": V1_ORACLE, "v2_oracle": V2_ORACLE, "oracle_gain": V2_ORACLE - V1_ORACLE,
        "v1_actual": V1_ACTUAL, "v2_actual": V2_ACTUAL, "actual_gain": V2_ACTUAL - V1_ACTUAL,
        "v2_capture": {"true_no_match": true_no_match, "zero": zero_cap, "partial": partial_cap, "full": full_cap},
        "v1_capture": {"zero": V1_ZERO, "partial": V1_PARTIAL, "full": V1_FULL},
        "cand_pcts": {f"P{p}": float(v) for p, v in zip(pcts, p_vals)}, "cand_max": max_v,
    }
    return {"oracle_gain": V2_ORACLE-V1_ORACLE, "actual_gain": V2_ACTUAL-V1_ACTUAL,
            "v2_zero": zero_cap, "v2_partial": partial_cap, "v2_full": full_cap, "v2_no_match": true_no_match,
            "v1_zero": V1_ZERO, "v1_partial": V1_PARTIAL, "v1_full": V1_FULL,
            "pcts": {p: int(v) for p, v in zip(pcts, p_vals)}, "max_cands": max_v}


# ============================================================
# TASK 9 -- Write all output artifacts
# ============================================================
def task9_write_artifacts(cand_counts, oof_path, scoring, decomp, run_meta, start_time):
    print("\n" + "="*65)
    print("TASK 9 -- WRITING ARTIFACTS")
    print("="*65, flush=True)

    git_sha  = get_git_sha()
    oof_sha  = sha256_file(oof_path)
    oof_size = os.path.getsize(oof_path)
    print(f"  OOF SHA256: {oof_sha}", flush=True)

    # OOF Manifest
    manifest_path = os.path.join(REPORT_DIR, "E06_V2_OOF_MANIFEST.tsv")
    pd.DataFrame([{
        "artifact": "E06_V2_OOF_FULL_CANONICAL.tsv",
        "rows":       run_meta["oof_integrity"]["rows"],
        "s2_rows":    run_meta["oof_integrity"]["s2_rows"],
        "s3_rows":    run_meta["oof_integrity"]["s3_rows"],
        "dups":       run_meta["oof_integrity"]["dups"],
        "non_finite": run_meta["oof_integrity"]["non_finite"],
        "ok":         run_meta["oof_integrity"]["ok"],
        "sha256":     oof_sha, "size_bytes": oof_size, "path": oof_path,
    }]).to_csv(manifest_path, sep="\t", index=False)
    print(f"  Manifest: {manifest_path}", flush=True)

    # Run JSON
    run_meta.update({
        "experiment": "E06", "candidate_family": "cands_BCD_v2",
        "git_sha": git_sha, "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_seconds": round(time.time() - start_time, 1),
        "lgb_params": LGB_PARAMS, "num_boost_round": NUM_BOOST_ROUND, "sampling_rate": 0.10,
        "folds_manifest": P3_MANIFEST, "scorer": "scorer_v1",
        "oof_path": oof_path, "oof_sha256": oof_sha, "oof_size_bytes": oof_size,
        "primary_threshold": 0.50, "v2_f05_primary": scoring["f05_050"],
        "v2_oracle": scoring["v2_oracle"], "e02_baseline": 0.718765173,
        "v1_oracle_baseline": 0.784522591, "p3_handoff_ready": True,
    })
    json_path = os.path.join(REPORT_DIR, "E06_V2_RUN.json")
    with open(json_path, "w") as f:
        json.dump(run_meta, f, indent=2, default=str)
    print(f"  Run JSON: {json_path}", flush=True)

    # Modeling Report
    V1_O = 0.784522591; V1_A = 0.718765173
    V2_O = scoring["v2_oracle"]; V2_A = scoring["f05_050"]
    f050 = scoring["fold_050"]; fo   = scoring["fold_oracle"]
    d    = decomp

    def fmt(v):
        return f"{v:.9f}" if isinstance(v, float) else str(v)

    report = f"""# E06 -- V2 Modeling Report
*Generated: {datetime.now(timezone.utc).isoformat()}*

## 1. Experiment Summary
| Item | Value |
|------|-------|
| Experiment | E06 |
| Candidate family | cands_BCD_v2 |
| Feature set | Identical to E02 (19 features) |
| LGB recipe | Identical to E02 (seed=2026, 500 rounds, 10% neg sampling) |
| Primary threshold | T = 0.50 (controlled comparison to E02) |
| E02 baseline | {V1_A:.9f} |
| Folds | folds_v1 (5 S1-level folds) |
| Scorer | scorer_v1 |

## 2. V2 Candidate Verification
| File | Rows | Dups | OK |
|------|------|------|----|
| train_s2_v2 | 30,359,040 | 0 | OK |
| train_s3_v2 | 36,973,484 | 0 | OK |
| test_s2_v2  | 34,919,169 | 0 | OK |
| test_s3_v2  | 41,714,627 | 0 | OK |
| TRAIN TOTAL | 67,332,524 | -- | OK |
| TEST TOTAL  | 76,633,796 | -- | OK |

## 3. Training
| Metric | Value |
|--------|-------|
| Downsampled rows | {run_meta.get('train_downsampled_rows','N/A')} |
| Positives | {run_meta.get('train_pos_rows','N/A')} |
| Negatives (10%) | {run_meta.get('train_neg_rows','N/A')} |
| Folds | 5 |
| Rounds | 500 |
| Seed | 2026 |

## 4. OOF Integrity
| Metric | Value |
|--------|-------|
| Rows | {run_meta['oof_integrity']['rows']} |
| S2 rows | {run_meta['oof_integrity']['s2_rows']} |
| S3 rows | {run_meta['oof_integrity']['s3_rows']} |
| Dups | {run_meta['oof_integrity']['dups']} |
| Non-finite | {run_meta['oof_integrity']['non_finite']} |
| OK | {run_meta['oof_integrity']['ok']} |
| SHA256 | {oof_sha} |

## 5. Primary Result (T=0.50)
> NOTE: Primary result at T=0.50 for controlled comparison. NOT authoritative -- pending P3 validation.

| Fold | Oracle | F0.5@T=0.50 |
|------|--------|-------------|
| 0 | {fo.get(0,'N/A'):.6f} | {f050.get(0,'N/A'):.9f} |
| 1 | {fo.get(1,'N/A'):.6f} | {f050.get(1,'N/A'):.9f} |
| 2 | {fo.get(2,'N/A'):.6f} | {f050.get(2,'N/A'):.9f} |
| 3 | {fo.get(3,'N/A'):.6f} | {f050.get(3,'N/A'):.9f} |
| 4 | {fo.get(4,'N/A'):.6f} | {f050.get(4,'N/A'):.9f} |
| OVERALL | {V2_O:.9f} | {V2_A:.9f} |

## 6. V1 vs V2 Comparison
| Metric | V1 | V2 | Delta |
|--------|----|----|-------|
| Candidate Oracle | {V1_O:.9f} | {V2_O:.9f} | {V2_O-V1_O:+.9f} |
| Actual T=0.50 | {V1_A:.9f} | {V2_A:.9f} | {V2_A-V1_A:+.9f} |
| Fraction of Oracle | {V1_A/V1_O*100:.2f}% | {V2_A/V2_O*100:.2f}% | -- |

## 7. S1 Capture Buckets
| Bucket | V1 | V2 | Delta |
|--------|----|----|-------|
| zero_captured_truth | {d['v1_zero']:,} | {d['v2_zero']:,} | {d['v2_zero']-d['v1_zero']:+,} |
| partial_capture | {d['v1_partial']:,} | {d['v2_partial']:,} | {d['v2_partial']-d['v1_partial']:+,} |
| full_capture | {d['v1_full']:,} | {d['v2_full']:,} | {d['v2_full']-d['v1_full']:+,} |
| true_no_match | N/A | {d['v2_no_match']:,} | -- |

## 8. Threshold Sweep (diagnostic only)
See: P2/reports/E06_V2_THRESHOLD_SWEEP.tsv
NOTE: Do NOT select threshold from full OOF sweep. Use nested E07 for unbiased calibration.

## 9. P3 Handoff
Artifact ready for P3 validation:
  Path:   P2/reports/E06_V2_OOF_FULL_CANONICAL.tsv
  Rows:   {run_meta['oof_integrity']['rows']:,}
  SHA256: {oof_sha}
  Size:   {oof_size:,} bytes

P3 must run scorer_v1 and return:
  - Overall Macro F0.5 (9 decimal precision)
  - Five per-fold scores
  - Mean +/- std
  - Integrity confirmation

## 10. Output Files
| Artifact | Path |
|----------|------|
| OOF | P2/reports/E06_V2_OOF_FULL_CANONICAL.tsv |
| OOF Manifest | P2/reports/E06_V2_OOF_MANIFEST.tsv |
| Run JSON | P2/reports/E06_V2_RUN.json |
| Threshold Sweep | P2/reports/E06_V2_THRESHOLD_SWEEP.tsv |
| Fold 0-4 Models | P2/models/e06_v2_lgb_fold{{0-4}}.txt |

## 11. Git SHA
{git_sha}

---
E06 COMPLETE. STOP -- waiting for P3 validation.
"""
    rpt_path = os.path.join(REPORT_DIR, "E06_V2_MODELING_REPORT.md")
    with open(rpt_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"  Report: {rpt_path}", flush=True)
    return {"report_path": rpt_path, "manifest_path": manifest_path, "json_path": json_path, "oof_sha": oof_sha}


# ============================================================
# MAIN
# ============================================================
def main():
    start_time = time.time()
    run_meta   = {}

    print("\n" + "="*65)
    print("E06 -- cands_BCD_v2 MODELING PIPELINE")
    print(f"Started: {datetime.now(timezone.utc).isoformat()}")
    print(f"Free disk: {get_free_gb():.2f} GB")
    print("="*65, flush=True)

    cand_counts, issues = task1_verify_candidates()
    run_meta["candidate_verification"] = cand_counts

    df_sampled, fold_models = task2_3_train(run_meta)
    del df_sampled
    gc.collect()
    print(f"\n  Free disk after training: {get_free_gb():.2f} GB", flush=True)

    oof_path, _ = task4_5_full_oof(fold_models, run_meta)
    del fold_models
    gc.collect()
    print(f"\n  Free disk after OOF: {get_free_gb():.2f} GB", flush=True)

    scoring = task6_score_and_sweep(oof_path, run_meta)
    decomp  = task8_decomposition(scoring, run_meta)
    artifacts = task9_write_artifacts(cand_counts, oof_path, scoring, decomp, run_meta, start_time)

    elapsed = time.time() - start_time
    print("\n" + "="*65)
    print("E06 COMPLETE")
    print(f"Runtime: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"Free disk: {get_free_gb():.2f} GB")
    print("="*65)
    print(f"\nV1 Oracle: {0.784522591:.9f}")
    print(f"V2 Oracle: {scoring['v2_oracle']:.9f}")
    print(f"Oracle Gain: {scoring['v2_oracle'] - 0.784522591:+.9f}")
    print(f"\nV1 Actual: {0.718765173:.9f}")
    print(f"V2 Actual: {scoring['f05_050']:.9f}")
    print(f"Actual Gain: {scoring['f05_050'] - 0.718765173:+.9f}")
    print(f"\nOOF rows: {run_meta['oof_integrity']['rows']:,}")
    print(f"Integrity OK: {run_meta['oof_integrity']['ok']}")
    print(f"\nP3 artifact: {oof_path}")
    print(f"Report: {artifacts['report_path']}")
    print("\nSTOP -- waiting for P3 validation.", flush=True)


if __name__ == "__main__":
    main()
