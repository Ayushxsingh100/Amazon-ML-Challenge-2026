#!/usr/bin/env python3
"""
E06 Test-Time Inference Pipeline
================================
- Evaluates 5-fold E06 LightGBM models on the complete V2 test candidate universe:
    S2: 34,919,169 candidate pairs
    S3: 41,714,627 candidate pairs
    Total: 76,633,796 candidate pairs
- Exact 19 features computed in-memory via DuckDB SQL (identical to E06 training)
- 5-fold ensemble probability mean: score = (p0 + p1 + p2 + p3 + p4) / 5.0
- Fixed threshold: T = 0.50
- Outputs:
    output/candidate_pairs.tsv  (Complete V2 candidate universe, all 1,732,544 S1 entities)
    output/matching_results.tsv (Predicted matches at T=0.50, all 1,732,544 S1 entities)
"""

import gc
import json
import os
import shutil
import sys
import time

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
NORM_DIR = os.path.join(REPO, "outputs", "person1_step1", "normalized")
CAND_DIR = os.path.join(REPO, "P2", "data", "candidates")
MODEL_DIR = os.path.join(REPO, "P2", "models")
OUTPUT_DIR = os.path.join(REPO, "output")
REPORT_DIR = os.path.join(REPO, "P2", "reports")
TMP_DIR = os.path.join(REPO, "P2", "data", "duckdb_tmp_test_inference")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(TMP_DIR, exist_ok=True)

TEST_S2_CAND = os.path.join(CAND_DIR, "test_candidate_pairs_s2_v2.tsv")
TEST_S3_CAND = os.path.join(CAND_DIR, "test_candidate_pairs_s3_v2.tsv")

OUTPUT_CANDIDATES = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")
OUTPUT_MATCHING = os.path.join(OUTPUT_DIR, "matching_results.tsv")

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


def load_models():
    models = []
    print("Loading 5-fold E06 LightGBM models...", flush=True)
    for fold in range(5):
        m_path = os.path.join(MODEL_DIR, f"e06_v2_lgb_fold{fold}.txt")
        if not os.path.exists(m_path):
            raise FileNotFoundError(f"Model file not found: {m_path}")
        booster = lgb.Booster(model_file=m_path)
        models.append(booster)
        print(f"  Fold {fold} loaded from {m_path}", flush=True)
    return models


def load_normalized_table(con, src):
    num = src[-1]
    path = os.path.join(NORM_DIR, f"test_source{num}_normalized.tsv").replace("\\", "/")
    con.execute(f"""
        CREATE OR REPLACE TABLE test_{src} AS
        SELECT entity_id, business_name, business_address, country
        FROM read_csv('{path}', delim='\\t', header=true, all_varchar=true)
    """)
    cnt = con.execute(f"SELECT COUNT(*) FROM test_{src}").fetchone()[0]
    print(f"  Loaded test_{src}: {cnt:,} entities", flush=True)
    return cnt


def main():
    t_start = time.time()
    print("=" * 70)
    print("E06 TEST INFERENCE & SUBMISSION GENERATION")
    print("=" * 70, flush=True)

    models = load_models()

    # Temporary files for positive matches
    TMP_MATCHES_S2 = os.path.join(TMP_DIR, "tmp_matches_s2.tsv")
    TMP_MATCHES_S3 = os.path.join(TMP_DIR, "tmp_matches_s3.tsv")
    if os.path.exists(TMP_MATCHES_S2):
        os.remove(TMP_MATCHES_S2)
    if os.path.exists(TMP_MATCHES_S3):
        os.remove(TMP_MATCHES_S3)

    # Initialize match files with headers
    pd.DataFrame(columns=["source1_entity_id", "matched_entity_id"]).to_csv(TMP_MATCHES_S2, sep="\t", index=False)
    pd.DataFrame(columns=["source1_entity_id", "matched_entity_id"]).to_csv(TMP_MATCHES_S3, sep="\t", index=False)

    THRESHOLD = 0.50
    BATCH_SIZE = 1000000

    score_histogram = {"<0.1": 0, "0.1-0.3": 0, "0.3-0.5": 0, "0.5-0.7": 0, "0.7-0.9": 0, ">=0.9": 0}
    total_evaluated = 0
    total_matched = 0

    # -------------------------------------------------------------
    # S2 & S3 Test Candidate Inference
    # -------------------------------------------------------------
    for src_name, cand_file, is_s3, tmp_out in [
        ("s2", TEST_S2_CAND, 0, TMP_MATCHES_S2),
        ("s3", TEST_S3_CAND, 1, TMP_MATCHES_S3),
    ]:
        print(f"\n{'='*50}", flush=True)
        print(f"Processing Test Candidates: {src_name.upper()} ({cand_file})", flush=True)
        print(f"{'='*50}", flush=True)

        con = duckdb.connect()
        con.execute(f"SET temp_directory='{TMP_DIR.replace(os.sep, '/')}'")
        con.execute("SET memory_limit='5GB'")
        con.execute("SET threads=4")
        con.execute("SET preserve_insertion_order=false")

        load_normalized_table(con, "s1")
        load_normalized_table(con, src_name)

        cand_clean = cand_file.replace("\\", "/")
        print(f"  Streaming feature query from {cand_clean} ...", flush=True)

        query = f"""
            SELECT c.source1_entity_id, c.matched_entity_id,
                   {FEATURE_SQL},
                   {is_s3} AS source_is_s3
            FROM read_csv('{cand_clean}', delim='\\t', header=true, all_varchar=true) c
            JOIN test_s1 s1 ON c.source1_entity_id = s1.entity_id
            JOIN test_{src_name} tgt ON c.matched_entity_id = tgt.entity_id
        """

        reader = con.sql(query).to_arrow_reader(batch_size=BATCH_SIZE)
        src_evaluated = 0
        src_matched = 0
        batch_idx = 0

        for batch in reader:
            t_batch_start = time.time()
            df_batch = batch.to_pandas()
            batch_len = len(df_batch)
            src_evaluated += batch_len
            batch_idx += 1

            # Cast data types to match E06 exact specifications
            for c in FEATURES:
                df_batch[c] = df_batch[c].astype("int8" if c in CAT_FEATURES else "float32")

            X = df_batch[FEATURES]

            # 5-fold ensemble prediction
            preds = np.zeros(batch_len, dtype=np.float32)
            for m in models:
                preds += m.predict(X).astype(np.float32)
            preds /= 5.0

            # Integrity checks: no NaNs / infs
            assert not np.isnan(preds).any(), f"NaN found in predictions for batch {batch_idx}"
            assert not np.isinf(preds).any(), f"Inf found in predictions for batch {batch_idx}"

            # Update score distribution histogram
            score_histogram["<0.1"]   += int((preds < 0.1).sum())
            score_histogram["0.1-0.3"] += int(((preds >= 0.1) & (preds < 0.3)).sum())
            score_histogram["0.3-0.5"] += int(((preds >= 0.3) & (preds < 0.5)).sum())
            score_histogram["0.5-0.7"] += int(((preds >= 0.5) & (preds < 0.7)).sum())
            score_histogram["0.7-0.9"] += int(((preds >= 0.7) & (preds < 0.9)).sum())
            score_histogram[">=0.9"]   += int((preds >= 0.9).sum())

            # Filter matches >= THRESHOLD
            mask = preds >= THRESHOLD
            matches_df = df_batch.loc[mask, ["source1_entity_id", "matched_entity_id"]]
            n_match = len(matches_df)
            src_matched += n_match

            if n_match > 0:
                matches_df.to_csv(tmp_out, sep="\t", index=False, mode="a", header=False)

            total_evaluated += batch_len
            total_matched += n_match
            print(
                f"  Batch {batch_idx} ({batch_len:,} pairs) | "
                f"Matches (T>={THRESHOLD}): {n_match:,} ({n_match/batch_len*100:.2f}%) | "
                f"Evaluated: {src_evaluated:,} | "
                f"Cum {src_name.upper()} Matches: {src_matched:,} | "
                f"Time: {time.time()-t_batch_start:.1f}s",
                flush=True,
            )

            del df_batch, X, preds, mask, matches_df
            gc.collect()

        con.close()
        print(f"  Completed {src_name.upper()} inference: {src_evaluated:,} evaluated, {src_matched:,} matched.", flush=True)

    print(f"\nTotal Candidates Evaluated: {total_evaluated:,}", flush=True)
    print(f"Total Matches Passing Threshold ({THRESHOLD}): {total_matched:,}", flush=True)

    # -------------------------------------------------------------
    # TASK 3 — CREATE FINAL CANDIDATE FILE (output/candidate_pairs.tsv)
    # -------------------------------------------------------------
    print("\n" + "="*70, flush=True)
    print("TASK 3 — BUILDING output/candidate_pairs.tsv", flush=True)
    print("="*70, flush=True)
    t_cands = time.time()

    con_out = duckdb.connect()
    con_out.execute(f"SET temp_directory='{TMP_DIR.replace(os.sep, '/')}'")
    con_out.execute("SET memory_limit='6GB'")
    con_out.execute("SET threads=4")
    con_out.execute("SET preserve_insertion_order=false")

    norm_s1_path = os.path.join(NORM_DIR, "test_source1_normalized.tsv").replace("\\", "/")
    con_out.execute(f"""
        CREATE TABLE test_s1 AS
        SELECT entity_id AS source1_entity_id
        FROM read_csv('{norm_s1_path}', delim='\\t', header=true, all_varchar=true)
    """)
    test_s1_count = con_out.execute("SELECT COUNT(*) FROM test_s1").fetchone()[0]
    print(f"  Test S1 Universe: {test_s1_count:,} entities", flush=True)

    s2_cand_clean = TEST_S2_CAND.replace("\\", "/")
    s3_cand_clean = TEST_S3_CAND.replace("\\", "/")

    print("  Loading all test candidate pairs into unified table...", flush=True)
    con_out.execute(f"""
        CREATE TABLE all_cands AS
        SELECT source1_entity_id, matched_entity_id
        FROM read_csv('{s2_cand_clean}', delim='\\t', header=true, all_varchar=true)
        UNION ALL
        SELECT source1_entity_id, matched_entity_id
        FROM read_csv('{s3_cand_clean}', delim='\\t', header=true, all_varchar=true)
    """)
    all_cands_count = con_out.execute("SELECT COUNT(*) FROM all_cands").fetchone()[0]
    print(f"  Total raw candidate pairs: {all_cands_count:,}", flush=True)
    assert all_cands_count == 76633796, f"Expected 76,633,796 candidates, got {all_cands_count:,}"

    print("  Aggregating candidates per Source 1 entity...", flush=True)
    con_out.execute("""
        CREATE TABLE cands_grouped AS
        SELECT source1_entity_id,
               string_agg(matched_entity_id, ',' ORDER BY matched_entity_id) AS candidate_entity_ids
        FROM all_cands
        GROUP BY source1_entity_id
    """)

    out_cands_clean = OUTPUT_CANDIDATES.replace("\\", "/")
    print(f"  Exporting to {OUTPUT_CANDIDATES} ...", flush=True)
    con_out.execute(f"""
        COPY (
            SELECT s1.source1_entity_id,
                   COALESCE(g.candidate_entity_ids, '') AS candidate_entity_ids
            FROM test_s1 s1
            LEFT JOIN cands_grouped g ON s1.source1_entity_id = g.source1_entity_id
            ORDER BY s1.source1_entity_id
        ) TO '{out_cands_clean}' WITH (FORMAT CSV, DELIMITER '\\t', HEADER)
    """)
    size_cands_mb = os.path.getsize(OUTPUT_CANDIDATES) / (1024 * 1024)
    print(f"  Created candidate_pairs.tsv ({size_cands_mb:.1f} MB) in {time.time()-t_cands:.1f}s", flush=True)

    # -------------------------------------------------------------
    # TASK 4 — CREATE MATCHING RESULTS (output/matching_results.tsv)
    # -------------------------------------------------------------
    print("\n" + "="*70, flush=True)
    print("TASK 4 — BUILDING output/matching_results.tsv", flush=True)
    print("="*70, flush=True)
    t_match = time.time()

    tmp_m_s2_clean = TMP_MATCHES_S2.replace("\\", "/")
    tmp_m_s3_clean = TMP_MATCHES_S3.replace("\\", "/")

    print("  Loading predicted matches...", flush=True)
    con_out.execute(f"""
        CREATE TABLE all_matches AS
        SELECT source1_entity_id, matched_entity_id
        FROM read_csv('{tmp_m_s2_clean}', delim='\\t', header=true, all_varchar=true)
        UNION ALL
        SELECT source1_entity_id, matched_entity_id
        FROM read_csv('{tmp_m_s3_clean}', delim='\\t', header=true, all_varchar=true)
    """)
    all_matches_count = con_out.execute("SELECT COUNT(*) FROM all_matches").fetchone()[0]
    print(f"  Total predicted match pairs: {all_matches_count:,}", flush=True)

    print("  Aggregating matches per Source 1 entity...", flush=True)
    con_out.execute("""
        CREATE TABLE matches_grouped AS
        SELECT source1_entity_id,
               string_agg(matched_entity_id, ',' ORDER BY matched_entity_id) AS matched_entity_ids
        FROM all_matches
        GROUP BY source1_entity_id
    """)

    out_match_clean = OUTPUT_MATCHING.replace("\\", "/")
    print(f"  Exporting to {OUTPUT_MATCHING} ...", flush=True)
    con_out.execute(f"""
        COPY (
            SELECT s1.source1_entity_id,
                   COALESCE(m.matched_entity_ids, '') AS matched_entity_ids
            FROM test_s1 s1
            LEFT JOIN matches_grouped m ON s1.source1_entity_id = m.source1_entity_id
            ORDER BY s1.source1_entity_id
        ) TO '{out_match_clean}' WITH (FORMAT CSV, DELIMITER '\\t', HEADER)
    """)
    size_match_mb = os.path.getsize(OUTPUT_MATCHING) / (1024 * 1024)
    print(f"  Created matching_results.tsv ({size_match_mb:.1f} MB) in {time.time()-t_match:.1f}s", flush=True)

    con_out.close()

    # Clean up temp files
    try:
        shutil.rmtree(TMP_DIR, ignore_errors=True)
    except Exception:
        pass

    total_runtime = time.time() - t_start
    print("\n" + "="*70, flush=True)
    print(f"TEST INFERENCE COMPLETED IN {total_runtime:.1f}s ({total_runtime/60:.2f} min)", flush=True)
    print(f"Score Distribution: {json.dumps(score_histogram, indent=2)}", flush=True)
    print("="*70, flush=True)


if __name__ == "__main__":
    main()
