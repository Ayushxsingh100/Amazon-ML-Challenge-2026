#!/usr/bin/env python3
"""
Full-Universe V3 Out-Of-Fold (OOF) Inference Pipeline
=====================================================
Executes full-universe inference over all 93,171,949 V3 training candidate pairs
using the 5 verified Phase 4 LightGBM fold models and P3 folds_v1 manifest.

Enforces:
- Fold-safe OOF rule: each S1 pair scored strictly by its held-out fold model
- Strict checkpointing: fold-by-fold streaming directly to ZSTD-compressed Parquet
- 100% full universe (no sampling, no negative downsampling, no score filtering)
- Complete mandatory assertions suite (A-L)
- Comprehensive diagnostics: Sample reproduction, 153-pair check, Full-universe threshold sweep,
  Score distribution, Entity-level diagnostics, Train/test parity, and Performance metrics.

Outputs:
  - E:/predictions/phase4/v3_train_oof_fold{0..4}.parquet
  - P2/manifests/V3_FULL_OOF_MANIFEST.tsv
  - P2/reports/V3_FULL_OOF_REPORT.json
  - P2/reports/v3_oof_progress.log
"""

import os
import sys
import gc
import time
import json
import shutil
import hashlib
import tempfile
import tracemalloc
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import lightgbm as lgb

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))
from validation.scorer_v1 import score_predictions

# Authoritative input paths
MODEL_DIR = REPO / "P2" / "models" / "phase4"
CAND_DIR = REPO / "P1" / "data" / "candidates" / "v3"
TRAIN_S2_CAND = CAND_DIR / "train_candidate_pairs_s2_v3.tsv"
TRAIN_S3_CAND = CAND_DIR / "train_candidate_pairs_s3_v3.tsv"
GT_PATH = REPO / "outputs" / "person1_step1" / "train_ground_truth_reconstructed.tsv"
FOLDS_PATH = REPO / "P3" / "reports" / "folds_v1_manifest.tsv"
S1_PATH = REPO / "outputs" / "person1_step1" / "normalized" / "train_source1_normalized.tsv"
S2_PARQ = REPO / "P1" / "data" / "entities" / "train" / "source2" / "train_s2_entities.parquet"
S3_PARQ = REPO / "P1" / "data" / "entities" / "train" / "source3" / "train_s3_entities.parquet"

# Output and temp directories on Drive E:
OUT_DIR = Path("E:/predictions/phase4")
OUT_DIR.mkdir(parents=True, exist_ok=True)
TEMP_DIR = Path("E:/duckdb_tmp_v3_oof")
TEMP_DIR.mkdir(parents=True, exist_ok=True)

REPORT_DIR = REPO / "P2" / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)
MANIFEST_DIR = REPO / "P2" / "manifests"
MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
PROGRESS_LOG = REPORT_DIR / "v3_oof_progress.log"

# Expected universe constants
EXPECTED_TOTAL_CANDIDATES = 93171949
EXPECTED_CAPTURED_POSITIVES = 5511986
EXPECTED_TOTAL_GT_POSITIVES = 7638365
EXPECTED_TRAIN_S1 = 2206821

EXPECTED_FOLD_ROWS = {
    0: 18557806,
    1: 18671223,
    2: 18613143,
    3: 18616336,
    4: 18713441,
}
EXPECTED_FOLD_POS = {
    0: 1100786,
    1: 1103649,
    2: 1103191,
    3: 1101707,
    4: 1102653,
}

CANONICAL_FEATURES = [
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

PARQUET_SCHEMA = pa.schema([
    ("source1_entity_id", pa.string()),
    ("candidate_entity_id", pa.string()),
    ("target_source", pa.string()),
    ("fold", pa.int8()),
    ("score", pa.float32()),
    ("label", pa.int8()),
])

def log_progress(msg: str):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    formatted = f"[{timestamp}] {msg}"
    print(formatted)
    try:
        with open(PROGRESS_LOG, "a", encoding="utf-8") as f:
            f.write(formatted + "\n")
    except Exception:
        pass

def sha256_file(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def load_lf_model(model_path: Path) -> lgb.Booster:
    with open(model_path, "r", encoding="utf-8") as f:
        text = f.read().replace("\r\n", "\n")
    with tempfile.NamedTemporaryFile("w", delete=False, newline="\n", suffix=".txt") as tmp:
        tmp.write(text)
        tmp_path = tmp.name
    try:
        bst = lgb.Booster(model_file=tmp_path)
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
    return bst

def main():
    t_global_start = time.time()
    tracemalloc.start()

    log_progress("=" * 70)
    log_progress("V3 FULL-UNIVERSE OUT-OF-FOLD (OOF) INFERENCE PIPELINE")
    log_progress("=" * 70)

    # 0. Check pre-conditions & disk space
    free_gb = shutil.disk_usage("E:/").free / (1024**3)
    log_progress(f"Checking environment: Output dir: {OUT_DIR}, Temp dir: {TEMP_DIR}")
    log_progress(f"Available free space on E: {free_gb:.2f} GB")
    assert free_gb >= 10.0, f"Insufficient disk space on E: {free_gb:.2f} GB"

    # 1. Load Models
    log_progress("\n[Step 1/6] Loading 5 Phase 4 LightGBM models with LF normalization...")
    models = []
    for f in range(5):
        mpath = MODEL_DIR / f"lgb_fold{f}.txt"
        bst = load_lf_model(mpath)
        log_progress(f"  Fold {f} Booster loaded: {bst.num_trees()} trees, {len(bst.feature_name())} features [OK]")
        models.append(bst)

    # 2. Setup DuckDB
    con = duckdb.connect()
    con.execute(f"SET temp_directory='{TEMP_DIR}'")
    con.execute("SET memory_limit='8GB'")
    con.execute("SET threads=8")
    con.execute("SET preserve_insertion_order=false")

    log_progress("\n[Step 2/6] Loading canonical entities, folds, and ground truth in DuckDB...")
    con.execute(f"""
        CREATE TABLE raw_gt AS
        SELECT 
            TRIM(source1_entity_id) AS source1_entity_id, 
            TRIM(UNNEST(string_split(matched_entity_ids, ','))) AS matched_entity_id
        FROM read_csv('{GT_PATH}', delim='\\t', header=true, all_varchar=true)
        WHERE matched_entity_ids IS NOT NULL AND TRIM(matched_entity_ids) <> '';
    """)

    con.execute(f"""
        CREATE TABLE folds AS
        SELECT source1_entity_id, fold
        FROM read_csv('{FOLDS_PATH}', delim='\\t', header=true);
    """)

    con.execute(f"""
        CREATE TABLE train_s1 AS 
        SELECT entity_id, business_name, business_address, country 
        FROM read_csv('{S1_PATH}', delim='\\t', header=true, all_varchar=true);
    """)

    con.execute(f"""
        CREATE TABLE train_s2 AS 
        SELECT entity_id, business_name_normalized as business_name, business_address_normalized as business_address, country_normalized as country 
        FROM read_parquet('{S2_PARQ}');
    """)

    con.execute(f"""
        CREATE TABLE train_s3 AS 
        SELECT entity_id, business_name_normalized as business_name, business_address_normalized as business_address, country_normalized as country 
        FROM read_parquet('{S3_PARQ}');
    """)
    log_progress("Canonical lookup tables ready in DuckDB.")

    # 3. Process Fold-by-Fold with Checkpointing and Streaming
    log_progress("\n[Step 3/6] Streaming Feature Extraction, Scoring & Parquet Generation by Fold...")
    
    oof_artifacts = []
    fold_stats = {}
    runtimes = {"feature_gen": 0.0, "scoring": 0.0, "writing": 0.0}

    for fold in range(5):
        log_progress(f"\n==================== PROCESSING FOLD {fold} ====================")
        t_fold_start = time.time()
        model = models[fold]
        out_parquet = OUT_DIR / f"v3_train_oof_fold{fold}.parquet"
        tmp_parquet = OUT_DIR / f"v3_train_oof_fold{fold}.parquet.tmp"
        
        expected_rows = EXPECTED_FOLD_ROWS[fold]
        expected_pos = EXPECTED_FOLD_POS[fold]

        # Checkpoint check: if already completed and valid, reuse
        if out_parquet.exists():
            meta = pq.read_metadata(out_parquet)
            if meta.num_rows == expected_rows:
                log_progress(f"  Fold {fold} Parquet already exists and matches expected row count ({meta.num_rows:,}). Reusing existing completed fold.")
                sz_mb = out_parquet.stat().st_size / (1024 * 1024)
                sha = sha256_file(out_parquet)
                oof_artifacts.append({
                    "fold": fold,
                    "path": str(out_parquet),
                    "rows": expected_rows,
                    "positives": expected_pos,
                    "size_mb": sz_mb,
                    "sha256": sha
                })
                fold_stats[fold] = {"rows": expected_rows, "positives": expected_pos}
                continue
            else:
                log_progress(f"  Existing fold {fold} Parquet has {meta.num_rows:,} rows (expected {expected_rows:,}). Overwriting invalid file.")
                out_parquet.unlink(missing_ok=True)

        if tmp_parquet.exists():
            tmp_parquet.unlink(missing_ok=True)

        # Open Parquet writer for this fold
        writer = pq.ParquetWriter(tmp_parquet, PARQUET_SCHEMA, compression="zstd", compression_level=3)
        fold_scored_count = 0
        fold_pos_count = 0

        # --- STREAM S2 CANDIDATES ---
        t0 = time.time()
        log_progress(f"  Fold {fold} S2: Extracting features in DuckDB...")
        con.execute(f"""
            CREATE OR REPLACE TABLE fold_s2 AS
            SELECT 
                c.source1_entity_id, 
                c.matched_entity_id AS candidate_entity_id,
                'source2' AS target_source,
                CAST({fold} AS INT8) AS fold,
                CAST(CASE WHEN gt.matched_entity_id IS NOT NULL THEN 1 ELSE 0 END AS INT8) AS label,
                CAST(0 AS INT8) AS source_is_s3,
                {FEATURE_SQL}
            FROM read_csv('{TRAIN_S2_CAND}', delim='\\t', header=true, all_varchar=true) c
            JOIN folds f ON c.source1_entity_id = f.source1_entity_id
            JOIN train_s1 s1 ON c.source1_entity_id = s1.entity_id
            JOIN train_s2 tgt ON c.matched_entity_id = tgt.entity_id
            LEFT JOIN raw_gt gt ON c.source1_entity_id = gt.source1_entity_id AND c.matched_entity_id = gt.matched_entity_id
            WHERE f.fold = {fold};
        """)
        dt_feat_s2 = time.time() - t0
        runtimes["feature_gen"] += dt_feat_s2
        s2_count = con.execute("SELECT COUNT(*) FROM fold_s2").fetchone()[0]
        s2_pos = con.execute("SELECT SUM(label) FROM fold_s2").fetchone()[0]
        fold_pos_count += int(s2_pos)
        log_progress(f"  Fold {fold} S2 extracted: {s2_count:,} rows ({s2_pos:,} pos) in {dt_feat_s2:.2f}s. Streaming & scoring...")

        reader_s2 = con.execute("SELECT * FROM fold_s2").to_arrow_reader(batch_size=500_000)
        batch_idx = 0
        for record_batch in reader_s2:
            batch_idx += 1
            df_chunk = record_batch.to_pandas()
            for col in CANONICAL_FEATURES:
                if col in CAT_FEATURES:
                    df_chunk[col] = df_chunk[col].astype("int8")
                else:
                    df_chunk[col] = df_chunk[col].astype("float32")

            t_sc0 = time.time()
            X_chunk = df_chunk[CANONICAL_FEATURES]
            preds = model.predict(X_chunk, num_threads=8)
            runtimes["scoring"] += (time.time() - t_sc0)

            df_chunk["score"] = preds.astype("float32")
            df_chunk["fold"] = df_chunk["fold"].astype("int8")
            df_chunk["label"] = df_chunk["label"].astype("int8")

            out_chunk = df_chunk[["source1_entity_id", "candidate_entity_id", "target_source", "fold", "score", "label"]]
            t_wr0 = time.time()
            pa_table = pa.Table.from_pandas(out_chunk, schema=PARQUET_SCHEMA, preserve_index=False)
            writer.write_table(pa_table)
            runtimes["writing"] += (time.time() - t_wr0)

            fold_scored_count += len(df_chunk)
            if batch_idx % 10 == 0:
                log_progress(f"    S2 Batch {batch_idx:02d}: scored & written {fold_scored_count:,} rows...")

            del df_chunk, X_chunk, preds, out_chunk, pa_table, record_batch
            gc.collect()

        con.execute("DROP TABLE fold_s2;")
        gc.collect()

        # --- STREAM S3 CANDIDATES ---
        t0 = time.time()
        log_progress(f"  Fold {fold} S3: Extracting features in DuckDB...")
        con.execute(f"""
            CREATE OR REPLACE TABLE fold_s3 AS
            SELECT 
                c.source1_entity_id, 
                c.matched_entity_id AS candidate_entity_id,
                'source3' AS target_source,
                CAST({fold} AS INT8) AS fold,
                CAST(CASE WHEN gt.matched_entity_id IS NOT NULL THEN 1 ELSE 0 END AS INT8) AS label,
                CAST(1 AS INT8) AS source_is_s3,
                {FEATURE_SQL}
            FROM read_csv('{TRAIN_S3_CAND}', delim='\\t', header=true, all_varchar=true) c
            JOIN folds f ON c.source1_entity_id = f.source1_entity_id
            JOIN train_s1 s1 ON c.source1_entity_id = s1.entity_id
            JOIN train_s3 tgt ON c.matched_entity_id = tgt.entity_id
            LEFT JOIN raw_gt gt ON c.source1_entity_id = gt.source1_entity_id AND c.matched_entity_id = gt.matched_entity_id
            WHERE f.fold = {fold};
        """)
        dt_feat_s3 = time.time() - t0
        runtimes["feature_gen"] += dt_feat_s3
        s3_count = con.execute("SELECT COUNT(*) FROM fold_s3").fetchone()[0]
        s3_pos = con.execute("SELECT SUM(label) FROM fold_s3").fetchone()[0]
        fold_pos_count += int(s3_pos)
        log_progress(f"  Fold {fold} S3 extracted: {s3_count:,} rows ({s3_pos:,} pos) in {dt_feat_s3:.2f}s. Streaming & scoring...")

        reader_s3 = con.execute("SELECT * FROM fold_s3").to_arrow_reader(batch_size=500_000)
        for record_batch in reader_s3:
            batch_idx += 1
            df_chunk = record_batch.to_pandas()
            for col in CANONICAL_FEATURES:
                if col in CAT_FEATURES:
                    df_chunk[col] = df_chunk[col].astype("int8")
                else:
                    df_chunk[col] = df_chunk[col].astype("float32")

            t_sc0 = time.time()
            X_chunk = df_chunk[CANONICAL_FEATURES]
            preds = model.predict(X_chunk, num_threads=8)
            runtimes["scoring"] += (time.time() - t_sc0)

            df_chunk["score"] = preds.astype("float32")
            df_chunk["fold"] = df_chunk["fold"].astype("int8")
            df_chunk["label"] = df_chunk["label"].astype("int8")

            out_chunk = df_chunk[["source1_entity_id", "candidate_entity_id", "target_source", "fold", "score", "label"]]
            t_wr0 = time.time()
            pa_table = pa.Table.from_pandas(out_chunk, schema=PARQUET_SCHEMA, preserve_index=False)
            writer.write_table(pa_table)
            runtimes["writing"] += (time.time() - t_wr0)

            fold_scored_count += len(df_chunk)
            if batch_idx % 10 == 0:
                log_progress(f"    S3 Batch {batch_idx:02d}: scored & written {fold_scored_count:,} rows...")

            del df_chunk, X_chunk, preds, out_chunk, pa_table, record_batch
            gc.collect()

        con.execute("DROP TABLE fold_s3;")
        gc.collect()

        writer.close()

        # Verify fold integrity before rename
        assert fold_scored_count == expected_rows, f"Fold {fold} row count mismatch: {fold_scored_count} vs {expected_rows}"
        assert fold_pos_count == expected_pos, f"Fold {fold} positive count mismatch: {fold_pos_count} vs {expected_pos}"

        # Atomically rename
        tmp_parquet.replace(out_parquet)

        sz_mb = out_parquet.stat().st_size / (1024 * 1024)
        sha = sha256_file(out_parquet)
        dt_fold = time.time() - t_fold_start
        log_progress(f"  Fold {fold} complete in {dt_fold:.2f}s -> {sz_mb:.2f} MB (SHA256: {sha[:16]}...)")

        oof_artifacts.append({
            "fold": fold,
            "path": str(out_parquet),
            "rows": fold_scored_count,
            "positives": fold_pos_count,
            "size_mb": sz_mb,
            "sha256": sha
        })
        fold_stats[fold] = {"rows": fold_scored_count, "positives": fold_pos_count}

    # 4. Mandatory Assertions Suite
    log_progress("\n" + "=" * 70)
    log_progress("[Step 4/6] MANDATORY IN-RUN INTEGRITY ASSERTIONS SUITE")
    log_progress("=" * 70)

    total_universe_rows = sum(a["rows"] for a in oof_artifacts)
    total_positives = sum(a["positives"] for a in oof_artifacts)

    # Assertion A: Candidate Count
    log_progress(f"Assertion A (Candidate Count): {total_universe_rows:,} (Expected: {EXPECTED_TOTAL_CANDIDATES:,})")
    assert total_universe_rows == EXPECTED_TOTAL_CANDIDATES, f"Assertion A FAILED: {total_universe_rows}"
    log_progress("  -> PASS: Exactly 93,171,949 candidate rows scored.")

    # Assertion B: V3 Captured Labels
    log_progress(f"Assertion B (Captured Labels): {total_positives:,} (Expected: {EXPECTED_CAPTURED_POSITIVES:,})")
    assert total_positives == EXPECTED_CAPTURED_POSITIVES, f"Assertion B FAILED: {total_positives}"
    log_progress("  -> PASS: Exactly 5,511,986 captured ground truth pairs.")

    # Register Full OOF View in DuckDB
    parquet_pattern = str(OUT_DIR / "v3_train_oof_fold*.parquet")
    con.execute(f"CREATE OR REPLACE VIEW full_oof AS SELECT * FROM read_parquet('{parquet_pattern}')")

    # Assertion C: Every Candidate Scored Exactly Once
    log_progress("Assertion C (Every candidate scored exactly once)...")
    total_in_view = con.execute("SELECT COUNT(*) FROM full_oof").fetchone()[0]
    assert total_in_view == EXPECTED_TOTAL_CANDIDATES, f"Assertion C FAILED: view count {total_in_view}"
    log_progress("  -> PASS: All 93,171,949 rows present in final Parquet files.")

    # Assertion D: Exactly One Fold per Candidate
    log_progress("Assertion D (Every candidate has exactly one fold in [0,4])...")
    bad_folds = con.execute("SELECT COUNT(*) FROM full_oof WHERE fold IS NULL OR fold < 0 OR fold > 4").fetchone()[0]
    assert bad_folds == 0, f"Assertion D FAILED: {bad_folds} invalid folds found"
    log_progress("  -> PASS: 100% of candidate rows have valid fold ID.")

    # Assertion E: Scored by Correct Held-Out Model
    log_progress("Assertion E (Scored by correct held-out model)...")
    fold_leakage = con.execute("""
        SELECT COUNT(*) 
        FROM full_oof o 
        JOIN folds f ON o.source1_entity_id = f.source1_entity_id 
        WHERE o.fold <> f.fold
    """).fetchone()[0]
    assert fold_leakage == 0, f"Assertion E FAILED: {fold_leakage} fold mismatches found"
    log_progress("  -> PASS: 0 fold mismatches. Every candidate scored strictly by its S1 held-out model.")

    # Assertion F: All Scores Finite & Valid Probabilities
    log_progress("Assertion F (All scores finite in [0.0, 1.0])...")
    score_check = con.execute("""
        SELECT 
            COUNT(CASE WHEN score IS NULL THEN 1 END) as nulls,
            COUNT(CASE WHEN score = 'inf' or score = '-inf' THEN 1 END) as infs,
            COUNT(CASE WHEN isnan(score) THEN 1 END) as nans,
            COUNT(CASE WHEN score < 0.0 or score > 1.0 THEN 1 END) as out_of_bounds,
            MIN(score) as min_score,
            MAX(score) as max_score
        FROM full_oof
    """).df()
    assert score_check["nulls"][0] == 0 and score_check["infs"][0] == 0 and score_check["nans"][0] == 0 and score_check["out_of_bounds"][0] == 0
    log_progress(f"  -> PASS: All scores finite in [{score_check['min_score'][0]:.6f}, {score_check['max_score'][0]:.6f}].")

    # Assertion G: 100% of S1 IDs Resolve
    log_progress("Assertion G (100% of S1 IDs resolve in normalized entities)...")
    unresolved_s1 = con.execute("""
        SELECT COUNT(*) 
        FROM full_oof o 
        LEFT JOIN train_s1 s1 ON o.source1_entity_id = s1.entity_id 
        WHERE s1.entity_id IS NULL
    """).fetchone()[0]
    assert unresolved_s1 == 0, f"Assertion G FAILED: {unresolved_s1} unresolved S1 IDs"
    log_progress("  -> PASS: 100.0% S1 IDs resolve in canonical entities.")

    # Assertion H: 100% of Target IDs Resolve
    log_progress("Assertion H (100% of target IDs resolve in canonical S2/S3)...")
    unresolved_s2 = con.execute("""
        SELECT COUNT(*) 
        FROM full_oof o 
        LEFT JOIN train_s2 s2 ON o.candidate_entity_id = s2.entity_id 
        WHERE o.target_source = 'source2' AND s2.entity_id IS NULL
    """).fetchone()[0]
    unresolved_s3 = con.execute("""
        SELECT COUNT(*) 
        FROM full_oof o 
        LEFT JOIN train_s3 s3 ON o.candidate_entity_id = s3.entity_id 
        WHERE o.target_source = 'source3' AND s3.entity_id IS NULL
    """).fetchone()[0]
    assert unresolved_s2 == 0 and unresolved_s3 == 0, f"Assertion H FAILED: S2 unresolved={unresolved_s2}, S3 unresolved={unresolved_s3}"
    log_progress("  -> PASS: 100.0% target IDs resolve in canonical S2 and S3.")

    # Assertion I: No Duplicate Candidate Pairs
    log_progress("Assertion I (No duplicate candidate pairs)...")
    dup_pairs = con.execute("""
        SELECT COUNT(*) FROM (
            SELECT source1_entity_id, candidate_entity_id 
            FROM full_oof 
            GROUP BY 1, 2 
            HAVING COUNT(*) > 1
        )
    """).fetchone()[0]
    assert dup_pairs == 0, f"Assertion I FAILED: {dup_pairs} duplicate candidate pairs found"
    log_progress("  -> PASS: 0 duplicate candidate pairs.")

    # Assertion J: No Accidental Negative Sampling
    log_progress("Assertion J (No accidental negative sampling)...")
    total_negatives = con.execute("SELECT COUNT(*) FROM full_oof WHERE label = 0").fetchone()[0]
    expected_negatives = EXPECTED_TOTAL_CANDIDATES - EXPECTED_CAPTURED_POSITIVES
    assert total_negatives == expected_negatives, f"Assertion J FAILED: negatives {total_negatives} vs expected {expected_negatives}"
    log_progress(f"  -> PASS: Exactly {total_negatives:,} negatives scored (0 negatives sampled or omitted).")

    # Assertion K & L: Per-fold row counts and positive counts
    log_progress("Assertion K & L (Per-fold candidate and positive counts):")
    for f in range(5):
        st = fold_stats[f]
        log_progress(f"  Fold {f}: {st['rows']:,} rows (Positives: {st['positives']:,})")
        assert st["rows"] == EXPECTED_FOLD_ROWS[f], f"Fold {f} row mismatch: {st['rows']}"
        assert st["positives"] == EXPECTED_FOLD_POS[f], f"Fold {f} pos mismatch: {st['positives']}"
    log_progress("  -> PASS: All per-fold row counts and positive counts match authoritative specification.")

    # 5. Diagnostic Evaluation Checks
    log_progress("\n" + "=" * 70)
    log_progress("[Step 5/6] DIAGNOSTIC EVALUATION CHECKS")
    log_progress("=" * 70)

    # 5A. Sample Reproduction Check & 153-Pair Check
    log_progress("\n--- Diagnostic 1: Historical Sample Reproduction & 153-Pair Check ---")
    log_progress("Selecting historical sampled subset: 100% positives + 10% hash negatives...")
    sample_df = con.execute("""
        SELECT 
            COUNT(*) as sample_rows,
            SUM(label) as sample_pos,
            COUNT(CASE WHEN score >= 0.60 THEN 1 END) as pred_pos,
            COUNT(CASE WHEN score >= 0.60 AND label = 1 THEN 1 END) as tp,
            COUNT(CASE WHEN score >= 0.60 AND label = 0 THEN 1 END) as fp,
            COUNT(CASE WHEN score < 0.60 AND label = 1 THEN 1 END) as fn
        FROM full_oof
        WHERE label = 1 OR (ABS(hash(source1_entity_id || candidate_entity_id)) % 10 = 0)
    """).df()
    
    sample_rows = int(sample_df["sample_rows"][0])
    sample_pos = int(sample_df["sample_pos"][0])
    sample_pred_pos = int(sample_df["pred_pos"][0])
    sample_tp = int(sample_df["tp"][0])
    sample_fp = int(sample_df["fp"][0])
    sample_fn = int(sample_df["fn"][0])
    sample_prec = sample_tp / (sample_tp + sample_fp) if (sample_tp + sample_fp) > 0 else 0.0
    sample_rec = sample_tp / (sample_tp + sample_fn) if (sample_tp + sample_fn) > 0 else 0.0
    sample_f05 = (1.25 * sample_prec * sample_rec) / (0.25 * sample_prec + sample_rec) if (0.25 * sample_prec + sample_rec) > 0 else 0.0

    log_progress(f"Sampled population rows: {sample_rows:,} (Expected: 14,282,556)")
    log_progress(f"Sampled positives:       {sample_pos:,} (Expected: 5,511,986)")
    log_progress(f"Sample Pred Positives:   {sample_pred_pos:,} (Target: 5,430,010)")
    log_progress(f"Sample Pairwise TP:      {sample_tp:,}")
    log_progress(f"Sample Pairwise FP:      {sample_fp:,}")
    log_progress(f"Sample Pairwise FN:      {sample_fn:,}")
    log_progress(f"Sample Pairwise Prec:    {sample_prec:.6f}")
    log_progress(f"Sample Pairwise Rec:     {sample_rec:.6f}")
    log_progress(f"Sample Pairwise F0.5:    {sample_f05:.6f}")

    # 153-Pair check determination
    pair_check_status = "5,318,532" if sample_tp == 5318532 else ("5,318,379" if sample_tp == 5318379 else f"Other ({sample_tp})")
    log_progress(f"153-PAIR REPRODUCIBILITY CHECK RESULT: Pairwise TP = {sample_tp:,} -> {pair_check_status}")

    # Compute Macro F0.5 on sample using scorer_v1
    log_progress("Computing Macro F0.5 on sampled population using scorer_v1...")
    sample_pos_preds = con.execute("""
        SELECT source1_entity_id, candidate_entity_id 
        FROM full_oof 
        WHERE (label = 1 OR (ABS(hash(source1_entity_id || candidate_entity_id)) % 10 = 0))
          AND score >= 0.60
    """).df()
    pred_map_sample = {}
    for s1, tgt in zip(sample_pos_preds["source1_entity_id"], sample_pos_preds["candidate_entity_id"]):
        if s1 not in pred_map_sample:
            pred_map_sample[s1] = set()
        pred_map_sample[s1].add(tgt)
    
    gt_df = con.execute("SELECT source1_entity_id, matched_entity_id FROM raw_gt").df()
    gt_map = {}
    for s1, tgt in zip(gt_df["source1_entity_id"], gt_df["matched_entity_id"]):
        if s1 not in gt_map:
            gt_map[s1] = set()
        gt_map[s1].add(tgt)

    all_train_s1 = set(con.execute("SELECT entity_id FROM train_s1").df()["entity_id"])
    for s1 in all_train_s1:
        if s1 not in gt_map:
            gt_map[s1] = set()

    macro_res = score_predictions(pred_map_sample, gt_map, entity_ids=all_train_s1)
    sample_macro_f05 = macro_res.get("macro_f0.5", 0.0)
    log_progress(f"Sample Macro F0.5 @ T=0.60: {sample_macro_f05:.6f} (Target: 0.841781)")

    # 5B. Full-Universe Threshold Diagnostics
    log_progress("\n--- Diagnostic 2: Full-Universe Threshold Diagnostics ---")
    log_progress("DISCLAIMER: These pairwise diagnostics are informational only.")
    log_progress("Do NOT call this the official Macro F0.5 result. P3 will independently score the OOF using scorer_v1.")

    thresholds = [0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.97, 0.98, 0.99]
    thresh_results = []
    
    for th in thresholds:
        stats = con.execute(f"""
            SELECT 
                COUNT(CASE WHEN score >= {th} THEN 1 END) as pred_pos,
                COUNT(CASE WHEN score >= {th} AND label = 1 THEN 1 END) as tp,
                COUNT(CASE WHEN score >= {th} AND label = 0 THEN 1 END) as fp,
                COUNT(CASE WHEN score < {th} AND label = 1 THEN 1 END) as fn_cand
            FROM full_oof
        """).df()
        
        tp = int(stats["tp"][0])
        fp = int(stats["fp"][0])
        pred_pos = int(stats["pred_pos"][0])
        fn_cand = int(stats["fn_cand"][0])
        fn_full = EXPECTED_TOTAL_GT_POSITIVES - tp
        
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec_cand = tp / EXPECTED_CAPTURED_POSITIVES
        rec_full = tp / EXPECTED_TOTAL_GT_POSITIVES
        
        f05_cand = (1.25 * prec * rec_cand) / (0.25 * prec + rec_cand) if (0.25 * prec + rec_cand) > 0 else 0.0
        f05_full = (1.25 * prec * rec_full) / (0.25 * prec + rec_full) if (0.25 * prec + rec_full) > 0 else 0.0
        matches_per_s1 = pred_pos / EXPECTED_TRAIN_S1

        thresh_results.append({
            "threshold": th,
            "pred_positives": pred_pos,
            "tp": tp,
            "fp": fp,
            "fn_candidate_universe": fn_cand,
            "fn_full_ground_truth": fn_full,
            "pairwise_precision": prec,
            "pairwise_recall_candidate": rec_cand,
            "pairwise_recall_full": rec_full,
            "pairwise_f0.5_cand": f05_cand,
            "pairwise_f0.5_full": f05_full,
            "predicted_matches_per_s1": matches_per_s1
        })
        log_progress(f"  T={th:.2f} | PredPos: {pred_pos:8,d} | TP: {tp:8,d} | FP: {fp:8,d} | Prec: {prec:.4f} | Rec(Cand): {rec_cand:.4f} | Rec(Full): {rec_full:.4f} | Matches/S1: {matches_per_s1:.4f}")

    # 5C. Full-Universe Score Distribution
    log_progress("\n--- Diagnostic 3: Score Distribution & Quantiles ---")
    quantiles_df = con.execute("""
        SELECT 
            quantile_cont(score, 0.01) as p01,
            quantile_cont(score, 0.05) as p05,
            quantile_cont(score, 0.10) as p10,
            quantile_cont(score, 0.25) as p25,
            quantile_cont(score, 0.50) as p50,
            quantile_cont(score, 0.75) as p75,
            quantile_cont(score, 0.90) as p90,
            quantile_cont(score, 0.95) as p95,
            quantile_cont(score, 0.99) as p99,
            quantile_cont(score, 0.995) as p99_5,
            quantile_cont(score, 0.999) as p99_9,
            quantile_cont(score, 0.9995) as p99_95,
            quantile_cont(score, 0.9999) as p99_99
        FROM full_oof
    """).df()
    quantiles_dict = {k: float(v) for k, v in quantiles_df.to_dict(orient="records")[0].items()}
    for qname, qval in quantiles_dict.items():
        log_progress(f"  {qname:8s}: {qval:.6f}")

    score_counts = {}
    for th in [0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]:
        cnt = con.execute(f"SELECT COUNT(*) FROM full_oof WHERE score >= {th}").fetchone()[0]
        score_counts[f">={th:.2f}"] = cnt
        log_progress(f"  Count >= {th:.2f}: {cnt:,} ({cnt/EXPECTED_TOTAL_CANDIDATES*100:.2f}%)")

    # 5D. Entity-Level Diagnostics
    log_progress("\n--- Diagnostic 4: Entity-Level Diagnostics ---")
    entity_diag = {}
    for th in [0.60, 0.90]:
        log_progress(f"Computing entity-level match distribution @ T={th:.2f}...")
        con.execute(f"""
            CREATE OR REPLACE TEMP TABLE entity_matches_{int(th*100)} AS
            SELECT 
                s.entity_id as source1_entity_id,
                COALESCE(p.cnt, 0) as pred_matches
            FROM train_s1 s
            LEFT JOIN (
                SELECT source1_entity_id, COUNT(*) as cnt
                FROM full_oof
                WHERE score >= {th}
                GROUP BY source1_entity_id
            ) p ON s.entity_id = p.source1_entity_id;
        """)
        
        hist = con.execute(f"""
            SELECT 
                COUNT(CASE WHEN pred_matches = 0 THEN 1 END) as matches_0,
                COUNT(CASE WHEN pred_matches = 1 THEN 1 END) as matches_1,
                COUNT(CASE WHEN pred_matches = 2 THEN 1 END) as matches_2,
                COUNT(CASE WHEN pred_matches = 3 THEN 1 END) as matches_3,
                COUNT(CASE WHEN pred_matches >= 4 THEN 1 END) as matches_4plus,
                AVG(pred_matches) as mean_matches,
                quantile_cont(pred_matches, 0.50) as median_matches,
                quantile_cont(pred_matches, 0.95) as p95_matches
            FROM entity_matches_{int(th*100)};
        """).df().to_dict(orient="records")[0]
        
        entity_diag[f"T={th:.2f}"] = hist
        log_progress(f"  T={th:.2f} Matches Distribution across {EXPECTED_TRAIN_S1:,} S1 entities:")
        log_progress(f"    0 matches: {hist['matches_0']:,} ({hist['matches_0']/EXPECTED_TRAIN_S1*100:.2f}%)")
        log_progress(f"    1 match:   {hist['matches_1']:,} ({hist['matches_1']/EXPECTED_TRAIN_S1*100:.2f}%)")
        log_progress(f"    2 matches: {hist['matches_2']:,} ({hist['matches_2']/EXPECTED_TRAIN_S1*100:.2f}%)")
        log_progress(f"    3 matches: {hist['matches_3']:,} ({hist['matches_3']/EXPECTED_TRAIN_S1*100:.2f}%)")
        log_progress(f"    >=4 matches: {hist['matches_4plus']:,} ({hist['matches_4plus']/EXPECTED_TRAIN_S1*100:.2f}%)")
        log_progress(f"    Mean: {hist['mean_matches']:.4f} | Median: {hist['median_matches']:.1f} | P95: {hist['p95_matches']:.1f}")

    # Ground truth capture distribution across all S1 entities
    log_progress("\nComputing S1 entity Ground Truth capture distribution...")
    con.execute("""
        CREATE OR REPLACE TEMP TABLE gt_per_s1 AS
        SELECT source1_entity_id, COUNT(*) as gt_count
        FROM raw_gt
        GROUP BY source1_entity_id;
    """)

    con.execute("""
        CREATE OR REPLACE TEMP TABLE captured_per_s1 AS
        SELECT source1_entity_id, COUNT(*) as captured_count
        FROM full_oof
        WHERE label = 1
        GROUP BY source1_entity_id;
    """)

    capture_dist = con.execute("""
        SELECT 
            CASE 
                WHEN COALESCE(g.gt_count, 0) = 0 THEN 'NO_TRUTH'
                WHEN COALESCE(c.captured_count, 0) = 0 THEN 'ZERO_CAPTURE'
                WHEN c.captured_count < g.gt_count THEN 'PARTIAL_CAPTURE'
                ELSE 'FULL_CAPTURE'
            END as capture_status,
            COUNT(*) as entity_count,
            ROUND(COUNT(*) * 100.0 / 2206821, 2) as pct
        FROM train_s1 s
        LEFT JOIN gt_per_s1 g ON s.entity_id = g.source1_entity_id
        LEFT JOIN captured_per_s1 c ON s.entity_id = c.source1_entity_id
        GROUP BY 1
        ORDER BY entity_count DESC;
    """).df().to_dict(orient="records")

    log_progress("  S1 Entity Capture Status Distribution:")
    for row in capture_dist:
        log_progress(f"    {row['capture_status']:16s}: {row['entity_count']:,} ({row['pct']}%)")

    # 5E. Train / Test Parity Diagnostics
    log_progress("\n--- Diagnostic 5: Train / Test Parity Diagnostic ---")
    test_s2_pred = REPO / "P2" / "predictions" / "phase4" / "test_predictions_s2.tsv"
    test_s3_pred = REPO / "P2" / "predictions" / "phase4" / "test_predictions_s3.tsv"
    test_s1_path = REPO / "outputs" / "person1_step1" / "normalized" / "test_source1_normalized.tsv"

    test_parity = {}
    if test_s2_pred.exists() and test_s3_pred.exists() and test_s1_path.exists():
        log_progress("Reading test predictions and test S1 entities...")
        test_s1_count = con.execute(f"SELECT COUNT(*) FROM read_csv('{test_s1_path}', delim='\\t', header=true, all_varchar=true)").fetchone()[0]
        
        con.execute(f"""
            CREATE OR REPLACE TEMP VIEW test_preds AS
            SELECT model_score AS score, source1_entity_id, candidate_entity_id 
            FROM read_csv('{test_s2_pred}', delim='\\t', header=true)
            UNION ALL
            SELECT model_score AS score, source1_entity_id, candidate_entity_id 
            FROM read_csv('{test_s3_pred}', delim='\\t', header=true);
        """)
        
        test_cnt = con.execute("SELECT COUNT(*) FROM test_preds").fetchone()[0]
        test_pos_60 = con.execute("SELECT COUNT(*) FROM test_preds WHERE score >= 0.60").fetchone()[0]
        test_pos_90 = con.execute("SELECT COUNT(*) FROM test_preds WHERE score >= 0.90").fetchone()[0]
        
        test_quantiles = con.execute("""
            SELECT 
                quantile_cont(score, 0.50) as p50,
                quantile_cont(score, 0.75) as p75,
                quantile_cont(score, 0.90) as p90,
                quantile_cont(score, 0.95) as p95,
                quantile_cont(score, 0.99) as p99,
                quantile_cont(score, 0.999) as p99_9
            FROM test_preds
        """).df().to_dict(orient="records")[0]

        train_matches_per_s1_60 = score_counts[">=0.60"] / EXPECTED_TRAIN_S1
        train_matches_per_s1_90 = score_counts[">=0.90"] / EXPECTED_TRAIN_S1
        test_matches_per_s1_60 = test_pos_60 / test_s1_count
        test_matches_per_s1_90 = test_pos_90 / test_s1_count

        test_parity = {
            "train_candidates_per_s1": EXPECTED_TOTAL_CANDIDATES / EXPECTED_TRAIN_S1,
            "test_candidates_per_s1": test_cnt / test_s1_count,
            "train_score_quantiles": {
                "p50": quantiles_dict["p50"],
                "p75": quantiles_dict["p75"],
                "p90": quantiles_dict["p90"],
                "p95": quantiles_dict["p95"],
                "p99": quantiles_dict["p99"],
                "p99_9": quantiles_dict["p99_9"]
            },
            "test_score_quantiles": {k: float(v) for k, v in test_quantiles.items()},
            "train_pred_matches_per_s1_t060": train_matches_per_s1_60,
            "test_pred_matches_per_s1_t060": test_matches_per_s1_60,
            "train_pred_matches_per_s1_t090": train_matches_per_s1_90,
            "test_pred_matches_per_s1_t090": test_matches_per_s1_90,
            "distribution_shift_flag": "NORMAL" if abs(train_matches_per_s1_60 - test_matches_per_s1_60) / train_matches_per_s1_60 < 0.25 else "LARGE_SHIFT_DETECTED"
        }

        log_progress(f"  Train candidates per S1: {test_parity['train_candidates_per_s1']:.2f}")
        log_progress(f"  Test candidates per S1:  {test_parity['test_candidates_per_s1']:.2f}")
        log_progress(f"  Train matches/S1 @ T=0.60: {train_matches_per_s1_60:.4f}")
        log_progress(f"  Test matches/S1 @ T=0.60:  {test_matches_per_s1_60:.4f}")
        log_progress(f"  Train matches/S1 @ T=0.90: {train_matches_per_s1_90:.4f}")
        log_progress(f"  Test matches/S1 @ T=0.90:  {test_matches_per_s1_90:.4f}")
        log_progress(f"  Train Score Quantiles:   {test_parity['train_score_quantiles']}")
        log_progress(f"  Test Score Quantiles:    {test_parity['test_score_quantiles']}")
        log_progress(f"  Distribution Shift Status: {test_parity['distribution_shift_flag']}")
    else:
        log_progress("  Test prediction files not found or incomplete. Skipping test parity comparison.")

    # 6. Performance & Storage Metrics
    log_progress("\n" + "=" * 70)
    log_progress("[Step 6/6] PERFORMANCE, STORAGE & REPORT GENERATION")
    log_progress("=" * 70)

    total_time = time.time() - t_global_start
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    e_usage = shutil.disk_usage("E:/")
    remaining_e_free_gb = e_usage.free / (1024**3)
    
    total_oof_mb = sum(a["size_mb"] for a in oof_artifacts)
    temp_dir_mb = sum(f.stat().st_size for f in TEMP_DIR.glob("**/*") if f.is_file()) / (1024 * 1024)

    perf_metrics = {
        "feature_generation_sec": runtimes["feature_gen"],
        "scoring_sec": runtimes["scoring"],
        "parquet_writing_sec": runtimes["writing"],
        "total_runtime_sec": total_time,
        "peak_python_memory_mb": peak_mem / (1024 * 1024),
        "temporary_disk_usage_mb": temp_dir_mb,
        "final_oof_disk_usage_mb": total_oof_mb,
        "remaining_e_free_gb": remaining_e_free_gb
    }

    log_progress(f"Feature Gen Runtime: {runtimes['feature_gen']:.2f}s ({runtimes['feature_gen']/60:.2f}m)")
    log_progress(f"Scoring Runtime:     {runtimes['scoring']:.2f}s ({runtimes['scoring']/60:.2f}m)")
    log_progress(f"Writing Runtime:     {runtimes['writing']:.2f}s ({runtimes['writing']/60:.2f}m)")
    log_progress(f"Total Runtime:       {total_time:.2f}s ({total_time/60:.2f}m)")
    log_progress(f"Peak Python Heap:    {perf_metrics['peak_python_memory_mb']:.2f} MB")
    log_progress(f"Temporary Disk:      {temp_dir_mb:.2f} MB")
    log_progress(f"Final OOF Disk:      {total_oof_mb:.2f} MB ({total_oof_mb/1024:.2f} GB)")
    log_progress(f"Remaining E: Free:   {remaining_e_free_gb:.2f} GB")

    # 7. Write Manifest and Report
    manifest_rows = []
    for item in oof_artifacts:
        manifest_rows.append({
            "artifact": Path(item["path"]).name,
            "path": item["path"],
            "fold": item["fold"],
            "rows": item["rows"],
            "positives": item["positives"],
            "size_mb": round(item["size_mb"], 2),
            "sha256": item["sha256"]
        })
    manifest_df = pd.DataFrame(manifest_rows)
    manifest_path = MANIFEST_DIR / "V3_FULL_OOF_MANIFEST.tsv"
    manifest_df.to_csv(manifest_path, sep="\t", index=False)
    log_progress(f"Manifest written to: {manifest_path}")

    report_data = {
        "pipeline": "V3 Full-Universe Out-of-Fold (OOF) Inference",
        "timestamp_utc": time.strftime("%Y-%m-%d %H:%M:%SZ", time.gmtime()),
        "total_candidate_rows": total_universe_rows,
        "captured_positive_labels": total_positives,
        "total_negative_labels": total_universe_rows - total_positives,
        "per_fold_stats": fold_stats,
        "model_to_fold_mapping": {f"fold_{f}": f"lgb_fold{f}.txt" for f in range(5)},
        "sample_reproduction_diagnostic": {
            "sample_rows": sample_rows,
            "sample_positives": sample_pos,
            "pred_positives": sample_pred_pos,
            "tp": sample_tp,
            "fp": sample_fp,
            "fn": sample_fn,
            "precision": sample_prec,
            "recall": sample_rec,
            "pairwise_f0.5": sample_f05,
            "macro_f0.5": sample_macro_f05
        },
        "153_pair_check": {
            "pairwise_tp": sample_tp,
            "status": pair_check_status,
            "metric_routine": "np.sum((probs >= 0.60) & (y_true == 1)) on 14,282,556 deterministic sampled subset"
        },
        "full_universe_threshold_diagnostics": thresh_results,
        "score_distribution": {
            "quantiles": quantiles_dict,
            "threshold_counts": score_counts
        },
        "entity_level_diagnostics": {
            "match_counts": entity_diag,
            "capture_distribution": capture_dist
        },
        "train_test_parity": test_parity,
        "performance": perf_metrics,
        "artifacts": manifest_rows
    }

    report_json_path = REPORT_DIR / "V3_FULL_OOF_REPORT.json"
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)
    log_progress(f"Report JSON written to: {report_json_path}")

    log_progress("\n" + "=" * 70)
    log_progress("FULL V3 OOF COMPLETE")
    log_progress("=" * 70)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log_progress(f"\nFATAL ERROR OCCURRED: {e}")
        import traceback
        traceback.print_exc()
        log_progress("FULL V3 OOF FAILED")
        sys.exit(1)
