#!/usr/bin/env python3
"""
Phase 4 Historical Sample Reproduction Script
============================================
Evaluates the exact 14,282,556 sampled population (100% positives + 10% hash negatives)
using the 5 existing Phase 4 LightGBM boosters and P3 folds_v1 manifest.
Compares metrics against historical Phase 4 report:
  Target: Macro F0.5 = 0.841781 @ T=0.60 (TP=5,318,532, FP=111,478, FN=193,454)
"""

import os
import sys
import time
import tempfile
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import lightgbm as lgb

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))
from validation.scorer_v1 import score_predictions

# Configure paths
MODEL_DIR = REPO / "P2" / "models" / "phase4"
CAND_DIR = REPO / "P1" / "data" / "candidates" / "v3"
TRAIN_S2_CAND = CAND_DIR / "train_candidate_pairs_s2_v3.tsv"
TRAIN_S3_CAND = CAND_DIR / "train_candidate_pairs_s3_v3.tsv"
GT_PATH = REPO / "outputs" / "person1_step1" / "train_ground_truth_reconstructed.tsv"
FOLDS_PATH = REPO / "P3" / "reports" / "folds_v1_manifest.tsv"
S1_PATH = REPO / "outputs" / "person1_step1" / "normalized" / "train_source1_normalized.tsv"
S2_PARQ = REPO / "P1" / "data" / "entities" / "train" / "source2" / "train_s2_entities.parquet"
S3_PARQ = REPO / "P1" / "data" / "entities" / "train" / "source3" / "train_s3_entities.parquet"

# Temp directory on Drive E:
TEMP_DIR = Path("E:/duckdb_tmp_reproduction")
TEMP_DIR.mkdir(parents=True, exist_ok=True)

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

def load_lf_model(model_path: Path) -> lgb.Booster:
    """Loads LightGBM booster model normalizing CRLF to LF line endings."""
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
    t_start = time.time()
    print("=" * 60)
    print("PHASE 4 HISTORICAL SAMPLE REPRODUCTION")
    print("=" * 60)

    # 1. Load models
    print("\n[1/5] Loading 5 Phase 4 LightGBM models with LF normalization...")
    models = []
    for fold in range(5):
        mpath = MODEL_DIR / f"lgb_fold{fold}.txt"
        bst = load_lf_model(mpath)
        print(f"  Fold {fold} Model ({mpath.name}): {bst.num_trees()} trees, {len(bst.feature_name())} features [OK]")
        models.append(bst)

    # 2. Setup DuckDB
    con = duckdb.connect()
    con.execute(f"SET temp_directory='{TEMP_DIR}'")
    con.execute("SET memory_limit='12GB'")
    con.execute("SET threads=4")
    con.execute("SET preserve_insertion_order=false")

    print("\n[2/5] Loading metadata, entities, and ground truth in DuckDB...")
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

    # 3. Extract exact sampled feature population
    print("\n[3/5] Extracting sampled pairs (100% positives + 10% hash negatives)...")
    t0 = time.time()
    con.execute(f"""
        CREATE TABLE train_sample_s2 AS
        SELECT 
            c.source1_entity_id, c.matched_entity_id,
            CASE WHEN gt.matched_entity_id IS NOT NULL THEN 1 ELSE 0 END AS label,
            f.fold,
            0 AS source_is_s3,
            {FEATURE_SQL}
        FROM read_csv('{TRAIN_S2_CAND}', delim='\\t', header=true, all_varchar=true) c
        JOIN folds f ON c.source1_entity_id = f.source1_entity_id
        JOIN train_s1 s1 ON c.source1_entity_id = s1.entity_id
        JOIN train_s2 tgt ON c.matched_entity_id = tgt.entity_id
        LEFT JOIN raw_gt gt ON c.source1_entity_id = gt.source1_entity_id AND c.matched_entity_id = gt.matched_entity_id
        WHERE gt.matched_entity_id IS NOT NULL OR (ABS(hash(c.source1_entity_id || c.matched_entity_id)) % 10 = 0);
    """)
    s2_cnt = con.execute("SELECT COUNT(*), SUM(label) FROM train_sample_s2").fetchone()
    print(f"  S2 sample: {s2_cnt[0]:,} rows ({s2_cnt[1]:,} pos, {s2_cnt[0]-s2_cnt[1]:,} neg) in {time.time()-t0:.2f}s")

    t0 = time.time()
    con.execute(f"""
        CREATE TABLE train_sample_s3 AS
        SELECT 
            c.source1_entity_id, c.matched_entity_id,
            CASE WHEN gt.matched_entity_id IS NOT NULL THEN 1 ELSE 0 END AS label,
            f.fold,
            1 AS source_is_s3,
            {FEATURE_SQL}
        FROM read_csv('{TRAIN_S3_CAND}', delim='\\t', header=true, all_varchar=true) c
        JOIN folds f ON c.source1_entity_id = f.source1_entity_id
        JOIN train_s1 s1 ON c.source1_entity_id = s1.entity_id
        JOIN train_s3 tgt ON c.matched_entity_id = tgt.entity_id
        LEFT JOIN raw_gt gt ON c.source1_entity_id = gt.source1_entity_id AND c.matched_entity_id = gt.matched_entity_id
        WHERE gt.matched_entity_id IS NOT NULL OR (ABS(hash(c.source1_entity_id || c.matched_entity_id)) % 10 = 0);
    """)
    s3_cnt = con.execute("SELECT COUNT(*), SUM(label) FROM train_sample_s3").fetchone()
    print(f"  S3 sample: {s3_cnt[0]:,} rows ({s3_cnt[1]:,} pos, {s3_cnt[0]-s3_cnt[1]:,} neg) in {time.time()-t0:.2f}s")

    con.execute("""
        CREATE TABLE train_data AS
        SELECT * FROM train_sample_s2
        UNION ALL
        SELECT * FROM train_sample_s3;
    """)
    con.execute("DROP TABLE train_sample_s2; DROP TABLE train_sample_s3;")

    df = con.execute("SELECT * FROM train_data").df()
    con.execute("DROP TABLE train_data;")

    total_rows = len(df)
    pos_rows = int(df["label"].sum())
    neg_rows = total_rows - pos_rows
    print(f"  Total Sampled Population: {total_rows:,} rows (Positives: {pos_rows:,}, Negatives: {neg_rows:,})")

    for col in CANONICAL_FEATURES:
        if col in CAT_FEATURES:
            df[col] = df[col].astype("int8")
        else:
            df[col] = df[col].astype("float32")
    df["label"] = df["label"].astype("int8")
    df["fold"] = df["fold"].astype("int8")

    # 4. Out-of-fold inference per fold
    print("\n[4/5] Running OOF scoring with held-out models per fold...")
    oof_preds = np.zeros(len(df), dtype=np.float32)
    fold_counts = {}

    for fold in range(5):
        t_fold = time.time()
        val_mask = (df["fold"] == fold).to_numpy()
        cnt_val = int(np.sum(val_mask))
        fold_counts[fold] = cnt_val
        X_val = df.loc[val_mask, CANONICAL_FEATURES]
        
        preds = models[fold].predict(X_val)
        oof_preds[val_mask] = preds
        print(f"  Fold {fold}: scored {cnt_val:,} rows with lgb_fold{fold}.txt in {time.time()-t_fold:.2f}s")

    df["oof_prob"] = oof_preds

    # 5. Scorer_v1 evaluation at T=0.60
    print("\n[5/5] Evaluating with validation.scorer_v1 at Threshold T = 0.60...")
    y_true = df["label"].to_numpy()
    probs = df["oof_prob"].to_numpy()
    th = 0.60

    pred_bin = (probs >= th).astype(int)
    tp = int(np.sum((pred_bin == 1) & (y_true == 1)))
    fp = int(np.sum((pred_bin == 1) & (y_true == 0)))
    fn = int(np.sum((pred_bin == 0) & (y_true == 1)))
    tn = int(np.sum((pred_bin == 0) & (y_true == 0)))

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    denom = 0.25 * prec + rec
    f05 = (1.25 * prec * rec / denom) if denom > 0 else 0.0

    print("  Building prediction and ground truth maps for macro scoring...")
    gt_df = con.execute("SELECT source1_entity_id, matched_entity_id FROM raw_gt").df()
    gt_map = {}
    for s1_id, tgt_id in zip(gt_df["source1_entity_id"], gt_df["matched_entity_id"]):
        if s1_id not in gt_map:
            gt_map[s1_id] = set()
        gt_map[s1_id].add(tgt_id)

    all_train_s1 = set(con.execute("SELECT entity_id FROM train_s1").df()["entity_id"])
    for s1 in all_train_s1:
        if s1 not in gt_map:
            gt_map[s1] = set()

    pos_df = df.loc[pred_bin == 1, ["source1_entity_id", "matched_entity_id"]]
    pred_map = {}
    for s1_id, tgt_id in zip(pos_df["source1_entity_id"], pos_df["matched_entity_id"]):
        if s1_id not in pred_map:
            pred_map[s1_id] = set()
        pred_map[s1_id].add(tgt_id)

    score_res = score_predictions(pred_map, gt_map, entity_ids=all_train_s1)
    macro_f05 = score_res.get("macro_f0.5", 0.0)

    runtime = time.time() - t_start

    print("\n" + "=" * 60)
    print("REPRODUCTION RESULTS SUMMARY")
    print("=" * 60)
    print(f"Total Sampled Rows:      {total_rows:,}")
    print(f"Positive Rows:           {pos_rows:,}")
    print(f"Negative Rows:           {neg_rows:,}")
    print(f"Fold Distribution:       {fold_counts}")
    print(f"T=0.60 Pred Positives:   {int(np.sum(pred_bin)):,}")
    print(f"TP:                      {tp:,}")
    print(f"FP:                      {fp:,}")
    print(f"FN:                      {fn:,}")
    print(f"Precision:               {prec:.6f}")
    print(f"Recall:                  {rec:.6f}")
    print(f"Pairwise F0.5:           {f05:.6f}")
    print(f"Macro F0.5:              {macro_f05:.6f}")
    print(f"Total Runtime:           {runtime:.2f}s")
    print("=" * 60)

    # Verification against historical targets
    expected_tp = 5318532
    expected_fp = 111478
    expected_fn = 193454
    expected_macro = 0.841781

    tp_diff = tp - expected_tp
    fp_diff = fp - expected_fp
    fn_diff = fn - expected_fn
    macro_diff = abs(macro_f05 - expected_macro)

    print(f"TP diff:    {tp_diff} (Actual: {tp:,} vs Expected: {expected_tp:,})")
    print(f"FP diff:    {fp_diff} (Actual: {fp:,} vs Expected: {expected_fp:,})")
    print(f"FN diff:    {fn_diff} (Actual: {fn:,} vs Expected: {expected_fn:,})")
    print(f"Macro diff: {macro_diff:.8f} (Actual: {macro_f05:.6f} vs Expected: {expected_macro:.6f})")

    if tp == expected_tp and fp == expected_fp and fn == expected_fn and macro_diff < 1e-5:
        print("\n>>> REPRODUCTION PASSED <<<")
    else:
        print("\n>>> REPRODUCTION FAILED <<<")

if __name__ == "__main__":
    main()
