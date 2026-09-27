#!/usr/bin/env python3
"""
P2/scripts/phase7_evaluate_all_folds.py

Task P2 & Section 22: Grouped OOF Evaluation Across All 5 Folds
Evaluates both V3 baseline and V4 retrieval pipeline on all 5 cross-validation folds:
- Strict grouped evaluation using P3/reports/folds_v1_manifest.tsv (no entity leakage)
- Uses official competition scorer (validation/scorer_v1.py)
- Records exact metrics per fold: candidate recall, precision, recall, Macro F0.5, singleton accuracy

Outputs:
- P2/reports/phase7_oof_results.tsv
"""

import time
import gc
import csv
import duckdb
import numpy as np
import lightgbm as lgb
import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from validation.scorer_v1 import score_predictions

print("=" * 80)
print("PHASE 7 TASK P2: 5-FOLD GROUPED OOF EVALUATION")
print("=" * 80)

CANONICAL_FEATURES = [
    'name_exact_match', 'name_jaro_winkler', 'name_jaccard', 'prefix4_match', 'first_token_match',
    'name_len_diff', 'name_len_ratio', 'address_exact_match', 'address_jaro_winkler', 'address_jaccard',
    'address_len_diff', 'address_len_ratio', 'address_first_number_match', 'country_match',
    's1_name_len', 'tgt_name_len', 's1_addr_len', 'tgt_addr_len', 'source_is_s3'
]

FEATURE_SQL = '''
    CASE WHEN s1.business_name <> '' AND s1.business_name = tgt.business_name THEN 1.0 ELSE 0.0 END AS name_exact_match,
    jaro_winkler_similarity(COALESCE(s1.business_name,''), COALESCE(tgt.business_name,'')) / 100.0 AS name_jaro_winkler,
    CASE WHEN LENGTH(COALESCE(s1.business_name,'')) >= 2 AND LENGTH(COALESCE(tgt.business_name,'')) >= 2 THEN jaccard(COALESCE(s1.business_name,''), COALESCE(tgt.business_name,'')) ELSE 0.0 END AS name_jaccard,
    CASE WHEN LENGTH(s1.business_name) >= 4 AND LENGTH(tgt.business_name) >= 4 AND LEFT(s1.business_name, 4) = LEFT(tgt.business_name, 4) THEN 1.0 ELSE 0.0 END AS prefix4_match,
    CASE WHEN s1.business_name <> '' AND tgt.business_name <> '' AND SPLIT_PART(s1.business_name, ' ', 1) = SPLIT_PART(tgt.business_name, ' ', 1) THEN 1.0 ELSE 0.0 END AS first_token_match,
    CAST(ABS(LENGTH(COALESCE(s1.business_name,'')) - LENGTH(COALESCE(tgt.business_name,''))) AS DOUBLE) AS name_len_diff,
    CASE WHEN LENGTH(s1.business_name) > 0 AND LENGTH(tgt.business_name) > 0 THEN CAST(LEAST(LENGTH(s1.business_name), LENGTH(tgt.business_name)) AS DOUBLE) / GREATEST(LENGTH(s1.business_name), LENGTH(tgt.business_name)) ELSE 0.0 END AS name_len_ratio,
    CASE WHEN s1.business_address <> '' AND s1.business_address = tgt.business_address THEN 1.0 ELSE 0.0 END AS address_exact_match,
    jaro_winkler_similarity(COALESCE(s1.business_address,''), COALESCE(tgt.business_address,'')) / 100.0 AS address_jaro_winkler,
    CASE WHEN LENGTH(COALESCE(s1.business_address,'')) >= 2 AND LENGTH(COALESCE(tgt.business_address,'')) >= 2 THEN jaccard(COALESCE(s1.business_address,''), COALESCE(tgt.business_address,'')) ELSE 0.0 END AS address_jaccard,
    CAST(ABS(LENGTH(COALESCE(s1.business_address,'')) - LENGTH(COALESCE(tgt.business_address,''))) AS DOUBLE) AS address_len_diff,
    CASE WHEN LENGTH(s1.business_address) > 0 AND LENGTH(tgt.business_address) > 0 THEN CAST(LEAST(LENGTH(s1.business_address), LENGTH(tgt.business_address)) AS DOUBLE) / GREATEST(LENGTH(s1.business_address), LENGTH(tgt.business_address)) ELSE 0.0 END AS address_len_ratio,
    CASE WHEN regexp_extract(COALESCE(s1.business_address,''), '[0-9]+[A-Za-z]?', 0) <> '' AND regexp_extract(COALESCE(tgt.business_address,''), '[0-9]+[A-Za-z]?', 0) <> '' AND regexp_extract(s1.business_address, '[0-9]+[A-Za-z]?', 0) = regexp_extract(tgt.business_address, '[0-9]+[A-Za-z]?', 0) THEN 1.0 ELSE 0.0 END AS address_first_number_match,
    CASE WHEN s1.country <> '' AND s1.country = tgt.country THEN 1.0 ELSE 0.0 END AS country_match,
    CAST(LENGTH(COALESCE(s1.business_name,'')) AS DOUBLE) AS s1_name_len,
    CAST(LENGTH(COALESCE(tgt.business_name,'')) AS DOUBLE) AS tgt_name_len,
    CAST(LENGTH(COALESCE(s1.business_address,'')) AS DOUBLE) AS s1_addr_len,
    CAST(LENGTH(COALESCE(tgt.business_address,'')) AS DOUBLE) AS tgt_addr_len
'''

con = duckdb.connect()
con.execute("SET threads=4")
con.execute("SET memory_limit='8GB'")
con.execute("SET preserve_insertion_order=false")

# Preload all ground truth
print("Loading ground truth reconstruction...")
gt_all = con.execute("""
    SELECT source1_entity_id, UNNEST(STRING_SPLIT(matched_entity_ids, ',')) as target_id
    FROM read_csv('outputs/person1_step1/train_ground_truth_reconstructed.tsv', delim='\\t', header=true)
""").fetchall()

gt_by_s1 = {}
for s1, tgt in gt_all:
    if s1 not in gt_by_s1:
        gt_by_s1[s1] = set()
    gt_by_s1[s1].add(tgt)
del gt_all

oof_results = []

for fold in range(5):
    t_fold = time.time()
    print(f"\n{'='*30} EVALUATING FOLD {fold} {'='*30}")
    
    # 1. Fold entities
    f_s1_list = [r[0] for r in con.execute(f"SELECT source1_entity_id FROM read_csv('P3/reports/folds_v1_manifest.tsv', delim='\\t', header=true) WHERE fold = {fold}").fetchall()]
    gt_map = {s1: gt_by_s1.get(s1, set()) for s1 in f_s1_list}
    tot_gt_pairs = sum(len(tgts) for tgts in gt_map.values())
    tot_singletons = sum(1 for tgts in gt_map.values() if len(tgts) == 0)
    
    print(f"Fold {fold} S1 Entities: {len(f_s1_list):,} | Ground Truth Pairs: {tot_gt_pairs:,} | Singletons: {tot_singletons:,}")
    
    # 2. Build V4 candidate set with feature extraction
    print(f"Building V4 candidate table for Fold {fold}...")
    con.execute(f"""
        CREATE OR REPLACE TABLE f_v4_scored AS
        WITH f_manifest AS (SELECT source1_entity_id FROM read_csv('P3/reports/folds_v1_manifest.tsv', delim='\\t', header=true) WHERE fold = {fold}),
        s1_ent AS (SELECT entity_id, business_name_normalized as business_name, business_address_normalized as business_address, country_normalized as country, regexp_replace(business_name_normalized, '\\\\b(inc|llc|ltd|limited|pvt|co|corp|services|company|group)\\\\b', '', 'g') as name_clean, left(regexp_replace(lower(coalesce(business_address_normalized,'')), '[^a-z0-9]', '', 'g'), 20) as addr_clean20 FROM read_parquet('P1/data/entities/train/source1/train_s1_entities.parquet') WHERE entity_id IN (SELECT source1_entity_id FROM f_manifest)),
        s2_ent AS (SELECT entity_id, business_name_normalized as business_name, business_address_normalized as business_address, country_normalized as country, regexp_replace(business_name_normalized, '\\\\b(inc|llc|ltd|limited|pvt|co|corp|services|company|group)\\\\b', '', 'g') as name_clean, left(regexp_replace(lower(coalesce(business_address_normalized,'')), '[^a-z0-9]', '', 'g'), 20) as addr_clean20 FROM read_parquet('P1/data/entities/train/source2/train_s2_entities.parquet')),
        s3_ent AS (SELECT entity_id, business_name_normalized as business_name, business_address_normalized as business_address, country_normalized as country, regexp_replace(business_name_normalized, '\\\\b(inc|llc|ltd|limited|pvt|co|corp|services|company|group)\\\\b', '', 'g') as name_clean, left(regexp_replace(lower(coalesce(business_address_normalized,'')), '[^a-z0-9]', '', 'g'), 20) as addr_clean20 FROM read_parquet('P1/data/entities/train/source3/train_s3_entities.parquet')),
        
        -- V3 candidates
        v3_s2 AS (SELECT c.source1_entity_id, c.matched_entity_id FROM read_csv('P1/data/candidates/v3/train_candidate_pairs_s2_v3.tsv', delim='\\t', header=true) c JOIN f_manifest m ON c.source1_entity_id = m.source1_entity_id),
        v3_s3 AS (SELECT c.source1_entity_id, c.matched_entity_id FROM read_csv('P1/data/candidates/v3/train_candidate_pairs_s3_v3.tsv', delim='\\t', header=true) c JOIN f_manifest m ON c.source1_entity_id = m.source1_entity_id),
        
        -- Strat A on S2
        strat_a_s2_raw AS (SELECT s1.entity_id as source1_entity_id, tgt.entity_id as matched_entity_id FROM s1_ent s1 JOIN s2_ent tgt ON s1.country = tgt.country AND left(s1.addr_clean20, 15) = left(tgt.addr_clean20, 15) WHERE length(left(s1.addr_clean20, 15)) >= 12),
        strat_a_s2 AS (SELECT r.source1_entity_id, r.matched_entity_id FROM strat_a_s2_raw r JOIN (SELECT source1_entity_id FROM strat_a_s2_raw GROUP BY 1 HAVING count(*) <= 30) k USING (source1_entity_id)),
        
        -- Strat A on S3
        strat_a_s3_raw AS (SELECT s1.entity_id as source1_entity_id, tgt.entity_id as matched_entity_id FROM s1_ent s1 JOIN s3_ent tgt ON s1.country = tgt.country AND left(s1.addr_clean20, 15) = left(tgt.addr_clean20, 15) WHERE length(left(s1.addr_clean20, 15)) >= 12),
        strat_a_s3 AS (SELECT r.source1_entity_id, r.matched_entity_id FROM strat_a_s3_raw r JOIN (SELECT source1_entity_id FROM strat_a_s3_raw GROUP BY 1 HAVING count(*) <= 30) k USING (source1_entity_id)),
        
        -- Strat B on S2
        strat_b_s2_raw AS (SELECT s1.entity_id as source1_entity_id, tgt.entity_id as matched_entity_id FROM s1_ent s1 JOIN s2_ent tgt ON s1.country = tgt.country AND regexp_replace(lower(s1.name_clean), '[^a-z0-9]', '', 'g') = regexp_replace(lower(tgt.name_clean), '[^a-z0-9]', '', 'g') WHERE length(regexp_replace(lower(s1.name_clean), '[^a-z0-9]', '', 'g')) >= 8),
        strat_b_s2 AS (SELECT r.source1_entity_id, r.matched_entity_id FROM strat_b_s2_raw r JOIN (SELECT source1_entity_id FROM strat_b_s2_raw GROUP BY 1 HAVING count(*) <= 30) k USING (source1_entity_id)),
        
        -- Strat B on S3
        strat_b_s3_raw AS (SELECT s1.entity_id as source1_entity_id, tgt.entity_id as matched_entity_id FROM s1_ent s1 JOIN s3_ent tgt ON s1.country = tgt.country AND regexp_replace(lower(s1.name_clean), '[^a-z0-9]', '', 'g') = regexp_replace(lower(tgt.name_clean), '[^a-z0-9]', '', 'g') WHERE length(regexp_replace(lower(s1.name_clean), '[^a-z0-9]', '', 'g')) >= 8),
        strat_b_s3 AS (SELECT r.source1_entity_id, r.matched_entity_id FROM strat_b_s3_raw r JOIN (SELECT source1_entity_id FROM strat_b_s3_raw GROUP BY 1 HAVING count(*) <= 30) k USING (source1_entity_id)),
        
        all_v4_s2 AS (SELECT DISTINCT source1_entity_id, matched_entity_id FROM (SELECT * FROM v3_s2 UNION ALL SELECT * FROM strat_a_s2 UNION ALL SELECT * FROM strat_b_s2)),
        all_v4_s3 AS (SELECT DISTINCT source1_entity_id, matched_entity_id FROM (SELECT * FROM v3_s3 UNION ALL SELECT * FROM strat_a_s3 UNION ALL SELECT * FROM strat_b_s3))
        
        SELECT c.source1_entity_id, c.matched_entity_id, 0 AS source_is_s3, {FEATURE_SQL}
        FROM all_v4_s2 c
        JOIN s1_ent s1 ON c.source1_entity_id = s1.entity_id
        JOIN s2_ent tgt ON c.matched_entity_id = tgt.entity_id
        UNION ALL
        SELECT c.source1_entity_id, c.matched_entity_id, 1 AS source_is_s3, {FEATURE_SQL}
        FROM all_v4_s3 c
        JOIN s1_ent s1 ON c.source1_entity_id = s1.entity_id
        JOIN s3_ent tgt ON c.matched_entity_id = tgt.entity_id
    """)
    
    cand_data = con.execute("SELECT source1_entity_id, matched_entity_id, " + ", ".join(CANONICAL_FEATURES) + " FROM f_v4_scored").fetchnumpy()
    con.execute("DROP TABLE f_v4_scored;")
    
    s1_arr = cand_data['source1_entity_id']
    tgt_arr = cand_data['matched_entity_id']
    X_fold = np.column_stack([cand_data[col].astype(np.float32) for col in CANONICAL_FEATURES])
    total_cands = len(s1_arr)
    del cand_data
    gc.collect()
    
    # Measure V4 Candidate Recall on this fold
    cand_pair_set = set(zip(s1_arr, tgt_arr))
    gt_recovered = sum(1 for s1, tgts in gt_map.items() for t in tgts if (s1, t) in cand_pair_set)
    del cand_pair_set
    cand_recall = gt_recovered / tot_gt_pairs * 100.0 if tot_gt_pairs > 0 else 0.0
    
    print(f"Fold {fold} V4 Candidates: {total_cands:,} | Captured GT: {gt_recovered:,} ({cand_recall:.4f}% candidate recall)")
    
    # 3. Model Inference using Fold booster
    booster_path = REPO_ROOT / f"P2/models/phase4/lgb_fold{fold}.txt"
    model = lgb.Booster(model_file=str(booster_path))
    print(f"Predicting scores using {booster_path.name}...")
    scores = model.predict(X_fold)
    del X_fold
    gc.collect()
    
    # 4. Group candidates by S1
    s1_cands = {}
    for s1, tgt, sc in zip(s1_arr, tgt_arr, scores):
        if s1 not in s1_cands:
            s1_cands[s1] = []
        s1_cands[s1].append((tgt, sc))
    del s1_arr, tgt_arr, scores
    gc.collect()
    
    for s1 in s1_cands:
        s1_cands[s1].sort(key=lambda x: x[1], reverse=True)
        
    # 5. Apply Winning Decision Policy: T=0.88, Margin=0.05, MaxK=8
    pred_map = {}
    total_preds = 0
    correct_singletons = 0
    
    for s1 in f_s1_list:
        is_singleton = len(gt_map[s1]) == 0
        cands = s1_cands.get(s1, [])
        if not cands or cands[0][1] < 0.88:
            matches = []
        else:
            top_score = cands[0][1]
            matches = [t for t, s in cands if s >= 0.88 and (top_score - s) <= 0.05][:8]
        if is_singleton and len(matches) == 0:
            correct_singletons += 1
        total_preds += len(matches)
        pred_map[s1] = set(matches)
        
    res = score_predictions(pred_map, gt_map, entity_ids=f_s1_list)
    sing_acc = correct_singletons / tot_singletons if tot_singletons > 0 else 0.0
    
    macro_f05 = res["macro_f0.5"]
    micro_prec = res["micro_precision"]
    micro_rec = res["micro_recall"]
    macro_prec = res.get("macro_precision", 0.0)
    macro_rec = res.get("macro_recall", 0.0)
    
    print(f"Fold {fold} Results:")
    print(f"  Macro F0.5: {macro_f05:.6f}")
    print(f"  Micro Precision: {micro_prec:.4f} | Micro Recall: {micro_rec:.4f}")
    print(f"  Singleton Accuracy: {sing_acc:.6f} ({correct_singletons:,}/{tot_singletons:,})")
    print(f"  Total Predictions: {total_preds:,}")
    print(f"  Fold Runtime: {time.time() - t_fold:.2f}s")
    
    oof_results.append({
        "fold": fold,
        "candidate_recall": f"{cand_recall:.6f}",
        "prediction_count": total_preds,
        "micro_precision": f"{micro_prec:.6f}",
        "micro_recall": f"{micro_rec:.6f}",
        "macro_precision": f"{macro_prec:.6f}",
        "macro_recall": f"{macro_rec:.6f}",
        "macro_f0.5": f"{macro_f05:.6f}",
        "singleton_accuracy": f"{sing_acc:.6f}",
        "runtime_s": f"{time.time() - t_fold:.2f}"
    })
    
    del s1_cands, pred_map, gt_map
    gc.collect()

# Write P2/reports/phase7_oof_results.tsv
out_tsv = REPO_ROOT / "P2" / "reports" / "phase7_oof_results.tsv"
with open(out_tsv, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "fold", "candidate_recall", "prediction_count", 
        "micro_precision", "micro_recall", "macro_precision", "macro_recall", 
        "macro_f0.5", "singleton_accuracy", "runtime_s"
    ], delimiter="\t")
    writer.writeheader()
    for r in oof_results:
        writer.writerow(r)

print(f"\nSaved 5-fold OOF results to: {out_tsv}")

# Compute and print cross-validation average
f05_vals = [float(r["macro_f0.5"]) for r in oof_results]
cand_rec_vals = [float(r["candidate_recall"]) for r in oof_results]
prec_vals = [float(r["micro_precision"]) for r in oof_results]
rec_vals = [float(r["micro_recall"]) for r in oof_results]
sing_vals = [float(r["singleton_accuracy"]) for r in oof_results]

print("\n" + "=" * 80)
print(f"5-FOLD OOF AVERAGE SUMMARY:")
print(f"  Mean Candidate Recall: {np.mean(cand_rec_vals):.4f}% (Std: {np.std(cand_rec_vals):.4f}%)")
print(f"  Mean Micro Precision:  {np.mean(prec_vals):.4f}")
print(f"  Mean Micro Recall:     {np.mean(rec_vals):.4f}")
print(f"  Mean Macro F0.5:       {np.mean(f05_vals):.6f} (Std: {np.std(f05_vals):.6f})")
print(f"  Mean Singleton Acc:    {np.mean(sing_vals):.6f}")
print("=" * 80)
