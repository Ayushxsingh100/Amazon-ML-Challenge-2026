import os
import time
import duckdb
import numpy as np
import pandas as pd
import json

REPO = r"c:\NEW AMAZON\Amazon-ML-Challenge-2026"
FOLDS_PATH = os.path.join(REPO, "P3", "reports", "folds_v1_manifest.tsv")
GT_PATH = os.path.join(REPO, "outputs", "person1_step1", "train_ground_truth_reconstructed.tsv")
REPORT_DIR = os.path.join(REPO, "P2", "reports")

con = duckdb.connect()
con.execute("SET threads=4")
con.execute("SET memory_limit='4GB'")

t0 = time.time()
print("Step 1: Loading folds manifest and ground truth counts...")
con.execute(f"CREATE TABLE folds AS SELECT source1_entity_id, fold FROM read_csv('{FOLDS_PATH}', delim='\\t', header=true)")
con.execute(f"""
    CREATE TABLE gt_counts AS 
    SELECT 
        source1_entity_id, 
        CASE WHEN matched_entity_ids IS NULL OR matched_entity_ids = '' THEN 0 ELSE len(string_split(matched_entity_ids, ',')) END AS A
    FROM read_csv('{GT_PATH}', delim='\\t', header=true)
""")

total_gt = con.execute("SELECT sum(A) FROM gt_counts").fetchone()[0]
print(f"Total ground truth matches: {total_gt}")

# Pre-registered threshold grid: 0.500 to 0.995 in steps of 0.005 (100 thresholds)
thresholds = [round(0.500 + i * 0.005, 3) for i in range(100)]
print(f"Threshold grid: {len(thresholds)} thresholds from {thresholds[0]:.3f} to {thresholds[-1]:.3f}")

# Prepare fold entity mapping
fold_entity_counts = {}
fold_entity_maps = {}
fold_A_arrays = {}

for f in range(5):
    df_fold = con.execute(f"""
        SELECT f.source1_entity_id, coalesce(g.A, 0) AS A
        FROM folds f
        LEFT JOIN gt_counts g ON f.source1_entity_id = g.source1_entity_id
        WHERE f.fold = {f}
        ORDER BY f.source1_entity_id
    """).df()
    
    n_f = len(df_fold)
    fold_entity_counts[f] = n_f
    # Map entity_id -> integer index [0, n_f - 1]
    ent_map = {ent_id: i for i, ent_id in enumerate(df_fold['source1_entity_id'].values)}
    fold_entity_maps[f] = ent_map
    fold_A_arrays[f] = df_fold['A'].values.astype(np.int16)
    print(f"Fold {f}: {n_f} entities loaded")

# Evaluate each fold across all 100 thresholds
# Store sum of F0.5 per fold per threshold: shape (5, 100)
# Also store total P and TP per fold per threshold
sum_f05_matrix = np.zeros((5, len(thresholds)), dtype=np.float64)
p_matrix = np.zeros((5, len(thresholds)), dtype=np.int64)
tp_matrix = np.zeros((5, len(thresholds)), dtype=np.int64)

for f in range(5):
    t_f = time.time()
    oof_shard = os.path.join(REPORT_DIR, f"E02_OOF_FULL_CANONICAL_fold{f}.tsv")
    print(f"\nProcessing Fold {f} candidates >= 0.500 from {os.path.basename(oof_shard)}...")
    
    # Read only candidates with oof_score >= 0.500
    df_cands = con.execute(f"""
        SELECT source1_entity_id, target, oof_score
        FROM read_csv('{oof_shard}', delim='\\t', header=true)
        WHERE oof_score >= 0.500
    """).df()
    
    n_cands = len(df_cands)
    print(f"  Loaded {n_cands} candidates with oof_score >= 0.500 in {time.time() - t_f:.2f}s")
    
    # Map to integer indices
    ent_map = fold_entity_maps[f]
    s1_indices = np.array([ent_map[eid] for eid in df_cands['source1_entity_id'].values], dtype=np.int32)
    targets = df_cands['target'].values.astype(np.int8)
    scores = df_cands['oof_score'].values.astype(np.float32)
    A = fold_A_arrays[f]
    n_entities = fold_entity_counts[f]
    
    # Sweep all 100 thresholds
    t_sweep = time.time()
    for t_idx, t in enumerate(thresholds):
        mask = scores >= (t - 1e-7)  # safety epsilon for float comparison
        p_idx = s1_indices[mask]
        p_tgt = targets[mask]
        
        P = np.bincount(p_idx, minlength=n_entities).astype(np.float32)
        TP = np.bincount(p_idx, weights=p_tgt, minlength=n_entities).astype(np.float32)
        
        # Scorer_v1 semantics:
        # A == 0: 1.0 if P == 0 else 0.0
        # A > 0: (5 * TP) / (A + 4 * P) if TP > 0 else 0.0
        denom = A + 4.0 * P
        f05 = np.zeros(n_entities, dtype=np.float64)
        
        # Where A == 0
        mask_a0 = (A == 0)
        f05[mask_a0 & (P == 0)] = 1.0
        
        # Where A > 0 and TP > 0
        mask_pos = (~mask_a0) & (TP > 0)
        f05[mask_pos] = (5.0 * TP[mask_pos]) / denom[mask_pos]
        
        sum_f05_matrix[f, t_idx] = np.sum(f05)
        p_matrix[f, t_idx] = np.sum(P)
        tp_matrix[f, t_idx] = np.sum(TP)
        
    print(f"  Sweep completed in {time.time() - t_sweep:.2f}s (Macro F0.5 at T=0.50: {sum_f05_matrix[f, 0]/n_entities:.9f})")

print("\n" + "=" * 60)
print("Step 2: Nested Threshold Selection")
print("=" * 60)

nested_results = []
total_dev_entities = sum(fold_entity_counts.values())

for f in range(5):
    # Other 4 folds
    dev_folds = [k for k in range(5) if k != f]
    dev_entities = sum(fold_entity_counts[k] for k in dev_folds)
    
    # Macro F0.5 on dev folds for each threshold
    dev_macro_f05 = np.sum(sum_f05_matrix[dev_folds, :], axis=0) / dev_entities
    
    # Best threshold on dev folds (tie-break: lower threshold)
    best_idx = int(np.argmax(dev_macro_f05))
    best_thresh = thresholds[best_idx]
    best_dev_score = dev_macro_f05[best_idx]
    
    # Held-out score on fold f using the frozen selected threshold
    held_out_score = sum_f05_matrix[f, best_idx] / fold_entity_counts[f]
    
    # E02 baseline score on fold f at T=0.500 (index 0)
    baseline_score = sum_f05_matrix[f, 0] / fold_entity_counts[f]
    
    # Score at T=0.900 (index 80: 0.500 + 80*0.005 = 0.900)
    idx_090 = thresholds.index(0.900)
    score_090 = sum_f05_matrix[f, idx_090] / fold_entity_counts[f]
    
    improvement = held_out_score - baseline_score
    
    nested_results.append({
        "held_out_fold": f,
        "n_entities": fold_entity_counts[f],
        "selected_threshold": best_thresh,
        "dev_macro_f05": best_dev_score,
        "held_out_macro_f05": held_out_score,
        "baseline_f05_at_050": baseline_score,
        "diagnostic_f05_at_090": score_090,
        "delta_vs_baseline": improvement,
        "delta_vs_090": held_out_score - score_090
    })
    
    print(f"Held-out Fold {f}:")
    print(f"  Selected threshold (from folds {dev_folds}): T = {best_thresh:.3f} (dev F0.5 = {best_dev_score:.6f})")
    print(f"  Held-out Macro F0.5: {held_out_score:.9f}")
    print(f"  E02 Baseline (T=0.50): {baseline_score:.9f} (Diff: {improvement:+.6f})")
    print(f"  Diagnostic (T=0.90):   {score_090:.9f}")

df_nested = pd.DataFrame(nested_results)

# Overall nested summary
overall_nested_f05 = sum(df_nested['held_out_macro_f05'] * df_nested['n_entities']) / total_dev_entities
fold_mean = df_nested['held_out_macro_f05'].mean()
fold_std = df_nested['held_out_macro_f05'].std(ddof=1)
overall_baseline = sum(df_nested['baseline_f05_at_050'] * df_nested['n_entities']) / total_dev_entities
overall_090 = sum(df_nested['diagnostic_f05_at_090'] * df_nested['n_entities']) / total_dev_entities
overall_gain = overall_nested_f05 - overall_baseline

print("-" * 60)
print(f"OVERALL NESTED MACRO F0.5: {overall_nested_f05:.9f}")
print(f"FOLD MEAN +/- STD:         {fold_mean:.9f} +/- {fold_std:.9f}")
print(f"E02 BASELINE (T=0.50):     {overall_baseline:.9f}")
print(f"OVERALL IMPROVEMENT:       {overall_gain:+.9f} ({overall_gain*100/overall_baseline:+.2f}%)")
print(f"DIAGNOSTIC (T=0.90):       {overall_090:.9f}")
print(f"DIFF VS T=0.90:            {overall_nested_f05 - overall_090:+.9f}")

# Save results TSV
out_results_tsv = os.path.join(REPORT_DIR, "E03_NESTED_THRESHOLD_RESULTS.tsv")
df_nested.to_csv(out_results_tsv, sep='\t', index=False)
print(f"\nSaved nested results to {out_results_tsv}")

# Step 3: Diagnostic Threshold Curve across all 2.2M entities
print("\nStep 3: Generating diagnostic threshold curve across all canonical OOF...")
curve_rows = []
for t_idx, t in enumerate(thresholds):
    tot_sum_f05 = np.sum(sum_f05_matrix[:, t_idx])
    global_macro = tot_sum_f05 / total_dev_entities
    fold_scores = sum_f05_matrix[:, t_idx] / [fold_entity_counts[k] for k in range(5)]
    f_mean = np.mean(fold_scores)
    f_std = np.std(fold_scores, ddof=1)
    
    tot_P = int(np.sum(p_matrix[:, t_idx]))
    tot_TP = int(np.sum(tp_matrix[:, t_idx]))
    tot_FP = tot_P - tot_TP
    tot_FN = int(total_gt - tot_TP)
    
    micro_p = tot_TP / tot_P if tot_P > 0 else 0.0
    micro_r = tot_TP / total_gt
    micro_f05 = (1.25 * micro_p * micro_r) / (0.25 * micro_p + micro_r) if (0.25 * micro_p + micro_r) > 0 else 0.0
    
    curve_rows.append({
        "threshold": t,
        "global_macro_f05": global_macro,
        "fold_mean": f_mean,
        "fold_std": f_std,
        "fold0_macro_f05": fold_scores[0],
        "fold1_macro_f05": fold_scores[1],
        "fold2_macro_f05": fold_scores[2],
        "fold3_macro_f05": fold_scores[3],
        "fold4_macro_f05": fold_scores[4],
        "predicted_matches": tot_P,
        "tp": tot_TP,
        "fp": tot_FP,
        "fn": tot_FN,
        "micro_precision": micro_p,
        "micro_recall": micro_r,
        "micro_f05": micro_f05
    })

df_curve = pd.DataFrame(curve_rows)
out_curve_tsv = os.path.join(REPORT_DIR, "E03_THRESHOLD_CURVE.tsv")
df_curve.to_csv(out_curve_tsv, sep='\t', index=False)
print(f"Saved threshold curve to {out_curve_tsv}")

print(f"Total time: {time.time() - t0:.2f}s")
