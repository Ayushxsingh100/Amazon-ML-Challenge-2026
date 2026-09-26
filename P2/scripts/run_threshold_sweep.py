import os
import time
import duckdb
import numpy as np
import pandas as pd

REPO = r"c:\NEW AMAZON\Amazon-ML-Challenge-2026"
FOLDS_PATH = os.path.join(REPO, "P3", "reports", "folds_v1_manifest.tsv")
GT_PATH = os.path.join(REPO, "outputs", "person1_step1", "train_ground_truth_reconstructed.tsv")
REPORT_DIR = os.path.join(REPO, "P2", "reports")
OUT_TSV = os.path.join(REPORT_DIR, "E02_FULL_OOF_THRESHOLD_SWEEP.tsv")

con = duckdb.connect()
con.execute("SET threads=4")
con.execute("SET memory_limit='4GB'")

t0 = time.time()
print("Loading ground truth counts and folds...")
con.execute(f"CREATE TABLE s1_all AS SELECT source1_entity_id, fold FROM read_csv('{FOLDS_PATH}', delim='\\t', header=true)")
con.execute(f"""
    CREATE TABLE gt_counts AS 
    SELECT 
        source1_entity_id, 
        CASE WHEN matched_entity_ids IS NULL OR matched_entity_ids = '' THEN 0 ELSE len(string_split(matched_entity_ids, ',')) END AS A
    FROM read_csv('{GT_PATH}', delim='\\t', header=true)
""")

total_gt_pairs = con.execute("SELECT sum(A) FROM gt_counts").fetchone()[0]
print(f"Total ground truth pairs: {total_gt_pairs}")

thresholds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]

# To be fast, let's load for each fold only rows with score >= 0.05 into memory / temp table
fold_dfs = []
for f in range(5):
    t_f = time.time()
    oof_shard = os.path.join(REPORT_DIR, f"E02_OOF_FULL_CANONICAL_fold{f}.tsv")
    print(f"Loading Fold {f} candidates with score >= 0.05...")
    con.execute(f"""
        CREATE OR REPLACE TABLE f{f}_cands AS
        SELECT source1_entity_id, target, oof_score
        FROM read_csv('{oof_shard}', delim='\\t', header=true)
        WHERE oof_score >= 0.05
    """)
    print(f"Fold {f} ready in {time.time() - t_f:.2f}s")

sweep_rows = []

for t in thresholds:
    t_step = time.time()
    fold_scores = []
    tot_P = 0
    tot_TP = 0
    total_f05_sum = 0.0
    
    for f in range(5):
        # Aggregate at threshold t
        con.execute(f"""
            CREATE OR REPLACE TABLE fold_p AS
            SELECT 
                source1_entity_id,
                count(*) AS P,
                sum(target) AS TP
            FROM f{f}_cands
            WHERE oof_score >= {t}
            GROUP BY source1_entity_id
        """)
        
        res = con.execute(f"""
            SELECT 
                count(*),
                sum(CASE 
                    WHEN coalesce(g.A, 0) = 0 THEN 
                        CASE WHEN coalesce(p.P, 0) = 0 THEN 1.0 ELSE 0.0 END
                    ELSE 
                        CASE WHEN coalesce(p.TP, 0) = 0 THEN 0.0 
                        ELSE (5.0 * p.TP) / (g.A + 4.0 * p.P) END
                END) AS sum_f05,
                sum(coalesce(p.P, 0)) AS fold_P,
                sum(coalesce(p.TP, 0)) AS fold_TP
            FROM s1_all s
            LEFT JOIN gt_counts g ON s.source1_entity_id = g.source1_entity_id
            LEFT JOIN fold_p p ON s.source1_entity_id = p.source1_entity_id
            WHERE s.fold = {f}
        """).fetchone()
        
        n_ent, s_f05, f_P, f_TP = res
        fold_m_f05 = s_f05 / n_ent
        fold_scores.append(fold_m_f05)
        total_f05_sum += s_f05
        tot_P += int(f_P)
        tot_TP += int(f_TP)
        
    overall_macro_f05 = total_f05_sum / 2206821
    fold_mean = np.mean(fold_scores)
    fold_std = np.std(fold_scores, ddof=1)
    tot_FP = tot_P - tot_TP
    tot_FN = total_gt_pairs - tot_TP
    micro_p = tot_TP / tot_P if tot_P > 0 else 0.0
    micro_r = tot_TP / total_gt_pairs
    micro_f05 = (1.25 * micro_p * micro_r) / (0.25 * micro_p + micro_r) if (0.25 * micro_p + micro_r) > 0 else 0.0
    
    sweep_rows.append({
        "threshold": t,
        "macro_f05": overall_macro_f05,
        "fold_mean": fold_mean,
        "fold_std": fold_std,
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
        "micro_f05": micro_f05,
    })
    print(f"T={t:.2f}: Macro F0.5={overall_macro_f05:.6f} (fold mean={fold_mean:.6f} +/- {fold_std:.6f}), P={tot_P:,}, TP={tot_TP:,}, FP={tot_FP:,}, FN={tot_FN:,}, time={time.time() - t_step:.2f}s")

df_sweep = pd.DataFrame(sweep_rows)
df_sweep.to_csv(OUT_TSV, sep='\t', index=False)
print(f"Saved threshold sweep to {OUT_TSV}")
print(f"Total time: {time.time() - t0:.2f}s")
