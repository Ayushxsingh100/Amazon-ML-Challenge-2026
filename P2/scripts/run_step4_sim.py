import os
import time
import duckdb
import numpy as np

repo_root = "c:/NEW AMAZON"
oof_pattern = "E:/predictions/phase4/v3_train_oof_fold*.parquet"
gt_path = os.path.join(repo_root, "Amazon-ML-Challenge-2026", "outputs", "person1_step1", "train_ground_truth_reconstructed.tsv")

con = duckdb.connect()
con.execute("PRAGMA threads=8;")
con.execute("PRAGMA memory_limit='24GB';")

print("--- Step 4: OOF Simulation ---")
t0 = time.time()

# 1. Load ground truth
con.execute(f"""
CREATE TEMP TABLE train_gt_raw AS
SELECT 
    source1_entity_id as s1_id,
    COALESCE(matched_entity_ids, '') as matched_entity_ids
FROM read_csv('{gt_path.replace(os.sep, '/')}', delim='\t', header=True, all_varchar=True);
""")

# 2. Find best candidate score per S1 in OOF
# To be memory efficient, we can get top-1 S2 and top-1 S3 per S1 and max score per S1
print("Finding max score per S1 in OOF...")
con.execute(f"""
CREATE TEMP TABLE s1_best_oof AS
SELECT 
    source1_entity_id as s1_id,
    MAX(score) as best_score
FROM '{oof_pattern}'
GROUP BY source1_entity_id;
""")

total_oof_s1 = con.execute("SELECT COUNT(*) FROM s1_best_oof").fetchone()[0]
lt_098_s1 = con.execute("SELECT COUNT(*) FROM s1_best_oof WHERE best_score < 0.98").fetchone()[0]
print(f"Total OOF S1: {total_oof_s1}")
print(f"OOF S1 with best score < 0.98: {lt_098_s1} ({lt_098_s1/total_oof_s1*100:.2f}%)")

# Filter eligible S1s
con.execute("""
CREATE TEMP TABLE eligible_s1 AS
SELECT s1_id, best_score
FROM s1_best_oof
WHERE best_score < 0.98;
""")

# Get top-1 S2 and top-1 S3 for these eligible S1s from OOF
print("Extracting top-1 S2 and top-1 S3 for eligible S1s...")
con.execute(f"""
CREATE TEMP TABLE eligible_cands AS
SELECT 
    o.source1_entity_id as s1_id,
    o.candidate_entity_id as candidate_id,
    o.score,
    o.label,
    CASE WHEN o.candidate_entity_id LIKE 'S2-%' THEN 'S2' ELSE 'S3' END as target_src
FROM '{oof_pattern}' o
JOIN eligible_s1 e ON o.source1_entity_id = e.s1_id;
""")

con.execute("""
CREATE TEMP TABLE top_cands_per_s1 AS
SELECT 
    s1_id,
    candidate_id,
    score,
    label,
    target_src
FROM (
    SELECT 
        s1_id,
        candidate_id,
        score,
        label,
        target_src,
        ROW_NUMBER() OVER (PARTITION BY s1_id, target_src ORDER BY score DESC, candidate_id ASC) as rnk
    FROM eligible_cands
)
WHERE rnk = 1;
""")

# Ground truth stats for eligible S1s
# Count how many true S2 and true S3 matches each eligible S1 has in ground truth
con.execute("""
CREATE TEMP TABLE gt_exploded AS
WITH split_pairs AS (
    SELECT 
        s1_id,
        unnest(string_split(matched_entity_ids, ',')) as candidate_id
    FROM train_gt_raw
    WHERE matched_entity_ids != ''
)
SELECT 
    s1_id,
    TRIM(candidate_id) as candidate_id,
    CASE WHEN candidate_id LIKE 'S2-%' THEN 'S2' ELSE 'S3' END as target_src
FROM split_pairs;

CREATE TEMP TABLE eligible_gt_counts AS
SELECT 
    e.s1_id,
    COUNT(g.candidate_id) as actual_total,
    COUNT(CASE WHEN g.target_src = 'S2' THEN 1 END) as actual_s2,
    COUNT(CASE WHEN g.target_src = 'S3' THEN 1 END) as actual_s3
FROM eligible_s1 e
LEFT JOIN gt_exploded g ON e.s1_id = g.s1_id
GROUP BY e.s1_id;
""")

# Check how many eligible S1s have actual_total == 0
empty_actual_count = con.execute("SELECT COUNT(*) FROM eligible_gt_counts WHERE actual_total = 0").fetchone()[0]
print(f"Eligible S1s with actual_total == 0 (true no-match): {empty_actual_count} ({empty_actual_count/lt_098_s1*100:.4f}%)")

# Baseline: predict empty for all
# If actual == 0, f0.5 = 1.0 (official rule). If actual > 0, f0.5 = 0.0.
macro_empty_official = empty_actual_count / lt_098_s1
print(f"Predict Empty Macro F0.5 (Official: empty-empty=1): {macro_empty_official:.6f}")
print(f"Predict Empty Macro F0.5 (Alternative: empty-empty=0): 0.000000")

# For each cutoff X in [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]:
# Simulate prediction:
# S2 candidate is predicted if score >= X
# S3 candidate is predicted if score >= X
# For each predicted candidate, label indicates if it is in actual (1 if true, 0 if false).
# Note: since label in OOF indicates whether candidate_id in ground_truth for that s1_id:
# true_positive = sum(label of predicted candidates)
# len(predicted) = count of predicted candidates (0, 1, or 2)
# len(actual) = actual_total
cutoffs = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]

print("\n--- Evaluating Cutoffs ---")
results = []
for X in cutoffs:
    # Compute per-entity metrics
    con.execute(f"""
    WITH preds AS (
        SELECT 
            s1_id,
            COUNT(*) as pred_total,
            SUM(label) as tp
        FROM top_cands_per_s1
        WHERE score >= {X}
        GROUP BY s1_id
    ),
    eval_table AS (
        SELECT 
            g.s1_id,
            g.actual_total,
            COALESCE(p.pred_total, 0) as pred_total,
            COALESCE(p.tp, 0) as tp
        FROM eligible_gt_counts g
        LEFT JOIN preds p ON g.s1_id = p.s1_id
    )
    SELECT 
        -- Official rule: if actual=0 and pred=0 -> 1.0; if actual=0 and pred>0 -> 0.0
        AVG(
            CASE 
                WHEN actual_total = 0 AND pred_total = 0 THEN 1.0
                WHEN actual_total = 0 AND pred_total > 0 THEN 0.0
                WHEN actual_total > 0 AND pred_total = 0 THEN 0.0
                ELSE 
                    -- F0.5 = 1.25 * P * R / (0.25 * P + R)
                    -- P = tp / pred_total, R = tp / actual_total
                    -- If tp == 0 -> 0.0
                    -- 1.25 * (tp/pred) * (tp/actual) / (0.25*(tp/pred) + (tp/actual))
                    -- = 1.25 * tp / (0.25 * actual + pred)
                    CASE WHEN tp = 0 THEN 0.0
                    ELSE (1.25 * tp) / (0.25 * actual_total + pred_total)
                    END
            END
        ) as macro_f05_official,
        -- Alternative rule: if actual=0 and pred=0 -> 0.0
        AVG(
            CASE 
                WHEN actual_total = 0 THEN 0.0
                WHEN pred_total = 0 THEN 0.0
                WHEN tp = 0 THEN 0.0
                ELSE (1.25 * tp) / (0.25 * actual_total + pred_total)
            END
        ) as macro_f05_alt,
        COUNT(CASE WHEN pred_total > 0 THEN 1 END) as entities_predicted,
        SUM(pred_total) as total_pairs_predicted,
        SUM(tp) as total_tp
    FROM eval_table;
    """)
    row = con.fetchone()
    results.append({
        "cutoff": X,
        "macro_f05_official": row[0],
        "macro_f05_alt": row[1],
        "entities_predicted": row[2],
        "total_pairs_predicted": row[3],
        "total_tp": row[4]
    })
    print(f"Cutoff X={X:4.2f} | Official Macro F0.5: {row[0]:.6f} | Alt Macro F0.5: {row[1]:.6f} | Pred Entities: {row[2]:6d} | Total Pairs: {row[3]:6d} | TP: {row[4]:6d}")

print(f"\nBaseline (Empty for all): Official Macro F0.5 = {macro_empty_official:.6f} | Alt Macro F0.5 = 0.000000")
