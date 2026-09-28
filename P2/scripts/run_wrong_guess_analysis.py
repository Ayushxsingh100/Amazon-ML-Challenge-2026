import os
import duckdb

repo_root = "c:/NEW AMAZON"
p2_data = os.path.join(repo_root, "Amazon-ML-Challenge-2026", "P2", "data")
v4_path = os.path.join(p2_data, "matching_results_v4.tsv")
pred_s2 = os.path.join(repo_root, "Amazon-ML-Challenge-2026", "P2", "predictions", "phase4", "test_predictions_s2.tsv")
pred_s3 = os.path.join(repo_root, "Amazon-ML-Challenge-2026", "P2", "predictions", "phase4", "test_predictions_s3.tsv")
gt_path = os.path.join(repo_root, "Amazon-ML-Challenge-2026", "outputs", "person1_step1", "train_ground_truth_reconstructed.tsv")
oof_pattern = "E:/predictions/phase4/v3_train_oof_fold*.parquet"

con = duckdb.connect()
con.execute("PRAGMA threads=8;")
con.execute("PRAGMA memory_limit='24GB';")

print("--- TASK 1: OOF Best Score vs 0 True Matches ---")
# 1. Ground truth 0 true match flag per S1
con.execute(f"""
CREATE TEMP TABLE train_gt_raw AS
SELECT 
    source1_entity_id as s1_id,
    COALESCE(matched_entity_ids, '') as matched_entity_ids,
    CASE WHEN matched_entity_ids IS NULL OR TRIM(matched_entity_ids) = '' THEN 1 ELSE 0 END as is_zero_match
FROM read_csv('{gt_path.replace(os.sep, '/')}', delim='\t', header=True, all_varchar=True);
""")

# 2. Best candidate score per S1 in OOF
print("Computing OOF best score per S1...")
con.execute(f"""
CREATE TEMP TABLE s1_best_oof AS
SELECT 
    source1_entity_id as s1_id,
    MAX(score) as best_score
FROM '{oof_pattern}'
GROUP BY source1_entity_id;
""")

# Check S1 entities in train_gt_raw that are in s1_best_oof
con.execute("""
CREATE TEMP TABLE oof_joined AS
SELECT 
    g.s1_id,
    g.is_zero_match,
    b.best_score,
    CASE WHEN b.best_score IS NULL THEN 1 ELSE 0 END as has_no_oof_cands
FROM train_gt_raw g
LEFT JOIN s1_best_oof b ON g.s1_id = b.s1_id;
""")

no_oof_cands = con.execute("SELECT COUNT(*) FROM oof_joined WHERE has_no_oof_cands = 1").fetchone()[0]
no_oof_cands_zero_match = con.execute("SELECT COUNT(*) FROM oof_joined WHERE has_no_oof_cands = 1 AND is_zero_match = 1").fetchone()[0]
print(f"Train S1 entities with no candidates in OOF: {no_oof_cands} ({no_oof_cands_zero_match} zero match, {no_oof_cands_zero_match/no_oof_cands*100:.2f}%)")

# Buckets: 0-0.05, 0.05-0.2, 0.2-0.5, 0.5-0.8, 0.8-0.9, 0.9-0.98, 0.98-0.995, 0.995-1.0
# Let's compute for OOF entities with candidates:
query_oof = """
SELECT 
    bucket,
    bucket_order,
    COUNT(*) as total_entities,
    SUM(is_zero_match) as zero_match_entities,
    ROUND(SUM(is_zero_match)::DOUBLE / COUNT(*) * 100, 4) as pct_zero_match
FROM (
    SELECT 
        s1_id,
        is_zero_match,
        best_score,
        CASE 
            WHEN best_score >= 0.00 AND best_score < 0.05 THEN '0-0.05'
            WHEN best_score >= 0.05 AND best_score < 0.20 THEN '0.05-0.2'
            WHEN best_score >= 0.20 AND best_score < 0.50 THEN '0.2-0.5'
            WHEN best_score >= 0.50 AND best_score < 0.80 THEN '0.5-0.8'
            WHEN best_score >= 0.80 AND best_score < 0.90 THEN '0.8-0.9'
            WHEN best_score >= 0.90 AND best_score < 0.98 THEN '0.9-0.98'
            WHEN best_score >= 0.98 AND best_score < 0.995 THEN '0.98-0.995'
            WHEN best_score >= 0.995 AND best_score <= 1.000001 THEN '0.995-1.0'
            ELSE 'OTHER'
        END as bucket,
        CASE 
            WHEN best_score >= 0.00 AND best_score < 0.05 THEN 1
            WHEN best_score >= 0.05 AND best_score < 0.20 THEN 2
            WHEN best_score >= 0.20 AND best_score < 0.50 THEN 3
            WHEN best_score >= 0.50 AND best_score < 0.80 THEN 4
            WHEN best_score >= 0.80 AND best_score < 0.90 THEN 5
            WHEN best_score >= 0.90 AND best_score < 0.98 THEN 6
            WHEN best_score >= 0.98 AND best_score < 0.995 THEN 7
            WHEN best_score >= 0.995 AND best_score <= 1.000001 THEN 8
            ELSE 9
        END as bucket_order
    FROM oof_joined
    WHERE has_no_oof_cands = 0
)
GROUP BY bucket, bucket_order
ORDER BY bucket_order;
"""

df_oof_buckets = con.execute(query_oof).df()
print("\nOOF Table (entities with candidates in OOF):")
print(df_oof_buckets.to_string(index=False))

# What if 0-0.05 includes the no-candidate entities (score effectively 0)?
query_oof_all = """
SELECT 
    bucket,
    bucket_order,
    COUNT(*) as total_entities,
    SUM(is_zero_match) as zero_match_entities,
    ROUND(SUM(is_zero_match)::DOUBLE / COUNT(*) * 100, 4) as pct_zero_match
FROM (
    SELECT 
        s1_id,
        is_zero_match,
        COALESCE(best_score, 0.0) as best_score,
        CASE 
            WHEN COALESCE(best_score, 0.0) >= 0.00 AND COALESCE(best_score, 0.0) < 0.05 THEN '0-0.05'
            WHEN best_score >= 0.05 AND best_score < 0.20 THEN '0.05-0.2'
            WHEN best_score >= 0.20 AND best_score < 0.50 THEN '0.2-0.5'
            WHEN best_score >= 0.50 AND best_score < 0.80 THEN '0.5-0.8'
            WHEN best_score >= 0.80 AND best_score < 0.90 THEN '0.8-0.9'
            WHEN best_score >= 0.90 AND best_score < 0.98 THEN '0.9-0.98'
            WHEN best_score >= 0.98 AND best_score < 0.995 THEN '0.98-0.995'
            WHEN best_score >= 0.995 AND best_score <= 1.000001 THEN '0.995-1.0'
            ELSE 'OTHER'
        END as bucket,
        CASE 
            WHEN COALESCE(best_score, 0.0) >= 0.00 AND COALESCE(best_score, 0.0) < 0.05 THEN 1
            WHEN best_score >= 0.05 AND best_score < 0.20 THEN 2
            WHEN best_score >= 0.20 AND best_score < 0.50 THEN 3
            WHEN best_score >= 0.50 AND best_score < 0.80 THEN 4
            WHEN best_score >= 0.80 AND best_score < 0.90 THEN 5
            WHEN best_score >= 0.90 AND best_score < 0.98 THEN 6
            WHEN best_score >= 0.98 AND best_score < 0.995 THEN 7
            WHEN best_score >= 0.995 AND best_score <= 1.000001 THEN 8
            ELSE 9
        END as bucket_order
    FROM oof_joined
)
GROUP BY bucket, bucket_order
ORDER BY bucket_order;
"""
df_oof_all = con.execute(query_oof_all).df()
print("\nOOF Table (including no-candidate entities in 0-0.05):")
print(df_oof_all.to_string(index=False))

# =========================================================
# TASK 2: Test Analysis for Non-Empty V4 Rows
# =========================================================
print("\n--- TASK 2: Test Analysis for Non-Empty V4 Rows ---")
con.execute(f"""
CREATE TEMP TABLE v4_raw AS
SELECT 
    source1_entity_id as s1_id,
    COALESCE(matched_entity_ids, '') as v4_matched_ids,
    CASE WHEN matched_entity_ids IS NULL OR TRIM(matched_entity_ids) = '' THEN 1 ELSE 0 END as is_empty_in_v4
FROM read_csv('{v4_path.replace(os.sep, "/")}', delim='\t', header=True, all_varchar=True);
""")

con.execute("""
CREATE TEMP TABLE v4_nonempty_rows AS
SELECT s1_id, v4_matched_ids
FROM v4_raw
WHERE is_empty_in_v4 = 0;
""")

total_nonempty_v4 = con.execute("SELECT COUNT(*) FROM v4_nonempty_rows").fetchone()[0]
print(f"Total non-empty V4 rows: {total_nonempty_v4}")

con.execute("""
CREATE TEMP TABLE v4_exploded AS
WITH split_pairs AS (
    SELECT 
        s1_id,
        unnest(string_split(v4_matched_ids, ',')) as candidate_id
    FROM v4_nonempty_rows
)
SELECT 
    s1_id,
    TRIM(candidate_id) as candidate_id
FROM split_pairs;
""")

print("Loading OUR_TEST...")
con.execute(f"""
CREATE TEMP TABLE our_test AS
SELECT 
    source1_entity_id as s1_id,
    candidate_entity_id as candidate_id,
    model_score as score
FROM read_csv('{pred_s2.replace(os.sep, "/")}', delim='\t', header=True, columns={{'source1_entity_id': 'VARCHAR', 'candidate_entity_id': 'VARCHAR', 'model_score': 'DOUBLE', 'predicted_match_label': 'TINYINT'}})
UNION ALL
SELECT 
    source1_entity_id as s1_id,
    candidate_entity_id as candidate_id,
    model_score as score
FROM read_csv('{pred_s3.replace(os.sep, "/")}', delim='\t', header=True, columns={{'source1_entity_id': 'VARCHAR', 'candidate_entity_id': 'VARCHAR', 'model_score': 'DOUBLE', 'predicted_match_label': 'TINYINT'}});
""")

# Measure (a): the max OUR_TEST score over the row's own pairs (only pairs found in OUR_TEST)
# Also report the number of non-empty rows where none of the row's pairs are in OUR_TEST.
print("Computing Measure (a)...")
con.execute("""
CREATE TEMP TABLE row_pairs_scored AS
SELECT 
    v.s1_id,
    v.candidate_id,
    t.score
FROM v4_exploded v
LEFT JOIN our_test t ON v.s1_id = t.s1_id AND v.candidate_id = t.candidate_id;

CREATE TEMP TABLE measure_a_per_row AS
SELECT 
    r.s1_id,
    MAX(p.score) as max_score_own_pairs,
    COUNT(p.score) as count_scored_pairs
FROM v4_nonempty_rows r
LEFT JOIN row_pairs_scored p ON r.s1_id = p.s1_id
GROUP BY r.s1_id;
""")

none_in_our_test = con.execute("SELECT COUNT(*) FROM measure_a_per_row WHERE count_scored_pairs = 0").fetchone()[0]
print(f"Non-empty rows where NONE of the row's pairs are in OUR_TEST: {none_in_our_test} ({none_in_our_test/total_nonempty_v4*100:.4f}%)")

# Measure (b): the max OUR_TEST score over ALL of that S1's candidates
print("Computing Measure (b)...")
con.execute("""
CREATE TEMP TABLE all_candidates_max_score AS
SELECT 
    s1_id,
    MAX(score) as max_score_all_cands
FROM our_test
GROUP BY s1_id;

CREATE TEMP TABLE measure_b_per_row AS
SELECT 
    r.s1_id,
    b.max_score_all_cands
FROM v4_nonempty_rows r
LEFT JOIN all_candidates_max_score b ON r.s1_id = b.s1_id;
""")

b_no_cands = con.execute("SELECT COUNT(*) FROM measure_b_per_row WHERE max_score_all_cands IS NULL").fetchone()[0]
print(f"Non-empty rows where S1 has NO candidates at all in OUR_TEST: {b_no_cands}")

# Now bucket both measures!
# Buckets: 0-0.05, 0.05-0.2, 0.2-0.5, 0.5-0.8, 0.8-0.9, 0.9-0.98, 0.98-0.995, 0.995-1.0
query_test_measures = """
WITH a_bucketed AS (
    SELECT 
        s1_id,
        CASE 
            WHEN max_score_own_pairs IS NULL THEN 'None in OUR_TEST'
            WHEN max_score_own_pairs >= 0.00 AND max_score_own_pairs < 0.05 THEN '0-0.05'
            WHEN max_score_own_pairs >= 0.05 AND max_score_own_pairs < 0.20 THEN '0.05-0.2'
            WHEN max_score_own_pairs >= 0.20 AND max_score_own_pairs < 0.50 THEN '0.2-0.5'
            WHEN max_score_own_pairs >= 0.50 AND max_score_own_pairs < 0.80 THEN '0.5-0.8'
            WHEN max_score_own_pairs >= 0.80 AND max_score_own_pairs < 0.90 THEN '0.8-0.9'
            WHEN max_score_own_pairs >= 0.90 AND max_score_own_pairs < 0.98 THEN '0.9-0.98'
            WHEN max_score_own_pairs >= 0.98 AND max_score_own_pairs < 0.995 THEN '0.98-0.995'
            WHEN max_score_own_pairs >= 0.995 AND max_score_own_pairs <= 1.000001 THEN '0.995-1.0'
            ELSE 'OTHER'
        END as bucket_a,
        CASE 
            WHEN max_score_own_pairs IS NULL THEN 99
            WHEN max_score_own_pairs >= 0.00 AND max_score_own_pairs < 0.05 THEN 1
            WHEN max_score_own_pairs >= 0.05 AND max_score_own_pairs < 0.20 THEN 2
            WHEN max_score_own_pairs >= 0.20 AND max_score_own_pairs < 0.50 THEN 3
            WHEN max_score_own_pairs >= 0.50 AND max_score_own_pairs < 0.80 THEN 4
            WHEN max_score_own_pairs >= 0.80 AND max_score_own_pairs < 0.90 THEN 5
            WHEN max_score_own_pairs >= 0.90 AND max_score_own_pairs < 0.98 THEN 6
            WHEN max_score_own_pairs >= 0.98 AND max_score_own_pairs < 0.995 THEN 7
            WHEN max_score_own_pairs >= 0.995 AND max_score_own_pairs <= 1.000001 THEN 8
            ELSE 9
        END as ord_a
    FROM measure_a_per_row
),
b_bucketed AS (
    SELECT 
        s1_id,
        CASE 
            WHEN max_score_all_cands IS NULL THEN 'No cands in OUR_TEST'
            WHEN max_score_all_cands >= 0.00 AND max_score_all_cands < 0.05 THEN '0-0.05'
            WHEN max_score_all_cands >= 0.05 AND max_score_all_cands < 0.20 THEN '0.05-0.2'
            WHEN max_score_all_cands >= 0.20 AND max_score_all_cands < 0.50 THEN '0.2-0.5'
            WHEN max_score_all_cands >= 0.50 AND max_score_all_cands < 0.80 THEN '0.5-0.8'
            WHEN max_score_all_cands >= 0.80 AND max_score_all_cands < 0.90 THEN '0.8-0.9'
            WHEN max_score_all_cands >= 0.90 AND max_score_all_cands < 0.98 THEN '0.9-0.98'
            WHEN max_score_all_cands >= 0.98 AND max_score_all_cands < 0.995 THEN '0.98-0.995'
            WHEN max_score_all_cands >= 0.995 AND max_score_all_cands <= 1.000001 THEN '0.995-1.0'
            ELSE 'OTHER'
        END as bucket_b,
        CASE 
            WHEN max_score_all_cands IS NULL THEN 99
            WHEN max_score_all_cands >= 0.00 AND max_score_all_cands < 0.05 THEN 1
            WHEN max_score_all_cands >= 0.05 AND max_score_all_cands < 0.20 THEN 2
            WHEN max_score_all_cands >= 0.20 AND max_score_all_cands < 0.50 THEN 3
            WHEN max_score_all_cands >= 0.50 AND max_score_all_cands < 0.80 THEN 4
            WHEN max_score_all_cands >= 0.80 AND max_score_all_cands < 0.90 THEN 5
            WHEN max_score_all_cands >= 0.90 AND max_score_all_cands < 0.98 THEN 6
            WHEN max_score_all_cands >= 0.98 AND max_score_all_cands < 0.995 THEN 7
            WHEN max_score_all_cands >= 0.995 AND max_score_all_cands <= 1.000001 THEN 8
            ELSE 9
        END as ord_b
    FROM measure_b_per_row
)
SELECT 
    coalesce(a.bucket_a, b.bucket_b) as bucket,
    coalesce(a.ord_a, b.ord_b) as ord,
    COUNT(a.s1_id) as count_a,
    ROUND(COUNT(a.s1_id)::DOUBLE / 1649822 * 100, 4) as pct_a
FROM a_bucketed a
GROUP BY bucket_a, ord_a
ORDER BY ord;
"""

df_a = con.execute("""
SELECT 
    CASE 
        WHEN max_score_own_pairs IS NULL THEN 'None in OUR_TEST'
        WHEN max_score_own_pairs >= 0.00 AND max_score_own_pairs < 0.05 THEN '0-0.05'
        WHEN max_score_own_pairs >= 0.05 AND max_score_own_pairs < 0.20 THEN '0.05-0.2'
        WHEN max_score_own_pairs >= 0.20 AND max_score_own_pairs < 0.50 THEN '0.2-0.5'
        WHEN max_score_own_pairs >= 0.50 AND max_score_own_pairs < 0.80 THEN '0.5-0.8'
        WHEN max_score_own_pairs >= 0.80 AND max_score_own_pairs < 0.90 THEN '0.8-0.9'
        WHEN max_score_own_pairs >= 0.90 AND max_score_own_pairs < 0.98 THEN '0.9-0.98'
        WHEN max_score_own_pairs >= 0.98 AND max_score_own_pairs < 0.995 THEN '0.98-0.995'
        WHEN max_score_own_pairs >= 0.995 AND max_score_own_pairs <= 1.000001 THEN '0.995-1.0'
        ELSE 'OTHER'
    END as bucket,
    CASE 
        WHEN max_score_own_pairs >= 0.00 AND max_score_own_pairs < 0.05 THEN 1
        WHEN max_score_own_pairs >= 0.05 AND max_score_own_pairs < 0.20 THEN 2
        WHEN max_score_own_pairs >= 0.20 AND max_score_own_pairs < 0.50 THEN 3
        WHEN max_score_own_pairs >= 0.50 AND max_score_own_pairs < 0.80 THEN 4
        WHEN max_score_own_pairs >= 0.80 AND max_score_own_pairs < 0.90 THEN 5
        WHEN max_score_own_pairs >= 0.90 AND max_score_own_pairs < 0.98 THEN 6
        WHEN max_score_own_pairs >= 0.98 AND max_score_own_pairs < 0.995 THEN 7
        WHEN max_score_own_pairs >= 0.995 AND max_score_own_pairs <= 1.000001 THEN 8
        WHEN max_score_own_pairs IS NULL THEN 9
        ELSE 10
    END as ord,
    COUNT(*) as rows_measure_a,
    ROUND(COUNT(*)::DOUBLE / 1649822 * 100, 4) as pct_measure_a
FROM measure_a_per_row
GROUP BY bucket, ord
ORDER BY ord;
""").df()
print("\nMeasure (a) Table:")
print(df_a.to_string(index=False))

df_b = con.execute("""
SELECT 
    CASE 
        WHEN max_score_all_cands IS NULL THEN 'No cands in OUR_TEST'
        WHEN max_score_all_cands >= 0.00 AND max_score_all_cands < 0.05 THEN '0-0.05'
        WHEN max_score_all_cands >= 0.05 AND max_score_all_cands < 0.20 THEN '0.05-0.2'
        WHEN max_score_all_cands >= 0.20 AND max_score_all_cands < 0.50 THEN '0.2-0.5'
        WHEN max_score_all_cands >= 0.50 AND max_score_all_cands < 0.80 THEN '0.5-0.8'
        WHEN max_score_all_cands >= 0.80 AND max_score_all_cands < 0.90 THEN '0.8-0.9'
        WHEN max_score_all_cands >= 0.90 AND max_score_all_cands < 0.98 THEN '0.9-0.98'
        WHEN max_score_all_cands >= 0.98 AND max_score_all_cands < 0.995 THEN '0.98-0.995'
        WHEN max_score_all_cands >= 0.995 AND max_score_all_cands <= 1.000001 THEN '0.995-1.0'
        ELSE 'OTHER'
    END as bucket,
    CASE 
        WHEN max_score_all_cands >= 0.00 AND max_score_all_cands < 0.05 THEN 1
        WHEN max_score_all_cands >= 0.05 AND max_score_all_cands < 0.20 THEN 2
        WHEN max_score_all_cands >= 0.20 AND max_score_all_cands < 0.50 THEN 3
        WHEN max_score_all_cands >= 0.50 AND max_score_all_cands < 0.80 THEN 4
        WHEN max_score_all_cands >= 0.80 AND max_score_all_cands < 0.90 THEN 5
        WHEN max_score_all_cands >= 0.90 AND max_score_all_cands < 0.98 THEN 6
        WHEN max_score_all_cands >= 0.98 AND max_score_all_cands < 0.995 THEN 7
        WHEN max_score_all_cands >= 0.995 AND max_score_all_cands <= 1.000001 THEN 8
        WHEN max_score_all_cands IS NULL THEN 9
        ELSE 10
    END as ord,
    COUNT(*) as rows_measure_b,
    ROUND(COUNT(*)::DOUBLE / 1649822 * 100, 4) as pct_measure_b
FROM measure_b_per_row
GROUP BY bucket, ord
ORDER BY ord;
""").df()
print("\nMeasure (b) Table:")
print(df_b.to_string(index=False))
