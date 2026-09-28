import os
import duckdb

repo_root = "c:/NEW AMAZON"
p2_data = os.path.join(repo_root, "Amazon-ML-Challenge-2026", "P2", "data")
v4_path = os.path.join(p2_data, "matching_results_v4.tsv")
pred_s2 = os.path.join(repo_root, "Amazon-ML-Challenge-2026", "P2", "predictions", "phase4", "test_predictions_s2.tsv")
pred_s3 = os.path.join(repo_root, "Amazon-ML-Challenge-2026", "P2", "predictions", "phase4", "test_predictions_s3.tsv")
gt_path = os.path.join(repo_root, "Amazon-ML-Challenge-2026", "outputs", "person1_step1", "train_ground_truth_reconstructed.tsv")

con = duckdb.connect()
con.execute("PRAGMA threads=8;")
con.execute("PRAGMA memory_limit='24GB';")

print("--- Step 1: Train Labels Zero-Match Analysis ---")
con.execute(f"""
CREATE TEMP TABLE train_gt_raw AS
SELECT 
    source1_entity_id as s1_id,
    COALESCE(matched_entity_ids, '') as matched_entity_ids,
    CASE WHEN matched_entity_ids IS NULL OR TRIM(matched_entity_ids) = '' THEN 1 ELSE 0 END as is_zero_match
FROM read_csv('{gt_path.replace(os.sep, '/')}', delim='\t', header=True, all_varchar=True);
""")

step1_stats = con.execute("""
SELECT 
    COUNT(*) as total_train_s1,
    COUNT(CASE WHEN is_zero_match = 1 THEN 1 END) as zero_matches_overall,
    ROUND(COUNT(CASE WHEN is_zero_match = 1 THEN 1 END)::DOUBLE / COUNT(*) * 100, 4) as pct_zero_overall,
    COUNT(CASE WHEN is_zero_match = 1 OR (matched_entity_ids NOT LIKE '%S2-%') THEN 1 END) as zero_s2_matches,
    ROUND(COUNT(CASE WHEN is_zero_match = 1 OR (matched_entity_ids NOT LIKE '%S2-%') THEN 1 END)::DOUBLE / COUNT(*) * 100, 4) as pct_zero_s2,
    COUNT(CASE WHEN is_zero_match = 1 OR (matched_entity_ids NOT LIKE '%S3-%') THEN 1 END) as zero_s3_matches,
    ROUND(COUNT(CASE WHEN is_zero_match = 1 OR (matched_entity_ids NOT LIKE '%S3-%') THEN 1 END)::DOUBLE / COUNT(*) * 100, 4) as pct_zero_s3
FROM train_gt_raw;
""").df()
print("Step 1 Results:")
print(step1_stats.to_string(index=False))

print("\n--- Step 3: V4 Empty Rows Best Score Distribution ---")
con.execute(f"""
CREATE TEMP TABLE v4_raw AS
SELECT 
    source1_entity_id as s1_id,
    COALESCE(matched_entity_ids, '') as v4_matched_ids,
    CASE WHEN matched_entity_ids IS NULL OR TRIM(matched_entity_ids) = '' THEN 1 ELSE 0 END as is_empty_in_v4
FROM read_csv('{v4_path.replace(os.sep, '/')}', delim='\t', header=True, all_varchar=True);
""")

v4_empty_count = con.execute("SELECT COUNT(*) FROM v4_raw WHERE is_empty_in_v4 = 1").fetchone()[0]
print(f"Total V4 empty rows: {v4_empty_count}")

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

con.execute("""
CREATE TEMP TABLE empty_s1_best_scores AS
SELECT 
    r.s1_id,
    MAX(t.score) as best_score
FROM v4_raw r
JOIN our_test t ON r.s1_id = t.s1_id
WHERE r.is_empty_in_v4 = 1
GROUP BY r.s1_id;
""")

any_cand_count = con.execute("SELECT COUNT(*) FROM empty_s1_best_scores").fetchone()[0]
no_cand_count = v4_empty_count - any_cand_count
print(f"V4 empty rows with ANY candidate in OUR_TEST: {any_cand_count} ({any_cand_count/v4_empty_count*100:.4f}%)")
print(f"V4 empty rows with NO candidates in OUR_TEST: {no_cand_count} ({no_cand_count/v4_empty_count*100:.4f}%)")

con.execute("""
SELECT 
    bucket,
    bucket_order,
    COUNT(*) as row_count,
    ROUND(COUNT(*)::DOUBLE / (SELECT COUNT(*) FROM empty_s1_best_scores) * 100, 4) as pct_of_with_cand,
    ROUND(COUNT(*)::DOUBLE / 82722 * 100, 4) as pct_of_all_empty
FROM (
    SELECT 
        s1_id,
        best_score,
        CASE 
            WHEN best_score >= 0.00 AND best_score < 0.05 THEN '0-0.05'
            WHEN best_score >= 0.05 AND best_score < 0.10 THEN '0.05-0.1'
            WHEN best_score >= 0.10 AND best_score < 0.20 THEN '0.1-0.2'
            WHEN best_score >= 0.20 AND best_score < 0.30 THEN '0.2-0.3'
            WHEN best_score >= 0.30 AND best_score < 0.50 THEN '0.3-0.5'
            WHEN best_score >= 0.50 AND best_score < 0.70 THEN '0.5-0.7'
            WHEN best_score >= 0.70 AND best_score < 0.90 THEN '0.7-0.9'
            WHEN best_score >= 0.90 AND best_score < 0.98 THEN '0.9-0.98'
            WHEN best_score >= 0.98 THEN '>=0.98'
            ELSE 'OTHER'
        END as bucket,
        CASE 
            WHEN best_score >= 0.00 AND best_score < 0.05 THEN 1
            WHEN best_score >= 0.05 AND best_score < 0.10 THEN 2
            WHEN best_score >= 0.10 AND best_score < 0.20 THEN 3
            WHEN best_score >= 0.20 AND best_score < 0.30 THEN 4
            WHEN best_score >= 0.30 AND best_score < 0.50 THEN 5
            WHEN best_score >= 0.50 AND best_score < 0.70 THEN 6
            WHEN best_score >= 0.70 AND best_score < 0.90 THEN 7
            WHEN best_score >= 0.90 AND best_score < 0.98 THEN 8
            WHEN best_score >= 0.98 THEN 9
            ELSE 10
        END as bucket_order
    FROM empty_s1_best_scores
)
GROUP BY bucket, bucket_order
ORDER BY bucket_order;
""")
df_dist = con.df()
print("\nDistribution of best score for empty S1s:")
print(df_dist.to_string(index=False))
