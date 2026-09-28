import duckdb
import time

con = duckdb.connect()
con.execute("PRAGMA threads=8;")
con.execute("PRAGMA memory_limit='20GB';")

print("--- Step 1: Exploding BEST and Left-Joining OUR_TEST ---")
t0 = time.time()

# 1. Load BEST table
print("Creating best_exploded table...")
con.execute("""
CREATE TEMP TABLE best_raw AS
SELECT 
    source1_entity_id as s1_id,
    matched_entity_ids
FROM read_csv('Amazon-ML-Challenge-2026/P2/data/matching_results_v1_conservative.tsv', delim='\t', header=True, all_varchar=True);
""")

con.execute("""
CREATE TEMP TABLE best_exploded AS
WITH split_pairs AS (
    SELECT 
        s1_id,
        unnest(string_split(matched_entity_ids, ',')) as candidate_id,
        generate_subscripts(string_split(matched_entity_ids, ','), 1) as position
    FROM best_raw
    WHERE matched_entity_ids IS NOT NULL AND TRIM(matched_entity_ids) != ''
)
SELECT 
    s1_id,
    TRIM(candidate_id) as candidate_id,
    position
FROM split_pairs;
""")

exploded_count = con.execute("SELECT COUNT(*) FROM best_exploded").fetchone()[0]
empty_rows_count = con.execute("""
    SELECT COUNT(*) 
    FROM best_raw 
    WHERE matched_entity_ids IS NULL OR TRIM(matched_entity_ids) = ''
""").fetchone()[0]
total_best_rows = con.execute("SELECT COUNT(*) FROM best_raw").fetchone()[0]
distinct_s1_best = con.execute("SELECT COUNT(DISTINCT s1_id) FROM best_raw").fetchone()[0]

print(f"Total BEST rows: {total_best_rows}")
print(f"Distinct S1 in BEST: {distinct_s1_best}")
print(f"Empty BEST rows: {empty_rows_count}")
print(f"Total exploded BEST pairs: {exploded_count}")

# 2. Prepare OUR_TEST table
print("Creating our_test table...")
con.execute("""
CREATE TEMP TABLE our_test AS
SELECT 
    source1_entity_id as s1_id,
    candidate_entity_id as candidate_id,
    model_score as score
FROM read_csv('Amazon-ML-Challenge-2026/P2/predictions/phase4/test_predictions_s2.tsv', delim='\t', header=True, columns={'source1_entity_id': 'VARCHAR', 'candidate_entity_id': 'VARCHAR', 'model_score': 'DOUBLE', 'predicted_match_label': 'TINYINT'})
UNION ALL
SELECT 
    source1_entity_id as s1_id,
    candidate_entity_id as candidate_id,
    model_score as score
FROM read_csv('Amazon-ML-Challenge-2026/P2/predictions/phase4/test_predictions_s3.tsv', delim='\t', header=True, columns={'source1_entity_id': 'VARCHAR', 'candidate_entity_id': 'VARCHAR', 'model_score': 'DOUBLE', 'predicted_match_label': 'TINYINT'});
""")

# 3. S1 Overlap between BEST and OUR_TEST
print("Computing S1 overlap...")
s1_overlap = con.execute("""
WITH best_s1 AS (SELECT DISTINCT s1_id FROM best_raw),
     test_s1 AS (SELECT DISTINCT s1_id FROM our_test)
SELECT 
    (SELECT COUNT(*) FROM best_s1) as best_total_s1,
    (SELECT COUNT(*) FROM test_s1) as our_test_total_s1,
    COUNT(b.s1_id) as s1_in_both,
    (SELECT COUNT(*) FROM best_s1) - COUNT(b.s1_id) as s1_in_best_only,
    (SELECT COUNT(*) FROM test_s1) - COUNT(b.s1_id) as s1_in_our_test_only
FROM best_s1 b
JOIN test_s1 t ON b.s1_id = t.s1_id;
""").df()
print("S1 Overlap:")
print(s1_overlap)

# 4. Left join BEST with OUR_TEST to check pairs found
print("Left joining best_exploded with our_test...")
con.execute("""
CREATE TEMP TABLE best_joined AS
SELECT 
    b.s1_id,
    b.candidate_id,
    b.position,
    t.score
FROM best_exploded b
LEFT JOIN our_test t 
  ON b.s1_id = t.s1_id AND b.candidate_id = t.candidate_id;
""")

pairs_stats = con.execute("""
SELECT 
    COUNT(*) as total_best_pairs,
    COUNT(score) as found_in_our_candidates,
    COUNT(*) - COUNT(score) as missing_in_our_candidates,
    ROUND(COUNT(score)::DOUBLE / COUNT(*) * 100, 4) as pct_found
FROM best_joined;
""").df()
print("Pairs Match Stats:")
print(pairs_stats)

# 5. Score distribution of found pairs
print("Computing score distribution of found pairs...")
score_dist = con.execute("""
SELECT 
    COUNT(score) as count,
    MIN(score) as min_score,
    ROUND(quantile_cont(score, 0.05), 6) as p05,
    ROUND(quantile_cont(score, 0.10), 6) as p10,
    ROUND(quantile_cont(score, 0.25), 6) as p25,
    ROUND(quantile_cont(score, 0.50), 6) as median,
    ROUND(AVG(score), 6) as mean,
    ROUND(quantile_cont(score, 0.75), 6) as p75,
    ROUND(quantile_cont(score, 0.90), 6) as p90,
    ROUND(quantile_cont(score, 0.95), 6) as p95,
    MAX(score) as max_score,
    ROUND(STDDEV(score), 6) as std_dev
FROM best_joined
WHERE score IS NOT NULL;
""").df()
print("Score Distribution Summary:")
print(score_dist)

# Score distribution across the same buckets as Step 2:
score_buckets_found = con.execute("""
SELECT 
    bucket,
    bucket_order,
    COUNT(*) as pair_count,
    ROUND(COUNT(*)::DOUBLE / (SELECT COUNT(score) FROM best_joined WHERE score IS NOT NULL) * 100, 4) as pct_of_found
FROM (
    SELECT 
        score,
        CASE 
            WHEN score >= 0.00 AND score < 0.02 THEN '0-0.02'
            WHEN score >= 0.02 AND score < 0.05 THEN '0.02-0.05'
            WHEN score >= 0.05 AND score < 0.10 THEN '0.05-0.1'
            WHEN score >= 0.10 AND score < 0.20 THEN '0.1-0.2'
            WHEN score >= 0.20 AND score < 0.50 THEN '0.2-0.5'
            WHEN score >= 0.50 AND score < 0.80 THEN '0.5-0.8'
            WHEN score >= 0.80 AND score < 0.90 THEN '0.8-0.9'
            WHEN score >= 0.90 AND score < 0.95 THEN '0.9-0.95'
            WHEN score >= 0.95 AND score < 0.98 THEN '0.95-0.98'
            WHEN score >= 0.98 AND score <= 1.000001 THEN '0.98-1.0'
            ELSE 'OTHER'
        END as bucket,
        CASE 
            WHEN score >= 0.00 AND score < 0.02 THEN 1
            WHEN score >= 0.02 AND score < 0.05 THEN 2
            WHEN score >= 0.05 AND score < 0.10 THEN 3
            WHEN score >= 0.10 AND score < 0.20 THEN 4
            WHEN score >= 0.20 AND score < 0.50 THEN 5
            WHEN score >= 0.50 AND score < 0.80 THEN 6
            WHEN score >= 0.80 AND score < 0.90 THEN 7
            WHEN score >= 0.90 AND score < 0.95 THEN 8
            WHEN score >= 0.95 AND score < 0.98 THEN 9
            WHEN score >= 0.98 AND score <= 1.000001 THEN 10
            ELSE 11
        END as bucket_order
    FROM best_joined
    WHERE score IS NOT NULL
)
GROUP BY bucket, bucket_order
ORDER BY bucket_order;
""").df()
print("Score Bucket Distribution for Found Pairs:")
print(score_buckets_found)

print(f"Total time: {time.time() - t0:.2f}s")
