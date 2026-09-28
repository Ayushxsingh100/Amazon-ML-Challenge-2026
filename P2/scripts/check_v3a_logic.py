import duckdb
import time

con = duckdb.connect()
con.execute("PRAGMA threads=8;")
con.execute("PRAGMA memory_limit='20GB';")

print("--- Deep Dive into V3a Logic ---")
t0 = time.time()

# 1. Load BEST raw and exploded
con.execute("""
CREATE TEMP TABLE best_raw AS
SELECT 
    row_number() over () as row_id,
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

# 2. Check candidate_ids claimed by > 1 S1 in BEST
con.execute("""
CREATE TEMP TABLE candidate_claim_counts AS
SELECT 
    candidate_id,
    COUNT(DISTINCT s1_id) as s1_count,
    COUNT(*) as pair_count
FROM best_exploded
GROUP BY candidate_id;
""")

stats = con.execute("""
SELECT 
    COUNT(*) as total_distinct_candidates,
    COUNT(CASE WHEN s1_count > 1 THEN 1 END) as candidates_claimed_by_gt_1_s1,
    SUM(CASE WHEN s1_count > 1 THEN pair_count ELSE 0 END) as pairs_with_candidates_gt_1_s1,
    MAX(s1_count) as max_claims_per_candidate
FROM candidate_claim_counts;
""").df()
print("Candidate Claim Stats in BEST:")
print(stats)

# Check if there are any duplicate candidate_ids within the SAME S1 in BEST
dups_within_s1 = con.execute("""
SELECT s1_id, candidate_id, COUNT(*) as cnt
FROM best_exploded
GROUP BY s1_id, candidate_id
HAVING COUNT(*) > 1;
""").df()
print(f"Duplicates within same S1 in BEST: {len(dups_within_s1)}")

# 3. Load OUR_TEST
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

# 4. Join multi-claimed candidates with our scores
con.execute("""
CREATE TEMP TABLE multi_claimed AS
SELECT 
    b.candidate_id,
    b.s1_id,
    b.position,
    t.score
FROM best_exploded b
JOIN candidate_claim_counts c ON b.candidate_id = c.candidate_id
LEFT JOIN our_test t ON b.s1_id = t.s1_id AND b.candidate_id = t.candidate_id
WHERE c.s1_count > 1;
""")

multi_stats = con.execute("""
SELECT 
    COUNT(*) as total_multi_pairs,
    COUNT(score) as pairs_with_score,
    COUNT(*) - COUNT(score) as pairs_without_score
FROM multi_claimed;
""").df()
print("Multi-claimed candidate pairs stats:")
print(multi_stats)

# Let's inspect cases for multi-claimed candidates:
# For each candidate_id:
# - How many have AT LEAST ONE score?
# - How many have NO scores across all claiming S1s?
cand_cases = con.execute("""
SELECT 
    candidate_id,
    COUNT(*) as s1_count,
    COUNT(score) as scores_present,
    MAX(score) as max_score
FROM multi_claimed
GROUP BY candidate_id;
""")

cand_cases_summary = con.execute("""
WITH c AS (
    SELECT 
        candidate_id,
        COUNT(*) as s1_count,
        COUNT(score) as scores_present,
        MAX(score) as max_score
    FROM multi_claimed
    GROUP BY candidate_id
)
SELECT 
    COUNT(*) as total_multi_cands,
    COUNT(CASE WHEN scores_present = 0 THEN 1 END) as zero_scores_present,
    COUNT(CASE WHEN scores_present = 1 THEN 1 END) as exactly_one_score_present,
    COUNT(CASE WHEN scores_present > 1 THEN 1 END) as multiple_scores_present
FROM c;
""").df()
print("Multi-candidate score presence breakdown:")
print(cand_cases_summary)

print(f"Elapsed time: {time.time() - t0:.2f}s")
