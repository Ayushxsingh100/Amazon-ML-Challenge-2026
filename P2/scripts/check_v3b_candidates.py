import duckdb

con = duckdb.connect()
con.execute("PRAGMA threads=8;")
con.execute("PRAGMA memory_limit='20GB';")

print("--- Testing V3b candidate availability for empty S1 rows ---")

# 1. Load BEST raw
con.execute("""
CREATE TEMP TABLE best_raw AS
SELECT 
    row_number() over () as row_id,
    source1_entity_id as s1_id,
    matched_entity_ids
FROM read_csv('Amazon-ML-Challenge-2026/P2/data/matching_results_v1_conservative.tsv', delim='\t', header=True, all_varchar=True);
""")

# 2. Get empty S1s in BEST
con.execute("""
CREATE TEMP TABLE empty_s1s AS
SELECT s1_id, row_id
FROM best_raw
WHERE matched_entity_ids IS NULL OR TRIM(matched_entity_ids) = '';
""")

print("Empty S1 count:", con.execute("SELECT COUNT(*) FROM empty_s1s").fetchone()[0])

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

# 4. Check candidates with score >= 0.98 for empty S1s
con.execute("""
CREATE TEMP TABLE empty_high_cands AS
SELECT 
    e.s1_id,
    t.candidate_id,
    t.score,
    CASE WHEN t.candidate_id LIKE 'S2-%' THEN 'S2' ELSE 'S3' END as src
FROM empty_s1s e
JOIN our_test t ON e.s1_id = t.s1_id
WHERE t.score >= 0.98;
""")

print("High score candidates for empty S1s count:", con.execute("SELECT COUNT(*) FROM empty_high_cands").fetchone()[0])
print("Empty S1s with at least one high score candidate:", con.execute("SELECT COUNT(DISTINCT s1_id) FROM empty_high_cands").fetchone()[0])

# Now check how many of these candidates are in BEST (or V3a)
con.execute("""
CREATE TEMP TABLE best_exploded AS
WITH split_pairs AS (
    SELECT 
        s1_id,
        unnest(string_split(matched_entity_ids, ',')) as candidate_id
    FROM best_raw
    WHERE matched_entity_ids IS NOT NULL AND TRIM(matched_entity_ids) != ''
)
SELECT DISTINCT TRIM(candidate_id) as candidate_id
FROM split_pairs;
""")

con.execute("""
CREATE TEMP TABLE eligible_cands AS
SELECT h.*
FROM empty_high_cands h
LEFT JOIN best_exploded b ON h.candidate_id = b.candidate_id
WHERE b.candidate_id IS NULL;
""")

print("Eligible candidates (not in BEST):", con.execute("SELECT COUNT(*) FROM eligible_cands").fetchone()[0])
print("Distinct candidates in eligible:", con.execute("SELECT COUNT(DISTINCT candidate_id) FROM eligible_cands").fetchone()[0])
print("Empty S1s with eligible candidates:", con.execute("SELECT COUNT(DISTINCT s1_id) FROM eligible_cands").fetchone()[0])

# Check if any candidate is claimed by multiple empty S1s
multi_claim_empty = con.execute("""
SELECT candidate_id, COUNT(DISTINCT s1_id) as s1_cnt
FROM eligible_cands
GROUP BY candidate_id
HAVING COUNT(DISTINCT s1_id) > 1;
""").df()
print("Candidates claimed by multiple empty S1s:", len(multi_claim_empty))
