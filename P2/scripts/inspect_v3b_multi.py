import duckdb

con = duckdb.connect()
con.execute("PRAGMA threads=8;")
con.execute("PRAGMA memory_limit='20GB';")

# Let's inspect some of those 201 candidates
con.execute("""
CREATE TEMP TABLE best_raw AS
SELECT 
    row_number() over () as row_id,
    source1_entity_id as s1_id,
    matched_entity_ids
FROM read_csv('Amazon-ML-Challenge-2026/P2/data/matching_results_v1_conservative.tsv', delim='\t', header=True, all_varchar=True);

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

CREATE TEMP TABLE best_empty_s1s AS
SELECT s1_id, row_id
FROM best_raw
WHERE matched_entity_ids IS NULL OR TRIM(matched_entity_ids) = '';

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

CREATE TEMP TABLE empty_eligible AS
SELECT 
    e.s1_id,
    t.candidate_id,
    t.score,
    CASE WHEN t.candidate_id LIKE 'S2-%' THEN 'S2' ELSE 'S3' END as target_type
FROM best_empty_s1s e
JOIN our_test t ON e.s1_id = t.s1_id
WHERE t.score >= 0.98
  AND t.candidate_id NOT IN (SELECT candidate_id FROM best_exploded);

CREATE TEMP TABLE top_per_s1 AS
SELECT s1_id, candidate_id, score, target_type
FROM (
    SELECT 
        s1_id, candidate_id, score, target_type,
        ROW_NUMBER() OVER (PARTITION BY s1_id, target_type ORDER BY score DESC, candidate_id ASC) as rnk
    FROM empty_eligible
)
WHERE rnk = 1;

CREATE TEMP TABLE multi_selected AS
SELECT candidate_id, COUNT(*) as cnt
FROM top_per_s1
GROUP BY candidate_id
HAVING COUNT(*) > 1;

SELECT t.*, m.cnt
FROM top_per_s1 t
JOIN multi_selected m ON t.candidate_id = m.candidate_id
ORDER BY t.candidate_id, t.score DESC;
""")

rows = con.fetchall()
for r in rows[:15]:
    print(r)
