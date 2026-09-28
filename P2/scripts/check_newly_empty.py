import duckdb

con = duckdb.connect()
con.execute("PRAGMA threads=8;")
con.execute("PRAGMA memory_limit='20GB';")

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

CREATE TEMP TABLE candidate_claim_counts AS
SELECT 
    candidate_id,
    COUNT(DISTINCT s1_id) as s1_count
FROM best_exploded
GROUP BY candidate_id;

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

CREATE TEMP TABLE best_joined AS
SELECT 
    b.s1_id,
    b.candidate_id,
    b.position,
    t.score,
    c.s1_count
FROM best_exploded b
JOIN candidate_claim_counts c ON b.candidate_id = c.candidate_id
LEFT JOIN our_test t ON b.s1_id = t.s1_id AND b.candidate_id = t.candidate_id;

CREATE TEMP TABLE v3a_pairs AS
WITH scored_multi AS (
    SELECT 
        candidate_id,
        MAX(score) as max_score,
        COUNT(score) as scores_present
    FROM best_joined
    WHERE s1_count > 1
    GROUP BY candidate_id
),
ranked_multi AS (
    SELECT 
        j.s1_id,
        j.candidate_id,
        j.position,
        j.score,
        j.s1_count,
        sm.scores_present,
        sm.max_score,
        ROW_NUMBER() OVER (
            PARTITION BY j.candidate_id 
            ORDER BY j.score DESC NULLS LAST, j.position ASC, j.s1_id ASC
        ) as rnk
    FROM best_joined j
    JOIN scored_multi sm ON j.candidate_id = sm.candidate_id
)
SELECT s1_id, candidate_id, position, score
FROM best_joined
WHERE s1_count = 1
UNION ALL
SELECT s1_id, candidate_id, position, score
FROM ranked_multi
WHERE scores_present = 0
UNION ALL
SELECT s1_id, candidate_id, position, score
FROM ranked_multi
WHERE scores_present > 0 AND rnk = 1;

CREATE TEMP TABLE newly_empty_s1s AS
SELECT DISTINCT s1_id 
FROM best_exploded
EXCEPT
SELECT DISTINCT s1_id
FROM v3a_pairs;

-- Check if newly_empty_s1s have any candidates with score >= 0.98 not in V3a
SELECT COUNT(*) 
FROM newly_empty_s1s n
JOIN our_test t ON n.s1_id = t.s1_id
WHERE t.score >= 0.98 
  AND t.candidate_id NOT IN (SELECT candidate_id FROM v3a_pairs);
""")

print("Newly empty S1 candidates >= 0.98 not in V3a:", con.fetchone()[0])
