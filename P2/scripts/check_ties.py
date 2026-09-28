import duckdb

con = duckdb.connect()
con.execute("PRAGMA threads=8;")
con.execute("PRAGMA memory_limit='20GB';")

con.execute("""
CREATE TEMP TABLE best_raw AS
SELECT 
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

# Check for score ties among highest scores
ties = con.execute("""
WITH ranked AS (
    SELECT 
        candidate_id,
        s1_id,
        score,
        DENSE_RANK() OVER (PARTITION BY candidate_id ORDER BY score DESC NULLS LAST) as rnk
    FROM multi_claimed
    WHERE score IS NOT NULL
)
SELECT candidate_id, COUNT(*) as tied_s1s, MAX(score) as tied_score
FROM ranked
WHERE rnk = 1
GROUP BY candidate_id
HAVING COUNT(*) > 1;
""").df()

print("Tied top scores count:", len(ties))
if len(ties) > 0:
    print(ties.head(10))

# Also let's check how many total candidates are multi-claimed, how many have scores_present == 0
zero_scores = con.execute("""
WITH c AS (
    SELECT candidate_id, COUNT(score) as scores_present
    FROM multi_claimed
    GROUP BY candidate_id
)
SELECT 
    COUNT(*) as total_multi_cands,
    COUNT(CASE WHEN scores_present = 0 THEN 1 END) as zero_scores,
    COUNT(CASE WHEN scores_present > 0 THEN 1 END) as has_scores
FROM c;
""").df()
print(zero_scores)
