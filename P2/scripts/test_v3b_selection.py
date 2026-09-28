import duckdb

con = duckdb.connect()
con.execute("PRAGMA threads=8;")
con.execute("PRAGMA memory_limit='20GB';")

print("--- Checking V3b empty S1 selection ---")

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
""")

# Let's build V3a first
con.execute("""
-- Join all best_exploded pairs with scores
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

-- For candidates with s1_count > 1:
-- If none has our score, keep all (keep BEST's choice)
-- If at least one has our score, keep only the S1 where score is highest!
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
-- Keep single-claim pairs
SELECT s1_id, candidate_id, position, score
FROM best_joined
WHERE s1_count = 1

UNION ALL

-- For multi-claim pairs:
-- If scores_present = 0, keep all
SELECT s1_id, candidate_id, position, score
FROM ranked_multi
WHERE scores_present = 0

UNION ALL

-- If scores_present > 0, keep only rnk = 1
SELECT s1_id, candidate_id, position, score
FROM ranked_multi
WHERE scores_present > 0 AND rnk = 1;
""")

print("Total pairs in BEST:", con.execute("SELECT COUNT(*) FROM best_exploded").fetchone()[0])
print("Total pairs in V3a:", con.execute("SELECT COUNT(*) FROM v3a_pairs").fetchone()[0])
print("Pairs removed in V3a:", con.execute("SELECT COUNT(*) FROM best_exploded").fetchone()[0] - con.execute("SELECT COUNT(*) FROM v3a_pairs").fetchone()[0])

# Distinct candidate_ids used anywhere in V3a
con.execute("""
CREATE TEMP TABLE v3a_used_candidates AS
SELECT DISTINCT candidate_id FROM v3a_pairs;
""")
print("Distinct candidates used in V3a:", con.execute("SELECT COUNT(*) FROM v3a_used_candidates").fetchone()[0])

# Empty S1s in BEST
con.execute("""
CREATE TEMP TABLE best_empty_s1s AS
SELECT s1_id, row_id
FROM best_raw
WHERE matched_entity_ids IS NULL OR TRIM(matched_entity_ids) = '';
""")

# Candidates for empty S1s: score >= 0.98, not in v3a_used_candidates
con.execute("""
CREATE TEMP TABLE empty_eligible_candidates AS
SELECT 
    e.s1_id,
    e.row_id,
    t.candidate_id,
    t.score,
    CASE WHEN t.candidate_id LIKE 'S2-%' THEN 'S2' ELSE 'S3' END as target_type
FROM best_empty_s1s e
JOIN our_test t ON e.s1_id = t.s1_id
WHERE t.score >= 0.98
  AND t.candidate_id NOT IN (SELECT candidate_id FROM v3a_used_candidates);
""")

print("Eligible candidates for empty S1s count:", con.execute("SELECT COUNT(*) FROM empty_eligible_candidates").fetchone()[0])

# For each empty S1, pick at most 1 S2 and 1 S3 with highest score
con.execute("""
CREATE TEMP TABLE best_candidate_per_type AS
SELECT 
    s1_id,
    row_id,
    candidate_id,
    score,
    target_type,
    ROW_NUMBER() OVER (
        PARTITION BY s1_id, target_type 
        ORDER BY score DESC, candidate_id ASC
    ) as rnk
FROM empty_eligible_candidates;
""")

con.execute("""
CREATE TEMP TABLE v3b_additions_unfiltered AS
SELECT s1_id, row_id, candidate_id, score, target_type
FROM best_candidate_per_type
WHERE rnk = 1;
""")

print("Unfiltered additions count (at most 1 S2, 1 S3 per empty S1):", con.execute("SELECT COUNT(*) FROM v3b_additions_unfiltered").fetchone()[0])
print("Distinct candidates in additions:", con.execute("SELECT COUNT(DISTINCT candidate_id) FROM v3b_additions_unfiltered").fetchone()[0])

# Check if any candidate is selected by > 1 empty S1
multi_add = con.execute("""
SELECT candidate_id, COUNT(*) as cnt
FROM v3b_additions_unfiltered
GROUP BY candidate_id
HAVING COUNT(*) > 1;
""").df()
print("Candidates selected by multiple empty S1s:", len(multi_add))
if len(multi_add) > 0:
    print(multi_add.head())
