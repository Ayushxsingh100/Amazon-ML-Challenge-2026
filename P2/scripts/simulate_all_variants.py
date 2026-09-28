import duckdb
import time

con = duckdb.connect()
con.execute("PRAGMA threads=8;")
con.execute("PRAGMA memory_limit='20GB';")

print("--- Comprehensive Simulation of V3a, V3b, V3c ---")
t0 = time.time()

# 1. Load BEST raw
con.execute("""
CREATE TEMP TABLE best_raw AS
SELECT 
    row_number() over () as row_id,
    source1_entity_id as s1_id,
    COALESCE(matched_entity_ids, '') as orig_matched_ids,
    CASE WHEN matched_entity_ids IS NULL OR TRIM(matched_entity_ids) = '' THEN 1 ELSE 0 END as is_empty_in_best
FROM read_csv('Amazon-ML-Challenge-2026/P2/data/matching_results_v1_conservative.tsv', delim='\t', header=True, all_varchar=True);
""")

con.execute("""
CREATE TEMP TABLE best_exploded AS
WITH split_pairs AS (
    SELECT 
        s1_id,
        unnest(string_split(orig_matched_ids, ',')) as candidate_id,
        generate_subscripts(string_split(orig_matched_ids, ','), 1) as position
    FROM best_raw
    WHERE is_empty_in_best = 0
)
SELECT 
    s1_id,
    TRIM(candidate_id) as candidate_id,
    position
FROM split_pairs;
""")

total_best_pairs = con.execute("SELECT COUNT(*) FROM best_exploded").fetchone()[0]
total_best_rows = con.execute("SELECT COUNT(*) FROM best_raw").fetchone()[0]
empty_best_rows = con.execute("SELECT COUNT(*) FROM best_raw WHERE is_empty_in_best = 1").fetchone()[0]

print(f"Total BEST rows: {total_best_rows}")
print(f"Empty BEST rows: {empty_best_rows}")
print(f"Total BEST pairs: {total_best_pairs}")

# 2. Load OUR_TEST
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

# 3. BUILD V3a
print("\n--- Building V3a ---")
con.execute("""
CREATE TEMP TABLE candidate_claim_counts AS
SELECT 
    candidate_id,
    COUNT(DISTINCT s1_id) as s1_count
FROM best_exploded
GROUP BY candidate_id;

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
""")

v3a_pairs_count = con.execute("SELECT COUNT(*) FROM v3a_pairs").fetchone()[0]
v3a_pairs_removed = total_best_pairs - v3a_pairs_count
v3a_pairs_added = 0

# Check empty rows in V3a
con.execute("""
CREATE TEMP TABLE v3a_aggregated AS
SELECT 
    s1_id,
    string_agg(candidate_id, ',' ORDER BY position ASC) as matched_entity_ids
FROM v3a_pairs
GROUP BY s1_id;

CREATE TEMP TABLE v3a_full_rows AS
SELECT 
    r.row_id,
    r.s1_id,
    r.orig_matched_ids,
    COALESCE(a.matched_entity_ids, '') as v3a_matched_ids,
    CASE WHEN a.matched_entity_ids IS NULL OR a.matched_entity_ids = '' THEN 1 ELSE 0 END as is_empty_v3a,
    CASE WHEN COALESCE(a.matched_entity_ids, '') != r.orig_matched_ids THEN 1 ELSE 0 END as is_changed_from_best
FROM best_raw r
LEFT JOIN v3a_aggregated a ON r.s1_id = a.s1_id;
""")

v3a_empty_rows = con.execute("SELECT COUNT(*) FROM v3a_full_rows WHERE is_empty_v3a = 1").fetchone()[0]
v3a_rows_changed = con.execute("SELECT COUNT(*) FROM v3a_full_rows WHERE is_changed_from_best = 1").fetchone()[0]

print(f"V3a: pairs_removed={v3a_pairs_removed}, pairs_added={v3a_pairs_added}, rows_changed={v3a_rows_changed}, empty_rows_left={v3a_empty_rows}")

# 4. BUILD V3b
print("\n--- Building V3b ---")
# Candidate IDs used anywhere in V3a:
con.execute("""
CREATE TEMP TABLE v3a_used_cands AS
SELECT DISTINCT candidate_id FROM v3a_pairs;
""")

# For S1 rows EMPTY in BEST:
# add our candidates with score >= 0.98, at most 1 S2 and 1 S3 per S1, only candidates not already used in V3a
con.execute("""
CREATE TEMP TABLE v3b_candidates_pool AS
SELECT 
    r.s1_id,
    t.candidate_id,
    t.score,
    CASE WHEN t.candidate_id LIKE 'S2-%' THEN 'S2' ELSE 'S3' END as target_type,
    ROW_NUMBER() OVER (
        PARTITION BY r.s1_id, CASE WHEN t.candidate_id LIKE 'S2-%' THEN 'S2' ELSE 'S3' END
        ORDER BY t.score DESC, t.candidate_id ASC
    ) as rnk
FROM best_raw r
JOIN our_test t ON r.s1_id = t.s1_id
WHERE r.is_empty_in_best = 1
  AND t.score >= 0.98
  AND t.candidate_id NOT IN (SELECT candidate_id FROM v3a_used_cands);

CREATE TEMP TABLE v3b_additions AS
SELECT 
    s1_id,
    candidate_id,
    score,
    target_type,
    CASE WHEN target_type = 'S2' THEN 1 ELSE 2 END as pos
FROM v3b_candidates_pool
WHERE rnk = 1;
""")

v3b_additions_count = con.execute("SELECT COUNT(*) FROM v3b_additions").fetchone()[0]
print(f"V3b added pairs: {v3b_additions_count}")

# Combine V3a pairs + V3b additions
con.execute("""
CREATE TEMP TABLE v3b_pairs AS
SELECT s1_id, candidate_id, position, score
FROM v3a_pairs
UNION ALL
SELECT s1_id, candidate_id, pos as position, score
FROM v3b_additions;
""")

v3b_pairs_count = con.execute("SELECT COUNT(*) FROM v3b_pairs").fetchone()[0]

con.execute("""
CREATE TEMP TABLE v3b_aggregated AS
SELECT 
    s1_id,
    string_agg(candidate_id, ',' ORDER BY position ASC) as matched_entity_ids
FROM v3b_pairs
GROUP BY s1_id;

CREATE TEMP TABLE v3b_full_rows AS
SELECT 
    r.row_id,
    r.s1_id,
    r.orig_matched_ids,
    COALESCE(a.matched_entity_ids, '') as v3b_matched_ids,
    CASE WHEN a.matched_entity_ids IS NULL OR a.matched_entity_ids = '' THEN 1 ELSE 0 END as is_empty_v3b,
    CASE WHEN COALESCE(a.matched_entity_ids, '') != r.orig_matched_ids THEN 1 ELSE 0 END as is_changed_from_best,
    CASE WHEN COALESCE(a.matched_entity_ids, '') != vf.v3a_matched_ids THEN 1 ELSE 0 END as is_changed_from_v3a
FROM best_raw r
LEFT JOIN v3b_aggregated a ON r.s1_id = a.s1_id
JOIN v3a_full_rows vf ON r.row_id = vf.row_id;
""")

v3b_empty_rows = con.execute("SELECT COUNT(*) FROM v3b_full_rows WHERE is_empty_v3b = 1").fetchone()[0]
v3b_rows_changed_from_best = con.execute("SELECT COUNT(*) FROM v3b_full_rows WHERE is_changed_from_best = 1").fetchone()[0]
v3b_rows_changed_from_v3a = con.execute("SELECT COUNT(*) FROM v3b_full_rows WHERE is_changed_from_v3a = 1").fetchone()[0]

print(f"V3b: pairs_count={v3b_pairs_count}, empty_rows={v3b_empty_rows}, changed_from_best={v3b_rows_changed_from_best}, changed_from_v3a={v3b_rows_changed_from_v3a}")

# 5. BUILD V3c
print("\n--- Building V3c ---")
# V3c: V3b + drop pairs that exist in our candidates with score < 0.5. Keep pairs that are not in our candidates.
# In v3b_pairs, score was already joined from our_test for both v3a_pairs and v3b_additions!
con.execute("""
CREATE TEMP TABLE v3c_pairs AS
SELECT 
    s1_id,
    candidate_id,
    position,
    score
FROM v3b_pairs
WHERE score IS NULL OR score >= 0.5;
""")

v3c_pairs_count = con.execute("SELECT COUNT(*) FROM v3c_pairs").fetchone()[0]
v3c_dropped_pairs = v3b_pairs_count - v3c_pairs_count
print(f"Pairs dropped in V3c vs V3b: {v3c_dropped_pairs}")

con.execute("""
CREATE TEMP TABLE v3c_aggregated AS
SELECT 
    s1_id,
    string_agg(candidate_id, ',' ORDER BY position ASC) as matched_entity_ids
FROM v3c_pairs
GROUP BY s1_id;

CREATE TEMP TABLE v3c_full_rows AS
SELECT 
    r.row_id,
    r.s1_id,
    r.orig_matched_ids,
    COALESCE(a.matched_entity_ids, '') as v3c_matched_ids,
    CASE WHEN a.matched_entity_ids IS NULL OR a.matched_entity_ids = '' THEN 1 ELSE 0 END as is_empty_v3c,
    CASE WHEN COALESCE(a.matched_entity_ids, '') != r.orig_matched_ids THEN 1 ELSE 0 END as is_changed_from_best,
    CASE WHEN COALESCE(a.matched_entity_ids, '') != vb.v3b_matched_ids THEN 1 ELSE 0 END as is_changed_from_v3b
FROM best_raw r
LEFT JOIN v3c_aggregated a ON r.s1_id = a.s1_id
JOIN v3b_full_rows vb ON r.row_id = vb.row_id;
""")

v3c_empty_rows = con.execute("SELECT COUNT(*) FROM v3c_full_rows WHERE is_empty_v3c = 1").fetchone()[0]
v3c_rows_changed_from_best = con.execute("SELECT COUNT(*) FROM v3c_full_rows WHERE is_changed_from_best = 1").fetchone()[0]
v3c_rows_changed_from_v3b = con.execute("SELECT COUNT(*) FROM v3c_full_rows WHERE is_changed_from_v3b = 1").fetchone()[0]

print(f"V3c: pairs_count={v3c_pairs_count}, empty_rows={v3c_empty_rows}, changed_from_best={v3c_rows_changed_from_best}, changed_from_v3b={v3c_rows_changed_from_v3b}")

print(f"Elapsed: {time.time() - t0:.2f}s")
