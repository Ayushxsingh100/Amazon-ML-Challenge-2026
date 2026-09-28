import os
import duckdb

repo_root = "c:/NEW AMAZON"
p2_data = os.path.join(repo_root, "Amazon-ML-Challenge-2026", "P2", "data")
v4_path = os.path.join(p2_data, "matching_results_v4.tsv")
pred_s2 = os.path.join(repo_root, "Amazon-ML-Challenge-2026", "P2", "predictions", "phase4", "test_predictions_s2.tsv")
pred_s3 = os.path.join(repo_root, "Amazon-ML-Challenge-2026", "P2", "predictions", "phase4", "test_predictions_s3.tsv")

con = duckdb.connect()
con.execute("PRAGMA threads=8;")
con.execute("PRAGMA memory_limit='24GB';")

print("Loading V4...")
con.execute(f"""
CREATE TEMP TABLE v4_raw AS
SELECT 
    row_number() over () as row_id,
    source1_entity_id as s1_id,
    COALESCE(matched_entity_ids, '') as orig_matched_ids,
    CASE WHEN matched_entity_ids IS NULL OR TRIM(matched_entity_ids) = '' THEN 1 ELSE 0 END as is_empty_in_v4
FROM read_csv('{v4_path.replace(os.sep, "/")}', delim='\t', header=True, all_varchar=True);
""")

con.execute("""
CREATE TEMP TABLE v4_exploded AS
WITH split_pairs AS (
    SELECT 
        s1_id,
        unnest(string_split(orig_matched_ids, ',')) as candidate_id
    FROM v4_raw
    WHERE is_empty_in_v4 = 0
)
SELECT 
    s1_id,
    TRIM(candidate_id) as candidate_id
FROM split_pairs;
""")

con.execute("""
CREATE TEMP TABLE v4_used_cands AS
SELECT DISTINCT candidate_id FROM v4_exploded;
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

# Step 1:
# "take pairs with score >= 0.98 whose S1 row is NOT empty in V4 and which are NOT already in V4.
#  Exclude candidate_ids already used anywhere in V4.
#  If a candidate_id qualifies for more than one S1, keep only its highest score."

print("Filtering qualifying candidate pool...")
con.execute("""
CREATE TEMP TABLE v5_candidate_pool AS
SELECT 
    t.s1_id,
    t.candidate_id,
    t.score
FROM our_test t
JOIN v4_raw r ON t.s1_id = r.s1_id
WHERE r.is_empty_in_v4 = 0
  AND t.score >= 0.98
  AND t.candidate_id NOT IN (SELECT candidate_id FROM v4_used_cands);
""")

# Note: Since candidate_id NOT IN v4_used_cands, and v4_used_cands contains ALL candidates in V4,
# that automatically guarantees the pair (s1_id, candidate_id) is NOT already in V4!

# Now resolve candidate conflicts:
# "If a candidate_id qualifies for more than one S1, keep only its highest score."
con.execute("""
CREATE TEMP TABLE v5_deduped_pool AS
WITH ranked AS (
    SELECT 
        s1_id,
        candidate_id,
        score,
        ROW_NUMBER() OVER (
            PARTITION BY candidate_id 
            ORDER BY score DESC, s1_id ASC
        ) as rnk
    FROM v5_candidate_pool
)
SELECT s1_id, candidate_id, score
FROM ranked
WHERE rnk = 1;
""")

print("--- Step 2: Cutoff Analysis ---")
# Let's inspect counts at cutoffs 0.98, 0.99, 0.995, 0.999
# Both raw (before capping per S1) and capped (at most 2 new pairs per S1)
for cutoff in [0.98, 0.99, 0.995, 0.999]:
    stats_raw = con.execute(f"""
    SELECT 
        COUNT(*) as pairs,
        COUNT(DISTINCT s1_id) as rows_affected
    FROM v5_deduped_pool
    WHERE score >= {cutoff};
    """).fetchone()

    # If capped at 2 per S1:
    stats_capped = con.execute(f"""
    WITH capped AS (
        SELECT 
            s1_id,
            candidate_id,
            score,
            ROW_NUMBER() OVER (PARTITION BY s1_id ORDER BY score DESC, candidate_id ASC) as rnk
        FROM v5_deduped_pool
        WHERE score >= {cutoff}
    )
    SELECT 
        COUNT(*) as pairs_capped,
        COUNT(DISTINCT s1_id) as rows_affected
    FROM capped
    WHERE rnk <= 2;
    """).fetchone()

    print(f"Cutoff {cutoff:5.3f} | Raw: pairs={stats_raw[0]:7d}, rows={stats_raw[1]:7d} | Capped (max 2/row): pairs={stats_capped[0]:7d}, rows={stats_capped[1]:7d}")
