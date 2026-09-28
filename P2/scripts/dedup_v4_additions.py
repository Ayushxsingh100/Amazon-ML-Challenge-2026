import os
import sys
import time
import duckdb

def main():
    repo_root = "c:/NEW AMAZON"
    p2_data = os.path.join(repo_root, "Amazon-ML-Challenge-2026", "P2", "data")
    best_path = os.path.join(p2_data, "matching_results_v1_conservative.tsv")
    pred_s2 = os.path.join(repo_root, "Amazon-ML-Challenge-2026", "P2", "predictions", "phase4", "test_predictions_s2.tsv")
    pred_s3 = os.path.join(repo_root, "Amazon-ML-Challenge-2026", "P2", "predictions", "phase4", "test_predictions_s3.tsv")
    out_v4 = os.path.join(p2_data, "matching_results_v4.tsv")

    con = duckdb.connect()
    con.execute("PRAGMA threads=8;")
    con.execute("PRAGMA memory_limit='24GB';")

    print("Loading BEST...")
    con.execute(f"""
    CREATE TEMP TABLE best_raw AS
    SELECT 
        row_number() over () as row_id,
        source1_entity_id as s1_id,
        COALESCE(matched_entity_ids, '') as orig_matched_ids,
        CASE WHEN matched_entity_ids IS NULL OR TRIM(matched_entity_ids) = '' THEN 1 ELSE 0 END as is_empty_in_best
    FROM read_csv('{best_path.replace(os.sep, "/")}', delim='\t', header=True, all_varchar=True);
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
    CREATE TEMP TABLE best_used_cands AS
    SELECT DISTINCT candidate_id FROM best_exploded;
    """)

    # 1. Candidate pool for empty S1s: score >= 0.98, not in best_used_cands
    print("Selecting 7,013 added candidates...")
    con.execute("""
    CREATE TEMP TABLE v4_fill_pool AS
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
      AND t.candidate_id NOT IN (SELECT candidate_id FROM best_used_cands);

    CREATE TEMP TABLE v4_added_7013 AS
    SELECT 
        s1_id,
        candidate_id,
        score,
        target_type,
        CASE WHEN target_type = 'S2' THEN 1 ELSE 2 END as position
    FROM v4_fill_pool
    WHERE rnk = 1;
    """)

    total_added_initial = con.execute("SELECT COUNT(*) FROM v4_added_7013").fetchone()[0]
    print(f"Total initial added pairs: {total_added_initial}")

    # Check multi-row candidate_ids among the 7,013 added pairs
    con.execute("""
    CREATE TEMP TABLE multi_added_counts AS
    SELECT 
        candidate_id,
        COUNT(DISTINCT s1_id) as row_count,
        COUNT(*) as pair_count
    FROM v4_added_7013
    GROUP BY candidate_id
    HAVING COUNT(DISTINCT s1_id) > 1;
    """)

    multi_cands_count = con.execute("SELECT COUNT(*) FROM multi_added_counts").fetchone()[0]
    multi_pairs_count = con.execute("SELECT SUM(pair_count) FROM multi_added_counts").fetchone()[0]
    print(f"\nCandidate IDs appearing in >1 row among added pairs: {multi_cands_count}")
    print(f"Total pair instances for these multi-row candidate IDs: {multi_pairs_count}")

    # Deduplicate: Keep each only in the row with our highest score
    # (If score ties occur, use deterministic tiebreaker: position ASC, s1_id ASC)
    con.execute("""
    CREATE TEMP TABLE v4_added_deduped AS
    WITH ranked_adds AS (
        SELECT 
            s1_id,
            candidate_id,
            score,
            target_type,
            position,
            ROW_NUMBER() OVER (
                PARTITION BY candidate_id 
                ORDER BY score DESC, position ASC, s1_id ASC
            ) as rnk
        FROM v4_added_7013
    )
    SELECT s1_id, candidate_id, score, target_type, position
    FROM ranked_adds
    WHERE rnk = 1;
    """)

    deduped_added_count = con.execute("SELECT COUNT(*) FROM v4_added_deduped").fetchone()[0]
    deduped_rows_filled = con.execute("SELECT COUNT(DISTINCT s1_id) FROM v4_added_deduped").fetchone()[0]
    pairs_removed_by_dedup = total_added_initial - deduped_added_count
    print(f"Pairs after candidate deduplication: {deduped_added_count} (dropped {pairs_removed_by_dedup} redundant claims)")
    print(f"Rows filled after deduplication: {deduped_rows_filled}")

    # 3. Prune BEST pairs with score < 0.02
    print("\nPruning BEST pairs with score < 0.02...")
    con.execute("""
    CREATE TEMP TABLE best_pruned_pairs AS
    SELECT 
        b.s1_id,
        b.candidate_id,
        b.position,
        t.score,
        CASE WHEN t.score IS NOT NULL AND t.score < 0.02 THEN 1 ELSE 0 END as is_dropped
    FROM best_exploded b
    LEFT JOIN our_test t ON b.s1_id = t.s1_id AND b.candidate_id = t.candidate_id;
    """)

    pairs_dropped = con.execute("SELECT COUNT(*) FROM best_pruned_pairs WHERE is_dropped = 1").fetchone()[0]

    # Combine kept BEST pairs + deduplicated fill additions
    con.execute("""
    CREATE TEMP TABLE v4_final_pairs AS
    SELECT s1_id, candidate_id, position
    FROM best_pruned_pairs
    WHERE is_dropped = 0
    UNION ALL
    SELECT s1_id, candidate_id, position
    FROM v4_added_deduped;
    """)

    con.execute("""
    CREATE TEMP TABLE v4_aggregated AS
    SELECT 
        s1_id,
        string_agg(candidate_id, ',' ORDER BY position ASC) as matched_entity_ids
    FROM v4_final_pairs
    GROUP BY s1_id;

    CREATE TEMP TABLE v4_full_rows AS
    SELECT 
        r.row_id,
        r.s1_id,
        r.orig_matched_ids,
        r.is_empty_in_best,
        COALESCE(a.matched_entity_ids, '') as v4_matched_ids,
        CASE WHEN a.matched_entity_ids IS NULL OR a.matched_entity_ids = '' THEN 1 ELSE 0 END as is_empty_in_v4,
        CASE WHEN COALESCE(a.matched_entity_ids, '') != r.orig_matched_ids THEN 1 ELSE 0 END as is_changed_from_best
    FROM best_raw r
    LEFT JOIN v4_aggregated a ON r.s1_id = a.s1_id;
    """)

    # Metrics
    rows_changed = con.execute("SELECT COUNT(*) FROM v4_full_rows WHERE is_changed_from_best = 1").fetchone()[0]
    empty_rows_left = con.execute("SELECT COUNT(*) FROM v4_full_rows WHERE is_empty_in_v4 = 1").fetchone()[0]
    rows_newly_emptied = con.execute("""
        SELECT COUNT(*) FROM v4_full_rows 
        WHERE is_empty_in_best = 0 AND is_empty_in_v4 = 1
    """).fetchone()[0]

    print(f"\n--- UPDATED V4 METRICS ---")
    print(f"Pairs Added:        {deduped_added_count}")
    print(f"Rows Filled:        {deduped_rows_filled}")
    print(f"Pairs Dropped:      {pairs_dropped}")
    print(f"Rows Changed:       {rows_changed}")
    print(f"Rows Newly Emptied: {rows_newly_emptied}")
    print(f"Empty Rows Left:    {empty_rows_left}")

    # Rewrite matching_results_v4.tsv
    print("\nRewriting matching_results_v4.tsv...")
    t_w0 = time.time()
    v4_rows = con.execute("SELECT s1_id, v4_matched_ids FROM v4_full_rows ORDER BY row_id ASC").fetchall()
    with open(out_v4, "w", encoding="utf-8", newline="\n") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for s1, m in v4_rows:
            f.write(f"{s1}\t{m}\n")
    print(f"Rewrote {out_v4} in {time.time() - t_w0:.2f}s")

    # Validation
    print("\nRe-running validation on matching_results_v4.tsv...")
    line_count = 0
    has_dup = False
    with open(out_v4, "r", encoding="utf-8") as f:
        header = f.readline().rstrip("\n")
        assert header == "source1_entity_id\tmatched_entity_ids", f"Invalid header: {header}"
        for line in f:
            line_count += 1
            parts = line.rstrip("\n").split("\t")
            assert len(parts) == 2, f"Line {line_count} does not have 2 fields"
            ids_str = parts[1]
            if ids_str:
                ids_list = ids_str.split(",")
                if len(ids_list) != len(set(ids_list)):
                    has_dup = True
                    break
    assert line_count == 1732544, f"Row count mismatch: expected 1732544, got {line_count}"
    assert not has_dup, "Duplicate ids found within a row"
    print(f"Validation PASSED for V4 ({out_v4}): 1,732,544 rows, header valid, no duplicate ids.")

if __name__ == "__main__":
    main()
