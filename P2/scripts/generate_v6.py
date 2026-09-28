import os
import sys
import time
import duckdb

def main():
    repo_root = "c:/NEW AMAZON"
    p2_data = os.path.join(repo_root, "Amazon-ML-Challenge-2026", "P2", "data")
    v4_path = os.path.join(p2_data, "matching_results_v4.tsv")
    v5_path = os.path.join(p2_data, "matching_results_v5.tsv")
    pred_s2 = os.path.join(repo_root, "Amazon-ML-Challenge-2026", "P2", "predictions", "phase4", "test_predictions_s2.tsv")
    pred_s3 = os.path.join(repo_root, "Amazon-ML-Challenge-2026", "P2", "predictions", "phase4", "test_predictions_s3.tsv")
    gt_path = os.path.join(repo_root, "Amazon-ML-Challenge-2026", "outputs", "person1_step1", "train_ground_truth_reconstructed.tsv")
    oof_pattern = "E:/predictions/phase4/v3_train_oof_fold*.parquet"

    out_v6 = os.path.join(p2_data, "matching_results_v6.tsv")
    out_v6_on_v5 = os.path.join(p2_data, "matching_results_v6_on_v5.tsv")

    con = duckdb.connect()
    con.execute("PRAGMA threads=8;")
    con.execute("PRAGMA memory_limit='24GB';")

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

    best_X = 0.90
    print(f"Using best X = {best_X}")

    # =========================================================
    # BUILD V6 (Base = V4)
    # =========================================================
    print("\n--- Building V6 (Base = V4) ---")
    con.execute(f"""
    CREATE TEMP TABLE v4_raw AS
    SELECT 
        row_number() over () as row_id,
        source1_entity_id as s1_id,
        COALESCE(matched_entity_ids, '') as v4_matched_ids,
        CASE WHEN matched_entity_ids IS NULL OR TRIM(matched_entity_ids) = '' THEN 1 ELSE 0 END as is_empty_in_v4
    FROM read_csv('{v4_path.replace(os.sep, "/")}', delim='\t', header=True, all_varchar=True);
    """)

    con.execute("""
    CREATE TEMP TABLE v4_exploded AS
    WITH split_pairs AS (
        SELECT 
            s1_id,
            unnest(string_split(v4_matched_ids, ',')) as candidate_id
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

    # Candidate pool for V4 empty rows: score >= best_X (0.90), not in v4_used_cands
    con.execute(f"""
    CREATE TEMP TABLE v6_pool AS
    SELECT 
        r.s1_id,
        t.candidate_id,
        t.score,
        CASE WHEN t.candidate_id LIKE 'S2-%' THEN 'S2' ELSE 'S3' END as target_type,
        ROW_NUMBER() OVER (
            PARTITION BY r.s1_id, CASE WHEN t.candidate_id LIKE 'S2-%' THEN 'S2' ELSE 'S3' END
            ORDER BY t.score DESC, t.candidate_id ASC
        ) as rnk_per_s1
    FROM v4_raw r
    JOIN our_test t ON r.s1_id = t.s1_id
    WHERE r.is_empty_in_v4 = 1
      AND t.score >= {best_X}
      AND t.candidate_id NOT IN (SELECT candidate_id FROM v4_used_cands);

    -- Keep top-1 S2 and top-1 S3 per S1
    CREATE TEMP TABLE v6_top1_per_s1 AS
    SELECT s1_id, candidate_id, score, target_type
    FROM v6_pool
    WHERE rnk_per_s1 = 1;

    -- "Give any candidate_id claimed by several rows only to its highest-scoring row"
    CREATE TEMP TABLE v6_deduped_adds AS
    WITH ranked_global AS (
        SELECT 
            s1_id,
            candidate_id,
            score,
            target_type,
            ROW_NUMBER() OVER (
                PARTITION BY candidate_id 
                ORDER BY score DESC, s1_id ASC
            ) as rnk_cand
        FROM v6_top1_per_s1
    )
    SELECT s1_id, candidate_id, score, target_type,
           CASE WHEN target_type = 'S2' THEN 1 ELSE 2 END as pos
    FROM ranked_global
    WHERE rnk_cand = 1;
    """)

    v6_pairs_added = con.execute("SELECT COUNT(*) FROM v6_deduped_adds").fetchone()[0]
    v6_rows_filled = con.execute("SELECT COUNT(DISTINCT s1_id) FROM v6_deduped_adds").fetchone()[0]
    v6_empty_left = 82722 - v6_rows_filled

    print(f"V6 Pairs Added:   {v6_pairs_added}")
    print(f"V6 Rows Filled:   {v6_rows_filled}")
    print(f"V6 Empty Left:    {v6_empty_left}")

    # Build final V6 table
    con.execute("""
    CREATE TEMP TABLE v6_aggregated_adds AS
    SELECT 
        s1_id,
        string_agg(candidate_id, ',' ORDER BY pos ASC) as added_ids
    FROM v6_deduped_adds
    GROUP BY s1_id;

    CREATE TEMP TABLE v6_full_rows AS
    SELECT 
        r.row_id,
        r.s1_id,
        CASE 
            WHEN a.added_ids IS NOT NULL AND a.added_ids != '' THEN a.added_ids
            ELSE r.v4_matched_ids
        END as v6_matched_ids
    FROM v4_raw r
    LEFT JOIN v6_aggregated_adds a ON r.s1_id = a.s1_id;
    """)

    # =========================================================
    # BUILD V6_on_V5 (Base = V5)
    # =========================================================
    print("\n--- Building V6_on_V5 (Base = V5) ---")
    con.execute(f"""
    CREATE TEMP TABLE v5_raw AS
    SELECT 
        row_number() over () as row_id,
        source1_entity_id as s1_id,
        COALESCE(matched_entity_ids, '') as v5_matched_ids,
        CASE WHEN matched_entity_ids IS NULL OR TRIM(matched_entity_ids) = '' THEN 1 ELSE 0 END as is_empty_in_v5
    FROM read_csv('{v5_path.replace(os.sep, "/")}', delim='\t', header=True, all_varchar=True);
    """)

    con.execute("""
    CREATE TEMP TABLE v5_exploded AS
    WITH split_pairs AS (
        SELECT 
            s1_id,
            unnest(string_split(v5_matched_ids, ',')) as candidate_id
        FROM v5_raw
        WHERE is_empty_in_v5 = 0
    )
    SELECT 
        s1_id,
        TRIM(candidate_id) as candidate_id
    FROM split_pairs;
    """)

    con.execute("""
    CREATE TEMP TABLE v5_used_cands AS
    SELECT DISTINCT candidate_id FROM v5_exploded;
    """)

    con.execute(f"""
    CREATE TEMP TABLE v6_v5_pool AS
    SELECT 
        r.s1_id,
        t.candidate_id,
        t.score,
        CASE WHEN t.candidate_id LIKE 'S2-%' THEN 'S2' ELSE 'S3' END as target_type,
        ROW_NUMBER() OVER (
            PARTITION BY r.s1_id, CASE WHEN t.candidate_id LIKE 'S2-%' THEN 'S2' ELSE 'S3' END
            ORDER BY t.score DESC, t.candidate_id ASC
        ) as rnk_per_s1
    FROM v5_raw r
    JOIN our_test t ON r.s1_id = t.s1_id
    WHERE r.is_empty_in_v5 = 1
      AND t.score >= {best_X}
      AND t.candidate_id NOT IN (SELECT candidate_id FROM v5_used_cands);

    -- Keep top-1 S2 and top-1 S3 per S1
    CREATE TEMP TABLE v6_v5_top1_per_s1 AS
    SELECT s1_id, candidate_id, score, target_type
    FROM v6_v5_pool
    WHERE rnk_per_s1 = 1;

    -- Give any candidate_id claimed by several rows only to its highest-scoring row
    CREATE TEMP TABLE v6_v5_deduped_adds AS
    WITH ranked_global AS (
        SELECT 
            s1_id,
            candidate_id,
            score,
            target_type,
            ROW_NUMBER() OVER (
                PARTITION BY candidate_id 
                ORDER BY score DESC, s1_id ASC
            ) as rnk_cand
        FROM v6_v5_top1_per_s1
    )
    SELECT s1_id, candidate_id, score, target_type,
           CASE WHEN target_type = 'S2' THEN 1 ELSE 2 END as pos
    FROM ranked_global
    WHERE rnk_cand = 1;
    """)

    v6_v5_pairs_added = con.execute("SELECT COUNT(*) FROM v6_v5_deduped_adds").fetchone()[0]
    v6_v5_rows_filled = con.execute("SELECT COUNT(DISTINCT s1_id) FROM v6_v5_deduped_adds").fetchone()[0]
    v6_v5_empty_left = 82722 - v6_v5_rows_filled

    print(f"V6_on_V5 Pairs Added: {v6_v5_pairs_added}")
    print(f"V6_on_V5 Rows Filled: {v6_v5_rows_filled}")
    print(f"V6_on_V5 Empty Left:  {v6_v5_empty_left}")

    # Build final V6_on_V5 table
    con.execute("""
    CREATE TEMP TABLE v6_v5_aggregated_adds AS
    SELECT 
        s1_id,
        string_agg(candidate_id, ',' ORDER BY pos ASC) as added_ids
    FROM v6_v5_deduped_adds
    GROUP BY s1_id;

    CREATE TEMP TABLE v6_v5_full_rows AS
    SELECT 
        r.row_id,
        r.s1_id,
        CASE 
            WHEN a.added_ids IS NOT NULL AND a.added_ids != '' THEN a.added_ids
            ELSE r.v5_matched_ids
        END as v6_v5_matched_ids
    FROM v5_raw r
    LEFT JOIN v6_v5_aggregated_adds a ON r.s1_id = a.s1_id;
    """)

    # Write files
    print("\nWriting V6 TSV...")
    t_w0 = time.time()
    v6_rows = con.execute("SELECT s1_id, v6_matched_ids FROM v6_full_rows ORDER BY row_id ASC").fetchall()
    with open(out_v6, "w", encoding="utf-8", newline="\n") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for s1, m in v6_rows:
            f.write(f"{s1}\t{m}\n")
    print(f"Wrote {out_v6} in {time.time() - t_w0:.2f}s")

    print("\nWriting V6_on_V5 TSV...")
    t_w0 = time.time()
    v6_v5_rows = con.execute("SELECT s1_id, v6_v5_matched_ids FROM v6_v5_full_rows ORDER BY row_id ASC").fetchall()
    with open(out_v6_on_v5, "w", encoding="utf-8", newline="\n") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for s1, m in v6_v5_rows:
            f.write(f"{s1}\t{m}\n")
    print(f"Wrote {out_v6_on_v5} in {time.time() - t_w0:.2f}s")

    # Validation
    print("\nValidating files...")
    for label, path in [("V6", out_v6), ("V6_on_V5", out_v6_on_v5)]:
        line_count = 0
        has_dup = False
        with open(path, "r", encoding="utf-8") as f:
            header = f.readline().rstrip("\n")
            assert header == "source1_entity_id\tmatched_entity_ids", f"Invalid header in {path}: {header}"
            for line in f:
                line_count += 1
                parts = line.rstrip("\n").split("\t")
                assert len(parts) == 2, f"Line {line_count} does not have 2 fields in {path}"
                ids_str = parts[1]
                if ids_str:
                    ids_list = ids_str.split(",")
                    if len(ids_list) != len(set(ids_list)):
                        has_dup = True
                        break
        assert line_count == 1732544, f"Row count mismatch in {path}: expected 1732544, got {line_count}"
        assert not has_dup, f"Duplicate ids found within a row in {path}"
        print(f"Validation PASSED for {label} ({path}): 1,732,544 rows, header valid, no duplicate ids.")

if __name__ == "__main__":
    main()
