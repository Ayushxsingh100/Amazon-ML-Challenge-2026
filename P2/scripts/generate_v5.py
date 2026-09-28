import os
import sys
import time
import duckdb

def main():
    repo_root = "c:/NEW AMAZON"
    p2_data = os.path.join(repo_root, "Amazon-ML-Challenge-2026", "P2", "data")
    v4_path = os.path.join(p2_data, "matching_results_v4.tsv")
    pred_s2 = os.path.join(repo_root, "Amazon-ML-Challenge-2026", "P2", "predictions", "phase4", "test_predictions_s2.tsv")
    pred_s3 = os.path.join(repo_root, "Amazon-ML-Challenge-2026", "P2", "predictions", "phase4", "test_predictions_s3.tsv")
    oof_pattern = "E:/predictions/phase4/v3_train_oof_fold*.parquet"
    out_v5 = os.path.join(p2_data, "matching_results_v5.tsv")

    con = duckdb.connect()
    con.execute("PRAGMA threads=8;")
    con.execute("PRAGMA memory_limit='24GB';")

    print("Loading V4...")
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
    # From OUR_TEST, take pairs with score >= 0.98 whose S1 row is NOT empty in V4
    # and which are NOT already in V4. Exclude candidate_ids already used anywhere in V4.
    # If a candidate_id qualifies for more than one S1, keep only its highest score.
    print("Filtering qualifying candidate pool for V5...")
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

    # Step 2: Cutoff analysis
    print("\n--- STEP 2: Cutoff Analysis ---")
    cutoff_results = []
    for cutoff in [0.98, 0.99, 0.995, 0.999]:
        raw_res = con.execute(f"""
        SELECT COUNT(*), COUNT(DISTINCT s1_id)
        FROM v5_deduped_pool
        WHERE score >= {cutoff};
        """).fetchone()

        capped_res = con.execute(f"""
        WITH capped AS (
            SELECT 
                s1_id,
                candidate_id,
                score,
                ROW_NUMBER() OVER (PARTITION BY s1_id ORDER BY score DESC, candidate_id ASC) as rnk
            FROM v5_deduped_pool
            WHERE score >= {cutoff}
        )
        SELECT COUNT(*), COUNT(DISTINCT s1_id)
        FROM capped
        WHERE rnk <= 2;
        """).fetchone()

        cutoff_results.append({
            "cutoff": cutoff,
            "raw_pairs": raw_res[0],
            "raw_rows": raw_res[1],
            "capped_pairs": capped_res[0],
            "capped_rows": capped_res[1]
        })
        print(f"Cutoff {cutoff:5.3f}: Raw Pairs = {raw_res[0]:6d}, Raw Rows = {raw_res[1]:6d} | Capped Pairs = {capped_res[0]:6d}, Capped Rows = {capped_res[1]:6d}")

    # Step 3: OOF Precision for sub-buckets
    print("\n--- STEP 3: OOF Precision ---")
    step3_df = con.execute(f"""
    SELECT 
        bucket,
        COUNT(*) as total_pairs,
        SUM(label) as true_positives,
        ROUND(SUM(label)::DOUBLE / COUNT(*), 6) as precision,
        ROUND(SUM(label)::DOUBLE / COUNT(*) * 100, 4) as precision_pct
    FROM (
        SELECT 
            score,
            label,
            CASE 
                WHEN score >= 0.980 AND score < 0.990 THEN '0.98-0.99'
                WHEN score >= 0.990 AND score < 0.995 THEN '0.99-0.995'
                WHEN score >= 0.995 AND score < 0.999 THEN '0.995-0.999'
                WHEN score >= 0.999 AND score <= 1.000001 THEN '0.999-1.0'
                ELSE 'OTHER'
            END as bucket,
            CASE 
                WHEN score >= 0.980 AND score < 0.990 THEN 1
                WHEN score >= 0.990 AND score < 0.995 THEN 2
                WHEN score >= 0.995 AND score < 0.999 THEN 3
                WHEN score >= 0.999 AND score <= 1.000001 THEN 4
                ELSE 5
            END as bucket_order
        FROM '{oof_pattern}'
        WHERE score >= 0.98
    )
    GROUP BY bucket, bucket_order
    ORDER BY bucket_order;
    """).df()
    print(step3_df.to_string(index=False))

    # Step 4: Build V5 using cutoff 0.995 (lowest cutoff whose OOF bucket precision is >= 97%)
    # Add at most 2 new pairs per S1 row
    print("\n--- STEP 4 & 5: Building V5 (Cutoff = 0.995, at most 2 pairs/row) ---")
    chosen_cutoff = 0.995
    con.execute(f"""
    CREATE TEMP TABLE v5_additions AS
    WITH ranked_per_s1 AS (
        SELECT 
            s1_id,
            candidate_id,
            score,
            ROW_NUMBER() OVER (PARTITION BY s1_id ORDER BY score DESC, candidate_id ASC) as rnk
        FROM v5_deduped_pool
        WHERE score >= {chosen_cutoff}
    )
    SELECT s1_id, candidate_id, score, rnk
    FROM ranked_per_s1
    WHERE rnk <= 2;
    """)

    pairs_added = con.execute("SELECT COUNT(*) FROM v5_additions").fetchone()[0]
    rows_changed = con.execute("SELECT COUNT(DISTINCT s1_id) FROM v5_additions").fetchone()[0]
    print(f"V5 Pairs Added: {pairs_added}")
    print(f"V5 Rows Changed: {rows_changed}")

    con.execute("""
    CREATE TEMP TABLE v5_aggregated_adds AS
    SELECT 
        s1_id,
        string_agg(candidate_id, ',' ORDER BY rnk ASC) as added_ids
    FROM v5_additions
    GROUP BY s1_id;

    CREATE TEMP TABLE v5_full_rows AS
    SELECT 
        r.row_id,
        r.s1_id,
        r.v4_matched_ids,
        CASE 
            WHEN a.added_ids IS NOT NULL AND a.added_ids != '' THEN
                r.v4_matched_ids || ',' || a.added_ids
            ELSE
                r.v4_matched_ids
        END as v5_matched_ids
    FROM v4_raw r
    LEFT JOIN v5_aggregated_adds a ON r.s1_id = a.s1_id;
    """)

    # Write P2/data/matching_results_v5.tsv
    print("\nWriting matching_results_v5.tsv...")
    t_w0 = time.time()
    v5_rows = con.execute("SELECT s1_id, v5_matched_ids FROM v5_full_rows ORDER BY row_id ASC").fetchall()
    with open(out_v5, "w", encoding="utf-8", newline="\n") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for s1, m in v5_rows:
            f.write(f"{s1}\t{m}\n")
    print(f"Wrote {out_v5} in {time.time() - t_w0:.2f}s")

    # Validation
    print("\nValidating matching_results_v5.tsv...")
    line_count = 0
    has_dup = False
    with open(out_v5, "r", encoding="utf-8") as f:
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
    print(f"Validation PASSED for V5 ({out_v5}): 1,732,544 rows, header valid, no duplicate ids.")

if __name__ == "__main__":
    main()
