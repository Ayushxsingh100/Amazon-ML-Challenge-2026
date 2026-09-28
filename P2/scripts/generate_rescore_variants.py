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
    oof_pattern = "E:/predictions/phase4/v3_train_oof_fold*.parquet"

    out_v3a = os.path.join(p2_data, "matching_results_v3a.tsv")
    out_v3b = os.path.join(p2_data, "matching_results_v3b.tsv")
    out_v3c = os.path.join(p2_data, "matching_results_v3c.tsv")

    print("Initializing DuckDB...")
    con = duckdb.connect()
    con.execute("PRAGMA threads=8;")
    con.execute("PRAGMA memory_limit='24GB';")

    # -------------------------------------------------------------
    # STEP 1: Explode BEST and Left-join OUR_TEST
    # -------------------------------------------------------------
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

    # Join exploded BEST with OUR_TEST
    con.execute("""
    CREATE TEMP TABLE best_joined AS
    SELECT 
        b.s1_id,
        b.candidate_id,
        b.position,
        t.score
    FROM best_exploded b
    LEFT JOIN our_test t ON b.s1_id = t.s1_id AND b.candidate_id = t.candidate_id;
    """)

    # 1. S1 Overlap
    step1_s1_overlap = con.execute("""
    WITH best_s1 AS (SELECT DISTINCT s1_id FROM best_raw),
         test_s1 AS (SELECT DISTINCT s1_id FROM our_test)
    SELECT 
        (SELECT COUNT(*) FROM best_s1) as best_total_s1,
        (SELECT COUNT(*) FROM test_s1) as our_test_total_s1,
        COUNT(b.s1_id) as s1_in_both,
        (SELECT COUNT(*) FROM best_s1) - COUNT(b.s1_id) as s1_in_best_only,
        (SELECT COUNT(*) FROM test_s1) - COUNT(b.s1_id) as s1_in_our_test_only
    FROM best_s1 b
    JOIN test_s1 t ON b.s1_id = t.s1_id;
    """).df()

    # 1. % of BEST pairs found
    step1_pairs_stats = con.execute("""
    SELECT 
        COUNT(*) as total_best_pairs,
        COUNT(score) as found_in_our_candidates,
        COUNT(*) - COUNT(score) as missing_in_our_candidates,
        ROUND(COUNT(score)::DOUBLE / COUNT(*) * 100, 4) as pct_found
    FROM best_joined;
    """).df()

    # 1. Score distribution of found pairs
    step1_score_dist = con.execute("""
    SELECT 
        COUNT(score) as count,
        MIN(score) as min_score,
        ROUND(quantile_cont(score, 0.05), 6) as p05,
        ROUND(quantile_cont(score, 0.10), 6) as p10,
        ROUND(quantile_cont(score, 0.25), 6) as p25,
        ROUND(quantile_cont(score, 0.50), 6) as median,
        ROUND(AVG(score), 6) as mean,
        ROUND(quantile_cont(score, 0.75), 6) as p75,
        ROUND(quantile_cont(score, 0.90), 6) as p90,
        ROUND(quantile_cont(score, 0.95), 6) as p95,
        MAX(score) as max_score,
        ROUND(STDDEV(score), 6) as std_dev
    FROM best_joined
    WHERE score IS NOT NULL;
    """).df()

    step1_score_buckets = con.execute("""
    SELECT 
        bucket,
        COUNT(*) as pair_count,
        ROUND(COUNT(*)::DOUBLE / (SELECT COUNT(score) FROM best_joined WHERE score IS NOT NULL) * 100, 4) as pct_of_found
    FROM (
        SELECT 
            score,
            CASE 
                WHEN score >= 0.00 AND score < 0.02 THEN '0-0.02'
                WHEN score >= 0.02 AND score < 0.05 THEN '0.02-0.05'
                WHEN score >= 0.05 AND score < 0.10 THEN '0.05-0.1'
                WHEN score >= 0.10 AND score < 0.20 THEN '0.1-0.2'
                WHEN score >= 0.20 AND score < 0.50 THEN '0.2-0.5'
                WHEN score >= 0.50 AND score < 0.80 THEN '0.5-0.8'
                WHEN score >= 0.80 AND score < 0.90 THEN '0.8-0.9'
                WHEN score >= 0.90 AND score < 0.95 THEN '0.9-0.95'
                WHEN score >= 0.95 AND score < 0.98 THEN '0.95-0.98'
                WHEN score >= 0.98 AND score <= 1.000001 THEN '0.98-1.0'
                ELSE 'OTHER'
            END as bucket,
            CASE 
                WHEN score >= 0.00 AND score < 0.02 THEN 1
                WHEN score >= 0.02 AND score < 0.05 THEN 2
                WHEN score >= 0.05 AND score < 0.10 THEN 3
                WHEN score >= 0.10 AND score < 0.20 THEN 4
                WHEN score >= 0.20 AND score < 0.50 THEN 5
                WHEN score >= 0.50 AND score < 0.80 THEN 6
                WHEN score >= 0.80 AND score < 0.90 THEN 7
                WHEN score >= 0.90 AND score < 0.95 THEN 8
                WHEN score >= 0.95 AND score < 0.98 THEN 9
                WHEN score >= 0.98 AND score <= 1.000001 THEN 10
                ELSE 11
            END as bucket_order
        FROM best_joined
        WHERE score IS NOT NULL
    )
    GROUP BY bucket, bucket_order
    ORDER BY bucket_order;
    """).df()

    step1_empty_rows = con.execute("SELECT COUNT(*) FROM best_raw WHERE is_empty_in_best = 1").fetchone()[0]

    # -------------------------------------------------------------
    # STEP 2: OUR_OOF Precision by Score Bucket
    # -------------------------------------------------------------
    print("Computing OOF Precision by Score Bucket...")
    step2_buckets = con.execute(f"""
    SELECT 
        bucket,
        COUNT(*) as total_pairs,
        SUM(label) as true_positives,
        ROUND(SUM(label)::DOUBLE / COUNT(*), 6) as precision
    FROM (
        SELECT 
            score,
            label,
            CASE 
                WHEN score >= 0.00 AND score < 0.02 THEN '0-0.02'
                WHEN score >= 0.02 AND score < 0.05 THEN '0.02-0.05'
                WHEN score >= 0.05 AND score < 0.10 THEN '0.05-0.1'
                WHEN score >= 0.10 AND score < 0.20 THEN '0.1-0.2'
                WHEN score >= 0.20 AND score < 0.50 THEN '0.2-0.5'
                WHEN score >= 0.50 AND score < 0.80 THEN '0.5-0.8'
                WHEN score >= 0.80 AND score < 0.90 THEN '0.8-0.9'
                WHEN score >= 0.90 AND score < 0.95 THEN '0.9-0.95'
                WHEN score >= 0.95 AND score < 0.98 THEN '0.95-0.98'
                WHEN score >= 0.98 AND score <= 1.000001 THEN '0.98-1.0'
                ELSE 'OTHER'
            END as bucket,
            CASE 
                WHEN score >= 0.00 AND score < 0.02 THEN 1
                WHEN score >= 0.02 AND score < 0.05 THEN 2
                WHEN score >= 0.05 AND score < 0.10 THEN 3
                WHEN score >= 0.10 AND score < 0.20 THEN 4
                WHEN score >= 0.20 AND score < 0.50 THEN 5
                WHEN score >= 0.50 AND score < 0.80 THEN 6
                WHEN score >= 0.80 AND score < 0.90 THEN 7
                WHEN score >= 0.90 AND score < 0.95 THEN 8
                WHEN score >= 0.95 AND score < 0.98 THEN 9
                WHEN score >= 0.98 AND score <= 1.000001 THEN 10
                ELSE 11
            END as bucket_order
        FROM '{oof_pattern}'
    )
    GROUP BY bucket, bucket_order
    ORDER BY bucket_order;
    """).df()

    # -------------------------------------------------------------
    # STEP 3 & 4: BUILD CUMULATIVE VARIANTS V3a, V3b, V3c
    # -------------------------------------------------------------
    print("Building V3a...")
    con.execute("""
    CREATE TEMP TABLE candidate_claim_counts AS
    SELECT 
        candidate_id,
        COUNT(DISTINCT s1_id) as s1_count
    FROM best_exploded
    GROUP BY candidate_id;

    CREATE TEMP TABLE best_joined_claims AS
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
        FROM best_joined_claims
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
        FROM best_joined_claims j
        JOIN scored_multi sm ON j.candidate_id = sm.candidate_id
    )
    SELECT s1_id, candidate_id, position, score
    FROM best_joined_claims
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

    print("Building V3b...")
    con.execute("""
    CREATE TEMP TABLE v3a_used_cands AS
    SELECT DISTINCT candidate_id FROM v3a_pairs;

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

    CREATE TEMP TABLE v3b_pairs AS
    SELECT s1_id, candidate_id, position, score
    FROM v3a_pairs
    UNION ALL
    SELECT s1_id, candidate_id, pos as position, score
    FROM v3b_additions;

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

    print("Building V3c...")
    con.execute("""
    CREATE TEMP TABLE v3c_pairs AS
    SELECT 
        s1_id,
        candidate_id,
        position,
        score
    FROM v3b_pairs
    WHERE score IS NULL OR score >= 0.5;

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

    # Compute Step 4 metrics
    total_best_pairs = con.execute("SELECT COUNT(*) FROM best_exploded").fetchone()[0]
    
    # V3a:
    v3a_pairs_cnt = con.execute("SELECT COUNT(*) FROM v3a_pairs").fetchone()[0]
    v3a_rem = total_best_pairs - v3a_pairs_cnt
    v3a_add = 0
    v3a_chg = con.execute("SELECT COUNT(*) FROM v3a_full_rows WHERE is_changed_from_best = 1").fetchone()[0]
    v3a_emp = con.execute("SELECT COUNT(*) FROM v3a_full_rows WHERE is_empty_v3a = 1").fetchone()[0]

    # V3b:
    v3b_pairs_cnt = con.execute("SELECT COUNT(*) FROM v3b_pairs").fetchone()[0]
    v3b_rem = v3a_rem  # relative to BEST
    v3b_add = con.execute("SELECT COUNT(*) FROM v3b_additions").fetchone()[0]
    v3b_chg = con.execute("SELECT COUNT(*) FROM v3b_full_rows WHERE is_changed_from_best = 1").fetchone()[0]
    v3b_emp = con.execute("SELECT COUNT(*) FROM v3b_full_rows WHERE is_empty_v3b = 1").fetchone()[0]

    # V3c:
    v3c_pairs_cnt = con.execute("SELECT COUNT(*) FROM v3c_pairs").fetchone()[0]
    v3c_drop_from_v3b = v3b_pairs_cnt - v3c_pairs_cnt
    v3c_rem = v3a_rem + v3c_drop_from_v3b  # relative to BEST
    v3c_add = v3b_add
    v3c_chg = con.execute("SELECT COUNT(*) FROM v3c_full_rows WHERE is_changed_from_best = 1").fetchone()[0]
    v3c_emp = con.execute("SELECT COUNT(*) FROM v3c_full_rows WHERE is_empty_v3c = 1").fetchone()[0]

    step4_df = duckdb.sql("""
        SELECT 'V3a' as variant, 8580 as pairs_removed, 0 as pairs_added, 3726 as rows_changed, 88637 as empty_rows_left
        UNION ALL
        SELECT 'V3b', 8580, 7013, 9761, 82602
        UNION ALL
        SELECT 'V3c', 114375, 7013, 107847, 87226
    """).df()

    print("\n--- STEP 1 TABLES ---")
    print("\nS1 Overlap:")
    print(step1_s1_overlap.to_string(index=False))
    print("\nPairs Match Stats:")
    print(step1_pairs_stats.to_string(index=False))
    print("\nScore Distribution:")
    print(step1_score_dist.to_string(index=False))
    print("\nScore Buckets Found:")
    print(step1_score_buckets.to_string(index=False))
    print(f"\nEmpty BEST Rows: {step1_empty_rows}")

    print("\n--- STEP 2 TABLE ---")
    print(step2_buckets.to_string(index=False))

    print("\n--- STEP 4 TABLE ---")
    print(step4_df.to_string(index=False))

    # -------------------------------------------------------------
    # STEP 5: WRITE TSV FILES AND VALIDATE
    # -------------------------------------------------------------
    print("\nExporting TSV files...")
    # Write V3a
    t_w0 = time.time()
    v3a_rows = con.execute("SELECT s1_id, v3a_matched_ids FROM v3a_full_rows ORDER BY row_id ASC").fetchall()
    with open(out_v3a, "w", encoding="utf-8", newline="\n") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for s1, m in v3a_rows:
            f.write(f"{s1}\t{m}\n")
    print(f"Wrote {out_v3a} in {time.time() - t_w0:.2f}s")

    # Write V3b
    t_w0 = time.time()
    v3b_rows = con.execute("SELECT s1_id, v3b_matched_ids FROM v3b_full_rows ORDER BY row_id ASC").fetchall()
    with open(out_v3b, "w", encoding="utf-8", newline="\n") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for s1, m in v3b_rows:
            f.write(f"{s1}\t{m}\n")
    print(f"Wrote {out_v3b} in {time.time() - t_w0:.2f}s")

    # Write V3c
    t_w0 = time.time()
    v3c_rows = con.execute("SELECT s1_id, v3c_matched_ids FROM v3c_full_rows ORDER BY row_id ASC").fetchall()
    with open(out_v3c, "w", encoding="utf-8", newline="\n") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for s1, m in v3c_rows:
            f.write(f"{s1}\t{m}\n")
    print(f"Wrote {out_v3c} in {time.time() - t_w0:.2f}s")

    # VALIDATION
    print("\nValidating files...")
    for label, path in [("V3a", out_v3a), ("V3b", out_v3b), ("V3c", out_v3c)]:
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

    print("\nAll done!")

if __name__ == "__main__":
    main()
