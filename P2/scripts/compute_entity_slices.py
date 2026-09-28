import os
import time
import duckdb
import numpy as np
import pandas as pd

REPO = r"c:\NEW AMAZON\Amazon-ML-Challenge-2026"
FOLDS_PATH = os.path.join(REPO, "P3", "reports", "folds_v1_manifest.tsv")
GT_PATH = os.path.join(REPO, "outputs", "person1_step1", "train_ground_truth_reconstructed.tsv")
S1_PATH = os.path.join(REPO, "outputs", "person1_step1", "normalized", "train_source1_normalized.tsv")
REPORT_DIR = os.path.join(REPO, "P2", "reports")
FEAT_DIR = os.path.join(REPO, "P2", "data", "features")
OUT_SLICES = os.path.join(REPORT_DIR, "E02_ENTITY_ERROR_SLICES.tsv")
TMP_DB = os.path.join(REPO, "P2", "data", "duckdb_slices_tmp.db")

if os.path.exists(TMP_DB):
    os.remove(TMP_DB)

con = duckdb.connect(TMP_DB)
con.execute("SET threads=4")
con.execute("SET memory_limit='6GB'")

t0 = time.time()
print("Step 1: Loading folds, S1 metadata, and ground truth...")

con.execute(f"""
    CREATE TABLE s1_meta AS
    SELECT 
        entity_id AS source1_entity_id,
        business_name,
        business_address,
        country,
        business_name IS NULL OR TRIM(business_name) = '' OR TRIM(business_name) = 'nan' AS missing_name,
        business_address IS NULL OR TRIM(business_address) = '' OR TRIM(business_address) = 'nan' AS missing_addr,
        regexp_extract(TRIM(business_address), '[0-9]+[A-Za-z]?', 0) AS house,
        (regexp_extract(TRIM(business_address), '[0-9]+[A-Za-z]?', 0) IS NULL OR regexp_extract(TRIM(business_address), '[0-9]+[A-Za-z]?', 0) = '') AS missing_house
    FROM read_csv('{S1_PATH}', delim='\\t', header=true, all_varchar=true)
""")

con.execute(f"CREATE TABLE folds AS SELECT source1_entity_id, fold FROM read_csv('{FOLDS_PATH}', delim='\\t', header=true)")

con.execute(f"""
    CREATE TABLE gt_counts AS 
    SELECT 
        source1_entity_id,
        CASE WHEN matched_entity_ids IS NULL OR matched_entity_ids = '' THEN 0 ELSE len(string_split(matched_entity_ids, ',')) END AS A,
        len(list_filter(string_split(coalesce(matched_entity_ids, ''), ','), x -> x LIKE 'S2-%')) AS A_s2,
        len(list_filter(string_split(coalesce(matched_entity_ids, ''), ','), x -> x LIKE 'S3-%')) AS A_s3
    FROM read_csv('{GT_PATH}', delim='\\t', header=true)
""")

print(f"Base tables ready in {time.time() - t0:.2f}s")

# Step 2: Aggregate candidate rule provenance per S1
print("Step 2: Aggregating rule provenance from features...")
t_rules = time.time()
con.execute(f"""
    CREATE TABLE s1_rule_prov AS
    WITH raw_rules AS (
        SELECT 
            source1_entity_id,
            CASE WHEN (address_exact_match=1 OR (prefix4_match=1 AND address_first_number_match=1)) THEN 1 ELSE 0 END AS has_b,
            CASE WHEN (first_token_match=1 AND address_first_number_match=1 AND prefix4_match=0 AND address_exact_match=0) THEN 1 ELSE 0 END AS has_c,
            CASE WHEN (name_exact_match=1) THEN 1 ELSE 0 END AS has_d
        FROM read_csv('{FEAT_DIR}/cands_BCD_v1_features_train_s2.tsv', delim='\\t', header=true)
        UNION ALL
        SELECT 
            source1_entity_id,
            CASE WHEN (address_exact_match=1 OR (prefix4_match=1 AND address_first_number_match=1)) THEN 1 ELSE 0 END AS has_b,
            CASE WHEN (first_token_match=1 AND address_first_number_match=1 AND prefix4_match=0 AND address_exact_match=0) THEN 1 ELSE 0 END AS has_c,
            CASE WHEN (name_exact_match=1) THEN 1 ELSE 0 END AS has_d
        FROM read_csv('{FEAT_DIR}/cands_BCD_v1_features_train_s3.tsv', delim='\\t', header=true)
    )
    SELECT 
        source1_entity_id,
        max(has_b) AS rule_b,
        max(has_c) AS rule_c,
        max(has_d) AS rule_d
    FROM raw_rules
    GROUP BY source1_entity_id
""")
print(f"Rule provenance aggregated in {time.time() - t_rules:.2f}s")

# Step 3: Process OOF shards fold by fold to compute S1 level metrics
print("Step 3: Processing OOF shards fold by fold...")
con.execute("""
    CREATE TABLE s1_oof_stats (
        source1_entity_id VARCHAR,
        total_cands BIGINT,
        K BIGINT,
        P BIGINT,
        TP BIGINT,
        top_score FLOAT,
        second_score FLOAT,
        ranking_oracle_f05 FLOAT,
        has_fp_80 BOOLEAN,
        has_tp_50_60 BOOLEAN,
        has_band_40_45 BOOLEAN,
        has_band_45_50 BOOLEAN,
        has_band_50_55 BOOLEAN,
        has_band_55_60 BOOLEAN
    )
""")

for f in range(5):
    t_f = time.time()
    oof_shard = os.path.join(REPORT_DIR, f"E02_OOF_FULL_CANONICAL_fold{f}.tsv")
    print(f"  Processing Fold {f}...")
    
    con.execute(f"""
        CREATE OR REPLACE TABLE oof_f{f} AS
        SELECT source1_entity_id, matched_entity_id, target, oof_score
        FROM read_csv('{oof_shard}', delim='\\t', header=true)
    """)
    
    # 1. Top 2 scores
    con.execute(f"""
        CREATE OR REPLACE TABLE top2_f{f} AS
        WITH ranked_cands AS (
            SELECT 
                source1_entity_id,
                oof_score,
                row_number() OVER (PARTITION BY source1_entity_id ORDER BY oof_score DESC) AS rn
            FROM oof_f{f}
        )
        SELECT 
            source1_entity_id,
            max(CASE WHEN rn = 1 THEN oof_score END) AS top_score,
            max(CASE WHEN rn = 2 THEN oof_score END) AS second_score
        FROM ranked_cands
        WHERE rn <= 2
        GROUP BY source1_entity_id
    """)
    
    # 2. Ranking oracle
    con.execute(f"""
        CREATE OR REPLACE TABLE ro_f{f} AS
        WITH ranked_all AS (
            SELECT 
                source1_entity_id,
                target,
                row_number() OVER (PARTITION BY source1_entity_id ORDER BY oof_score DESC) AS rank
            FROM oof_f{f}
        ),
        ranked_pos AS (
            SELECT 
                source1_entity_id,
                rank AS pos_j,
                row_number() OVER (PARTITION BY source1_entity_id ORDER BY rank ASC) AS j
            FROM ranked_all
            WHERE target = 1
        )
        SELECT 
            r.source1_entity_id,
            max((5.0 * r.j) / (g.A + 4.0 * r.pos_j)) AS ro_f05
        FROM ranked_pos r
        JOIN gt_counts g ON r.source1_entity_id = g.source1_entity_id
        GROUP BY r.source1_entity_id
    """)
    
    # 3. Candidate aggregates
    con.execute(f"""
        CREATE OR REPLACE TABLE agg_f{f} AS
        SELECT 
            source1_entity_id,
            count(*) AS total_cands,
            sum(target) AS K,
            sum(CASE WHEN oof_score >= 0.50 THEN 1 ELSE 0 END) AS P,
            sum(CASE WHEN oof_score >= 0.50 AND target = 1 THEN 1 ELSE 0 END) AS TP,
            max(CASE WHEN target = 0 AND oof_score >= 0.80 THEN 1 ELSE 0 END) = 1 AS has_fp_80,
            max(CASE WHEN target = 1 AND oof_score >= 0.50 AND oof_score < 0.60 THEN 1 ELSE 0 END) = 1 AS has_tp_50_60,
            max(CASE WHEN oof_score >= 0.40 AND oof_score < 0.45 THEN 1 ELSE 0 END) = 1 AS has_band_40_45,
            max(CASE WHEN oof_score >= 0.45 AND oof_score < 0.50 THEN 1 ELSE 0 END) = 1 AS has_band_45_50,
            max(CASE WHEN oof_score >= 0.50 AND oof_score < 0.55 THEN 1 ELSE 0 END) = 1 AS has_band_50_55,
            max(CASE WHEN oof_score >= 0.55 AND oof_score < 0.60 THEN 1 ELSE 0 END) = 1 AS has_band_55_60
        FROM oof_f{f}
        GROUP BY source1_entity_id
    """)
    
    con.execute(f"""
        INSERT INTO s1_oof_stats
        SELECT 
            a.source1_entity_id,
            a.total_cands,
            a.K,
            a.P,
            a.TP,
            t.top_score,
            t.second_score,
            ro.ro_f05,
            a.has_fp_80,
            a.has_tp_50_60,
            a.has_band_40_45,
            a.has_band_45_50,
            a.has_band_50_55,
            a.has_band_55_60
        FROM agg_f{f} a
        LEFT JOIN top2_f{f} t ON a.source1_entity_id = t.source1_entity_id
        LEFT JOIN ro_f{f} ro ON a.source1_entity_id = ro.source1_entity_id
    """)
    
    con.execute(f"DROP TABLE oof_f{f}")
    con.execute(f"DROP TABLE top2_f{f}")
    con.execute(f"DROP TABLE ro_f{f}")
    con.execute(f"DROP TABLE agg_f{f}")
    print(f"  Fold {f} completed in {time.time() - t_f:.2f}s")

# Step 4: Create Master S1 Table
print("Step 4: Assembling Master S1 Entity Table...")
con.execute("""
    CREATE TABLE s1_master AS
    SELECT 
        f.source1_entity_id,
        f.fold,
        -- metadata
        m.missing_name,
        m.missing_addr,
        m.missing_house,
        -- ground truth
        coalesce(g.A, 0) AS A,
        coalesce(g.A_s2, 0) AS A_s2,
        coalesce(g.A_s3, 0) AS A_s3,
        -- candidates & OOF
        coalesce(o.total_cands, 0) AS total_cands,
        coalesce(o.K, 0) AS K,
        coalesce(o.P, 0) AS P,
        coalesce(o.TP, 0) AS TP,
        coalesce(o.P, 0) - coalesce(o.TP, 0) AS FP,
        coalesce(g.A, 0) - coalesce(o.TP, 0) AS FN,
        o.top_score,
        o.second_score,
        CASE 
            WHEN o.total_cands >= 2 THEN o.top_score - o.second_score
            ELSE NULL
        END AS margin,
        -- scores
        CASE 
            WHEN coalesce(g.A, 0) = 0 THEN 
                CASE WHEN coalesce(o.P, 0) = 0 THEN 1.0 ELSE 0.0 END
            ELSE 
                CASE WHEN coalesce(o.TP, 0) = 0 THEN 0.0 
                ELSE (5.0 * o.TP) / (g.A + 4.0 * o.P) END
        END AS f05_actual,
        CASE 
            WHEN coalesce(g.A, 0) = 0 THEN 1.0
            ELSE 
                CASE WHEN coalesce(o.K, 0) = 0 THEN 0.0 
                ELSE (5.0 * o.K) / (g.A + 4.0 * o.K) END
        END AS f05_cand_oracle,
        CASE 
            WHEN coalesce(g.A, 0) = 0 THEN 1.0
            ELSE coalesce(o.ranking_oracle_f05, 0.0)
        END AS f05_ranking_oracle,
        -- booleans
        coalesce(o.has_fp_80, false) AS has_fp_80,
        coalesce(o.has_tp_50_60, false) AS has_tp_50_60,
        (coalesce(g.A, 0) - coalesce(o.TP, 0)) > 0 AS has_fn,
        coalesce(o.has_band_40_45, false) AS has_band_40_45,
        coalesce(o.has_band_45_50, false) AS has_band_45_50,
        coalesce(o.has_band_50_55, false) AS has_band_50_55,
        coalesce(o.has_band_55_60, false) AS has_band_55_60,
        -- rule provenance
        CASE 
            WHEN coalesce(o.total_cands, 0) = 0 THEN 'ZERO_CANDS'
            WHEN coalesce(rp.rule_b, 0) + coalesce(rp.rule_c, 0) + coalesce(rp.rule_d, 0) > 1 THEN 'MULTIPLE_RULES'
            WHEN coalesce(rp.rule_b, 0) = 1 THEN 'B_ONLY'
            WHEN coalesce(rp.rule_c, 0) = 1 THEN 'C_ONLY'
            WHEN coalesce(rp.rule_d, 0) = 1 THEN 'D_ONLY'
            ELSE 'OTHER_RULE'
        END AS rule_provenance
    FROM folds f
    LEFT JOIN s1_meta m ON f.source1_entity_id = m.source1_entity_id
    LEFT JOIN gt_counts g ON f.source1_entity_id = g.source1_entity_id
    LEFT JOIN s1_oof_stats o ON f.source1_entity_id = o.source1_entity_id
    LEFT JOIN s1_rule_prov rp ON f.source1_entity_id = rp.source1_entity_id
""")

print("Master table ready. Verifying baseline...")
base_chk = con.execute("SELECT count(*), avg(f05_actual), avg(f05_cand_oracle), avg(f05_ranking_oracle) FROM s1_master").fetchone()
print(f"Entities: {base_chk[0]}, Actual Macro F0.5: {base_chk[1]:.9f}, Cand Oracle: {base_chk[2]:.9f}, Ranking Oracle: {base_chk[3]:.9f}")

# Step 5: Compute S1 Error Slices
print("Step 5: Computing Error Slices...")

slices_def = [
    # 1. zero candidates
    ("candidate_coverage", "zero_candidates", "total_cands = 0"),
    ("candidate_coverage", "has_candidates", "total_cands > 0"),
    
    # 2-4. captured truth status
    ("truth_capture_status", "true_no_match", "A = 0"),
    ("truth_capture_status", "zero_captured_truth", "A > 0 AND K = 0"),
    ("truth_capture_status", "partial_truth_capture", "A > 0 AND K > 0 AND K < A"),
    ("truth_capture_status", "full_truth_capture", "A > 0 AND K = A"),
    
    # 5. truth size
    ("truth_size", "truth_size_0", "A = 0"),
    ("truth_size", "truth_size_1", "A = 1"),
    ("truth_size", "truth_size_2plus", "A >= 2"),
    
    # 6. candidate-count buckets
    ("candidate_count_bucket", "cands_0", "total_cands = 0"),
    ("candidate_count_bucket", "cands_1", "total_cands = 1"),
    ("candidate_count_bucket", "cands_2_to_5", "total_cands BETWEEN 2 AND 5"),
    ("candidate_count_bucket", "cands_6_to_10", "total_cands BETWEEN 6 AND 10"),
    ("candidate_count_bucket", "cands_11_to_25", "total_cands BETWEEN 11 AND 25"),
    ("candidate_count_bucket", "cands_26_to_50", "total_cands BETWEEN 26 AND 50"),
    ("candidate_count_bucket", "cands_51_to_100", "total_cands BETWEEN 51 AND 100"),
    ("candidate_count_bucket", "cands_100plus", "total_cands > 100"),
    
    # 7. top-score vs second-score margin
    ("score_margin", "margin_very_small_lt_0.05", "total_cands >= 2 AND margin < 0.05"),
    ("score_margin", "margin_small_0.05_0.20", "total_cands >= 2 AND margin >= 0.05 AND margin < 0.20"),
    ("score_margin", "margin_medium_0.20_0.50", "total_cands >= 2 AND margin >= 0.20 AND margin < 0.50"),
    ("score_margin", "margin_large_gte_0.50", "total_cands >= 2 AND margin >= 0.50"),
    ("score_margin", "single_candidate_no_margin", "total_cands = 1"),
    ("score_margin", "zero_candidates_no_margin", "total_cands = 0"),
    
    # 8. high-confidence false positives
    ("error_modes", "has_high_conf_fp_gte_0.80", "has_fp_80 = true"),
    ("error_modes", "no_high_conf_fp", "has_fp_80 = false"),
    
    # 9. low-confidence true positives
    ("error_modes", "has_low_conf_tp_0.50_0.60", "has_tp_50_60 = true"),
    ("error_modes", "no_low_conf_tp", "has_tp_50_60 = false"),
    
    # 10. false negatives
    ("error_modes", "has_false_negatives", "has_fn = true"),
    ("error_modes", "zero_false_negatives", "has_fn = false"),
    
    # 11-12. S2 vs S3 composition
    ("source_composition", "s2_only_truth", "A_s2 > 0 AND A_s3 = 0"),
    ("source_composition", "s3_only_truth", "A_s2 = 0 AND A_s3 > 0"),
    ("source_composition", "both_s2_and_s3_truth", "A_s2 > 0 AND A_s3 > 0"),
    ("source_composition", "neither_s2_nor_s3_truth", "A = 0"),
    
    # 13. rule provenance
    ("rule_provenance", "rule_b_only", "rule_provenance = 'B_ONLY'"),
    ("rule_provenance", "rule_c_only", "rule_provenance = 'C_ONLY'"),
    ("rule_provenance", "rule_d_only", "rule_provenance = 'D_ONLY'"),
    ("rule_provenance", "multiple_rules", "rule_provenance = 'MULTIPLE_RULES'"),
    ("rule_provenance", "zero_candidates", "rule_provenance = 'ZERO_CANDS'"),
    
    # 14. missing fields
    ("missing_fields", "missing_business_name", "missing_name = true"),
    ("missing_fields", "missing_business_address", "missing_addr = true"),
    ("missing_fields", "missing_house_number", "missing_house = true"),
    ("missing_fields", "all_fields_present", "missing_name = false AND missing_addr = false AND missing_house = false"),
    
    # 15. score bands around threshold 0.50
    ("threshold_neighborhood", "has_candidate_in_0.40_0.45", "has_band_40_45 = true"),
    ("threshold_neighborhood", "has_candidate_in_0.45_0.50", "has_band_45_50 = true"),
    ("threshold_neighborhood", "has_candidate_in_0.50_0.55", "has_band_50_55 = true"),
    ("threshold_neighborhood", "has_candidate_in_0.55_0.60", "has_band_55_60 = true"),
]

slice_results = []
TOTAL_POP = 2206821
TOTAL_LOSS = 1.0 - base_chk[1]

for cat, name, cond in slices_def:
    res = con.execute(f"""
        SELECT 
            count(*),
            avg(f05_actual),
            avg(f05_cand_oracle),
            avg(f05_ranking_oracle),
            sum(1.0 - f05_actual) / {TOTAL_POP} AS loss_impact,
            avg(A),
            avg(P),
            avg(TP),
            avg(FP),
            avg(FN)
        FROM s1_master
        WHERE {cond}
    """).fetchone()
    
    cnt, act_f05, cand_f05, rank_f05, loss_imp, mA, mP, mTP, mFP, mFN = res
    pct_pop = cnt * 100.0 / TOTAL_POP
    pct_loss = (loss_imp / TOTAL_LOSS) * 100.0 if TOTAL_LOSS > 0 else 0.0
    
    slice_results.append({
        "slice_category": cat,
        "slice_name": name,
        "condition": cond,
        "entity_count": cnt,
        "pct_of_all_s1": pct_pop,
        "actual_e02_macro_f05": act_f05 if cnt > 0 else 0.0,
        "cand_oracle_macro_f05": cand_f05 if cnt > 0 else 0.0,
        "ranking_oracle_macro_f05": rank_f05 if cnt > 0 else 0.0,
        "macro_f05_loss_impact": loss_imp if cnt > 0 else 0.0,
        "pct_of_total_loss": pct_loss if cnt > 0 else 0.0,
        "mean_actual_matches": mA if cnt > 0 else 0.0,
        "mean_predicted_matches": mP if cnt > 0 else 0.0,
        "mean_true_positives": mTP if cnt > 0 else 0.0,
        "mean_false_positives": mFP if cnt > 0 else 0.0,
        "mean_false_negatives": mFN if cnt > 0 else 0.0
    })

df_slices = pd.DataFrame(slice_results)
df_slices.to_csv(OUT_SLICES, sep='\\t', index=False)
print(f"Saved error slices to {OUT_SLICES}")
print(f"Total script runtime: {time.time() - t0:.2f}s")
con.close()
