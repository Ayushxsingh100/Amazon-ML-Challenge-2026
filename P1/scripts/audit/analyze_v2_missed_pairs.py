#!/usr/bin/env python3
"""
Phase 1 — Forensic Analysis of V2 Missed Pairs
Amazon ML Challenge 2026 — Person 1 (P1)

Constructs missed_s2.tsv and missed_s3.tsv, verifies them against Ground Truth,
and performs forensic lexical, address, country, script, and error-bucket analyses.
"""

import os
import sys
import time
import json
import duckdb
import unicodedata

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
EXP_DIR = os.path.join(REPO_ROOT, "P1", "experiments", "phase1", "v2_missed_pairs")
REPORT_DIR = os.path.join(REPO_ROOT, "P1", "reports")

os.makedirs(EXP_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

def run_analysis():
    con = duckdb.connect()
    con.execute("PRAGMA threads=4;")
    con.execute("PRAGMA memory_limit='8GB';")

    print("\n========================================================")
    print("1. Extracting Ground Truth and Building Missed Pairs")
    print("========================================================")

    # 1. Load Ground Truth exploded
    t0 = time.time()
    con.execute("""
        CREATE TEMP TABLE gt_exploded AS
        SELECT 
            source1_entity_id,
            TRIM(UNNEST(STRING_SPLIT(matched_entity_ids, ','))) AS matched_entity_id
        FROM read_csv('data/train/train_ground_truth.tsv', sep='\t', header=true)
        WHERE matched_entity_ids IS NOT NULL AND TRIM(matched_entity_ids) <> '';
    """)
    print(f"Ground truth loaded in {time.time() - t0:.2f}s")

    # 2. Extract Missed S2 Pairs
    print("\nExtracting missed S2 pairs...")
    t0 = time.time()
    s2_missed_tsv = os.path.join(EXP_DIR, "missed_s2.tsv")
    con.execute(f"""
        CREATE TEMP TABLE missed_s2 AS
        SELECT 
            gt.source1_entity_id,
            gt.matched_entity_id AS target_entity_id,
            'source2' AS target_source,
            s1.country_raw AS country_s1,
            s2.country_raw AS country_target,
            s1.business_name_raw AS name_s1_raw,
            s2.business_name_raw AS name_target_raw,
            s1.business_address_raw AS address_s1_raw,
            s2.business_address_raw AS address_target_raw,
            s1.business_name_normalized AS name_s1_normalized,
            s2.business_name_normalized AS name_target_normalized,
            s1.business_address_normalized AS address_s1_normalized,
            s2.business_address_normalized AS address_target_normalized,
            s1.name_clean AS name_s1_clean,
            s2.name_clean AS name_target_clean,
            s1.address_clean AS address_s1_clean,
            s2.address_clean AS address_target_clean,
            s1.name_prefix_4 AS prefix4_s1,
            s2.name_prefix_4 AS prefix4_target,
            s1.name_prefix_3 AS prefix3_s1,
            s2.name_prefix_3 AS prefix3_target,
            s1.first_token AS token1_s1,
            s2.first_token AS token1_target,
            s1.root_token AS root_token_s1,
            s2.root_token AS root_token_target,
            s1.house_number AS house_s1,
            s2.house_number AS house_target,
            s1.house_number_norm AS house_norm_s1,
            s2.house_number_norm AS house_norm_target,
            s1.postal_code AS zip_s1,
            s2.postal_code AS zip_target
        FROM gt_exploded gt
        ANTI JOIN (
            SELECT source1_entity_id, matched_entity_id 
            FROM read_csv('P2/data/candidates/train_candidate_pairs_s2_v2.tsv', sep='\t', header=true)
        ) c ON gt.source1_entity_id = c.source1_entity_id AND gt.matched_entity_id = c.matched_entity_id
        JOIN read_parquet('P1/data/entities/train/source1/train_s1_entities.parquet') s1
          ON gt.source1_entity_id = s1.entity_id
        JOIN read_parquet('P1/data/entities/train/source2/train_s2_entities.parquet') s2
          ON gt.matched_entity_id = s2.entity_id
        WHERE gt.matched_entity_id LIKE 'S2-%';
    """)

    # Export missed_s2.tsv (only the required 13 columns)
    con.execute(f"""
        COPY (
            SELECT 
                source1_entity_id,
                target_entity_id,
                target_source,
                country_s1,
                country_target,
                name_s1_raw,
                name_target_raw,
                address_s1_raw,
                address_target_raw,
                name_s1_normalized,
                name_target_normalized,
                address_s1_normalized,
                address_target_normalized
            FROM missed_s2
            ORDER BY source1_entity_id ASC, target_entity_id ASC
        ) TO '{s2_missed_tsv}' (FORMAT CSV, DELIMITER '\t', HEADER TRUE);
    """)
    s2_cnt = con.execute("SELECT COUNT(*) FROM missed_s2").fetchone()[0]
    print(f"Created {s2_missed_tsv}: {s2_cnt:,} rows in {time.time() - t0:.2f}s")

    # 3. Extract Missed S3 Pairs
    print("\nExtracting missed S3 pairs...")
    t0 = time.time()
    s3_missed_tsv = os.path.join(EXP_DIR, "missed_s3.tsv")
    con.execute(f"""
        CREATE TEMP TABLE missed_s3 AS
        SELECT 
            gt.source1_entity_id,
            gt.matched_entity_id AS target_entity_id,
            'source3' AS target_source,
            s1.country_raw AS country_s1,
            s3.country_raw AS country_target,
            s1.business_name_raw AS name_s1_raw,
            s3.business_name_raw AS name_target_raw,
            s1.business_address_raw AS address_s1_raw,
            s3.business_address_raw AS address_target_raw,
            s1.business_name_normalized AS name_s1_normalized,
            s3.business_name_normalized AS name_target_normalized,
            s1.business_address_normalized AS address_s1_normalized,
            s3.business_address_normalized AS address_target_normalized,
            s1.name_clean AS name_s1_clean,
            s3.name_clean AS name_target_clean,
            s1.address_clean AS address_s1_clean,
            s3.address_clean AS address_target_clean,
            s1.name_prefix_4 AS prefix4_s1,
            s3.name_prefix_4 AS prefix4_target,
            s1.name_prefix_3 AS prefix3_s1,
            s3.name_prefix_3 AS prefix3_target,
            s1.first_token AS token1_s1,
            s3.first_token AS token1_target,
            s1.root_token AS root_token_s1,
            s3.root_token AS root_token_target,
            s1.house_number AS house_s1,
            s3.house_number AS house_target,
            s1.house_number_norm AS house_norm_s1,
            s3.house_number_norm AS house_norm_target,
            s1.postal_code AS zip_s1,
            s3.postal_code AS zip_target
        FROM gt_exploded gt
        ANTI JOIN (
            SELECT source1_entity_id, matched_entity_id 
            FROM read_csv('P2/data/candidates/train_candidate_pairs_s3_v2.tsv', sep='\t', header=true)
        ) c ON gt.source1_entity_id = c.source1_entity_id AND gt.matched_entity_id = c.matched_entity_id
        JOIN read_parquet('P1/data/entities/train/source1/train_s1_entities.parquet') s1
          ON gt.source1_entity_id = s1.entity_id
        JOIN read_parquet('P1/data/entities/train/source3/train_s3_entities.parquet') s3
          ON gt.matched_entity_id = s3.entity_id
        WHERE gt.matched_entity_id LIKE 'S3-%';
    """)

    con.execute(f"""
        COPY (
            SELECT 
                source1_entity_id,
                target_entity_id,
                target_source,
                country_s1,
                country_target,
                name_s1_raw,
                name_target_raw,
                address_s1_raw,
                address_target_raw,
                name_s1_normalized,
                name_target_normalized,
                address_s1_normalized,
                address_target_normalized
            FROM missed_s3
            ORDER BY source1_entity_id ASC, target_entity_id ASC
        ) TO '{s3_missed_tsv}' (FORMAT CSV, DELIMITER '\t', HEADER TRUE);
    """)
    s3_cnt = con.execute("SELECT COUNT(*) FROM missed_s3").fetchone()[0]
    print(f"Created {s3_missed_tsv}: {s3_cnt:,} rows in {time.time() - t0:.2f}s")

    print(f"\nTotal V2 Missed Pairs: {s2_cnt + s3_cnt:,} (S2: {s2_cnt:,}, S3: {s3_cnt:,})")

    # 4. Integrity Verification of Missed Pairs
    print("\n========================================================")
    print("2. Verifying Missed-Pair Integrity")
    print("========================================================")
    # Check duplicates in missed pairs
    s2_dups = con.execute("SELECT COUNT(*) - COUNT(DISTINCT source1_entity_id || '-' || target_entity_id) FROM missed_s2").fetchone()[0]
    s3_dups = con.execute("SELECT COUNT(*) - COUNT(DISTINCT source1_entity_id || '-' || target_entity_id) FROM missed_s3").fetchone()[0]
    print(f"Duplicate pairs in missed_s2: {s2_dups}")
    print(f"Duplicate pairs in missed_s3: {s3_dups}")
    assert s2_dups == 0 and s3_dups == 0, "Duplicate missed pairs detected!"

    # Combine into union table for comprehensive analysis
    con.execute("""
        CREATE TEMP TABLE missed_all AS
        SELECT * FROM missed_s2
        UNION ALL
        SELECT * FROM missed_s3;
    """)

    # 5. Country Breakdown Analysis
    print("\n========================================================")
    print("3. Analyzing Country Breakdown")
    print("========================================================")
    country_q = """
        SELECT 
            target_source,
            coalesce(country_target, 'UNKNOWN') AS country,
            COUNT(*) AS missed_count,
            ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (PARTITION BY target_source), 4) AS pct_of_target_source,
            ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM missed_all), 4) AS pct_of_total_missed
        FROM missed_all
        GROUP BY target_source, country_target
        ORDER BY target_source, missed_count DESC;
    """
    country_rows = con.execute(country_q).fetchall()
    country_tsv = os.path.join(REPORT_DIR, "V2_MISSED_PAIR_COUNTRY_BREAKDOWN.tsv")
    with open(country_tsv, "w") as f:
        f.write("target_source\tcountry\tmissed_count\tpct_of_target_source\tpct_of_total_missed\n")
        for r in country_rows:
            f.write(f"{r[0]}\t{r[1]}\t{r[2]}\t{r[3]}%\t{r[4]}%\n")
    print(f"Written {country_tsv}")

    # Check country mismatch count
    country_mismatches = con.execute("""
        SELECT COUNT(*) 
        FROM missed_all 
        WHERE country_s1 <> country_target
    """).fetchone()[0]
    print(f"Country mismatch between S1 and target in GT: {country_mismatches:,} pairs")

    # 6. Script / Character Set Analysis
    print("\n========================================================")
    print("4. Analyzing Unicode & Script Composition")
    print("========================================================")
    # Define script detection logic in DuckDB SQL:
    # Devanagari range: \u0900-\u097F
    # Non-ASCII: any char with ASCII code > 127
    script_q = """
        SELECT 
            target_source,
            COUNT(*) AS total_missed,
            SUM(CASE WHEN regexp_matches(name_s1_raw || name_target_raw, '[\u0900-\u097f]') THEN 1 ELSE 0 END) AS devanagari_count,
            SUM(CASE WHEN NOT regexp_matches(name_s1_raw || name_target_raw, '[^\x01-\x7f]') THEN 1 ELSE 0 END) AS ascii_only_count,
            SUM(CASE WHEN regexp_matches(name_s1_raw || name_target_raw, '[^\x01-\x7f]') AND NOT regexp_matches(name_s1_raw || name_target_raw, '[\u0900-\u097f]') THEN 1 ELSE 0 END) AS other_non_ascii_count,
            SUM(CASE WHEN (regexp_matches(name_s1_raw, '[^\x01-\x7f]') AND NOT regexp_matches(name_target_raw, '[^\x01-\x7f]')) 
                      OR (NOT regexp_matches(name_s1_raw, '[^\x01-\x7f]') AND regexp_matches(name_target_raw, '[^\x01-\x7f]')) THEN 1 ELSE 0 END) AS cross_script_transliteration_candidate
        FROM missed_all
        GROUP BY target_source;
    """
    script_res = con.execute(script_q).fetchall()
    script_tsv = os.path.join(REPORT_DIR, "V2_MISSED_PAIR_SCRIPT_ANALYSIS.tsv")
    with open(script_tsv, "w") as f:
        f.write("target_source\ttotal_missed\tascii_only\tpct_ascii\tdevanagari\tpct_devanagari\tother_non_ascii\tpct_other_non_ascii\tcross_script_candidates\tpct_cross_script\n")
        for r in script_res:
            tot = r[1]
            dev = r[2]
            asc = r[3]
            oth = r[4]
            crs = r[5]
            f.write(f"{r[0]}\t{tot}\t{asc}\t{asc*100.0/tot:.2f}%\t{dev}\t{dev*100.0/tot:.2f}%\t{oth}\t{oth*100.0/tot:.2f}%\t{crs}\t{crs*100.0/tot:.2f}%\n")
    print(f"Written {script_tsv}")

    # 7. House Number & Address Analysis
    print("\n========================================================")
    print("5. Analyzing House Number & Address Patterns")
    print("========================================================")
    house_q = """
        SELECT 
            target_source,
            COUNT(*) AS total_missed,
            SUM(CASE WHEN house_s1 <> '' AND house_target <> '' AND house_s1 = house_target THEN 1 ELSE 0 END) AS house_equal,
            SUM(CASE WHEN house_s1 <> '' AND house_target <> '' AND house_s1 <> house_target THEN 1 ELSE 0 END) AS house_different,
            SUM(CASE WHEN (house_s1 = '' AND house_target <> '') OR (house_s1 <> '' AND house_target = '') THEN 1 ELSE 0 END) AS house_missing_one_side,
            SUM(CASE WHEN house_s1 = '' AND house_target = '' THEN 1 ELSE 0 END) AS house_missing_both_sides,
            SUM(CASE WHEN house_norm_s1 <> '' AND house_norm_target <> '' AND house_norm_s1 = house_norm_target THEN 1 ELSE 0 END) AS house_norm_equal,
            SUM(CASE WHEN zip_s1 <> '' AND zip_target <> '' AND zip_s1 = zip_target THEN 1 ELSE 0 END) AS zip_equal
        FROM missed_all
        GROUP BY target_source;
    """
    house_res = con.execute(house_q).fetchall()
    print("House number analysis results:")
    for r in house_res:
        print(f"  {r[0]}: total={r[1]:,}, equal={r[2]:,}, diff={r[3]:,}, missing_one={r[4]:,}, missing_both={r[5]:,}, norm_equal={r[6]:,}, zip_equal={r[7]:,}")

    # 8. Name Lexical Variation & Distribution Analysis
    print("\n========================================================")
    print("6. Analyzing Name Lexical Variation Distributions")
    print("========================================================")
    # Add computed metric columns
    con.execute("""
        CREATE TEMP TABLE missed_metrics AS
        SELECT 
            target_source,
            jaro_winkler_similarity(name_s1_normalized, name_target_normalized) AS jw_name,
            jaro_winkler_similarity(name_s1_clean, name_target_clean) AS jw_clean_name,
            abs(length(name_s1_normalized) - length(name_target_normalized)) AS len_diff_name,
            abs(length(address_s1_normalized) - length(address_target_normalized)) AS len_diff_addr,
            CASE WHEN name_s1_raw = name_target_raw THEN 1 ELSE 0 END AS raw_name_exact,
            CASE WHEN name_s1_normalized = name_target_normalized THEN 1 ELSE 0 END AS norm_name_exact,
            CASE WHEN name_s1_clean = name_target_clean AND name_s1_clean <> '' THEN 1 ELSE 0 END AS clean_name_exact,
            CASE WHEN address_s1_raw = address_target_raw THEN 1 ELSE 0 END AS raw_addr_exact,
            CASE WHEN address_s1_normalized = address_target_normalized THEN 1 ELSE 0 END AS norm_addr_exact,
            CASE WHEN address_s1_clean = address_target_clean AND address_s1_clean <> '' THEN 1 ELSE 0 END AS clean_addr_exact,
            CASE WHEN prefix4_s1 = prefix4_target AND prefix4_s1 <> '' THEN 1 ELSE 0 END AS prefix4_match,
            CASE WHEN prefix3_s1 = prefix3_target AND prefix3_s1 <> '' THEN 1 ELSE 0 END AS prefix3_match,
            CASE WHEN token1_s1 = token1_target AND token1_s1 <> '' THEN 1 ELSE 0 END AS token1_match,
            CASE WHEN root_token_s1 = root_token_target AND root_token_s1 <> '' THEN 1 ELSE 0 END AS root_token_match,
            CASE WHEN house_s1 = house_target AND house_s1 <> '' THEN 1 ELSE 0 END AS house_match,
            CASE WHEN house_norm_s1 = house_norm_target AND house_norm_s1 <> '' THEN 1 ELSE 0 END AS house_norm_match,
            CASE WHEN zip_s1 = zip_target AND zip_s1 <> '' THEN 1 ELSE 0 END AS zip_match,
            CASE WHEN country_s1 = country_target AND country_s1 <> '' THEN 1 ELSE 0 END AS country_match
        FROM missed_all;
    """)

    # Compute percentiles for Jaro-Winkler
    jw_percentiles = con.execute("""
        SELECT 
            target_source,
            MIN(jw_name) AS min_jw,
            QUANTILE_CONT(jw_name, 0.01) AS p01,
            QUANTILE_CONT(jw_name, 0.05) AS p05,
            QUANTILE_CONT(jw_name, 0.10) AS p10,
            QUANTILE_CONT(jw_name, 0.25) AS p25,
            QUANTILE_CONT(jw_name, 0.50) AS median_jw,
            QUANTILE_CONT(jw_name, 0.75) AS p75,
            QUANTILE_CONT(jw_name, 0.90) AS p90,
            QUANTILE_CONT(jw_name, 0.95) AS p95,
            QUANTILE_CONT(jw_name, 0.99) AS p99,
            MAX(jw_name) AS max_jw
        FROM missed_metrics
        GROUP BY target_source;
    """).fetchall()
    print("\nJaro-Winkler Similarity Percentiles on Missed Pairs:")
    for p in jw_percentiles:
        print(f"  {p[0]}: min={p[1]:.4f}, p05={p[3]:.4f}, p25={p[5]:.4f}, median={p[6]:.4f}, p75={p[7]:.4f}, p95={p[9]:.4f}, max={p[11]:.4f}")

    # 9. V2 Rule Failure Analysis
    print("\n========================================================")
    print("7. Analyzing Why V2 Rules Failed on Missed Pairs")
    print("========================================================")
    rule_diag_q = """
        SELECT 
            target_source,
            COUNT(*) AS total_missed,
            SUM(CASE WHEN country_match = 0 THEN 1 ELSE 0 END) AS country_mismatch_or_empty,
            -- Why Rule A failed (country match AND prefix4 match AND house match)
            SUM(CASE WHEN country_match = 1 AND prefix4_match = 1 AND house_match = 0 THEN 1 ELSE 0 END) AS rule_a_prefix4_ok_house_failed,
            SUM(CASE WHEN country_match = 1 AND prefix4_match = 0 AND house_match = 1 THEN 1 ELSE 0 END) AS rule_a_house_ok_prefix4_failed,
            -- Why Rule B/H failed (exact/clean address match)
            SUM(CASE WHEN country_match = 1 AND clean_addr_exact = 1 THEN 1 ELSE 0 END) AS clean_addr_matched,
            -- Why Rule C failed (token1 match AND house match)
            SUM(CASE WHEN country_match = 1 AND token1_match = 1 AND house_match = 0 THEN 1 ELSE 0 END) AS rule_c_token1_ok_house_failed,
            SUM(CASE WHEN country_match = 1 AND token1_match = 0 AND house_match = 1 THEN 1 ELSE 0 END) AS rule_c_house_ok_token1_failed,
            -- Why Rule D/E failed (exact/clean name match)
            SUM(CASE WHEN country_match = 1 AND clean_name_exact = 1 THEN 1 ELSE 0 END) AS clean_name_matched,
            -- Why Rule G failed (root_token match AND house_norm match)
            SUM(CASE WHEN country_match = 1 AND root_token_match = 1 AND house_norm_match = 0 THEN 1 ELSE 0 END) AS rule_g_root_ok_house_norm_failed,
            SUM(CASE WHEN country_match = 1 AND root_token_match = 0 AND house_norm_match = 1 THEN 1 ELSE 0 END) AS rule_g_house_norm_ok_root_failed,
            -- Why Rule I failed (zip match AND prefix3 match)
            SUM(CASE WHEN country_match = 1 AND zip_match = 1 AND prefix3_match = 0 THEN 1 ELSE 0 END) AS rule_i_zip_ok_prefix3_failed,
            SUM(CASE WHEN country_match = 1 AND zip_match = 0 AND prefix3_match = 1 THEN 1 ELSE 0 END) AS rule_i_prefix3_ok_zip_failed
        FROM missed_metrics
        GROUP BY target_source;
    """
    rule_res = con.execute(rule_diag_q).fetchall()
    for r in rule_res:
        print(f"Rule diagnosis for {r[0]}:")
        print(f"  Country mismatch: {r[2]:,}")
        print(f"  Prefix4 match but house failed: {r[3]:,}")
        print(f"  House match but prefix4 failed: {r[4]:,}")
        print(f"  House match but token1 failed: {r[6]:,}")
        print(f"  House norm match but root failed: {r[8]:,}")
        print(f"  Zip match but prefix3 failed: {r[9]:,}")

    # 10. Build Error Buckets
    print("\n========================================================")
    print("8. Categorizing Error Buckets")
    print("========================================================")
    # Define mutually exhaustive / deterministic hierarchy of failure causes:
    # 1. cross_country_mismatch: country differs between S1 and target
    # 2. devanagari_indic_script: Indic / non-ASCII name variation where Latin blocking fails
    # 3. house_number_missing_both: Neither S1 nor target contains an extractable house number
    # 4. house_number_missing_one: Exactly one entity has house number
    # 5. address_match_but_name_severe: House number matches exactly, but name has severe variation (JW < 0.60)
    # 6. address_match_name_moderate: House number matches, name JW between 0.60 and 0.85
    # 7. zip_match_name_severe: 5-6 digit zip code matches, but name has no prefix3 match
    # 8. name_similar_address_diff: Name JW >= 0.85, but address has different house number or no house number
    # 9. multi_token_permutation: Token overlap exists, but token 1/prefix differs
    # 10. other_severe_divergence: Both name and address differ substantially
    bucket_q = """
        SELECT 
            target_source,
            CASE 
                WHEN country_match = 0 THEN 'country_mismatch'
                WHEN regexp_matches(name_s1_raw || name_target_raw, '[\u0900-\u097F]') THEN 'indic_devanagari_script_failure'
                WHEN house_s1 = '' AND house_target = '' THEN 'missing_house_number_both_sides'
                WHEN (house_s1 = '' AND house_target <> '') OR (house_s1 <> '' AND house_target = '') THEN 'missing_house_number_one_side'
                WHEN house_norm_match = 1 AND jw_name < 0.60 THEN 'house_match_severe_name_variation'
                WHEN house_norm_match = 1 AND jw_name >= 0.60 AND jw_name < 0.85 THEN 'house_match_moderate_name_variation'
                WHEN zip_match = 1 AND prefix3_match = 0 THEN 'zip_match_prefix_mismatch'
                WHEN jw_name >= 0.85 AND house_norm_match = 0 THEN 'high_name_similarity_house_divergence'
                ELSE 'unstructured_divergence'
            END AS error_bucket,
            COUNT(*) AS pair_count,
            ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (PARTITION BY target_source), 4) AS pct_of_missed
        FROM missed_all ma
        JOIN missed_metrics mm ON ma.source1_entity_id = mm.source1_entity_id AND ma.target_entity_id = mm.target_entity_id
        GROUP BY target_source, error_bucket
        ORDER BY target_source, pair_count DESC;
    """
    # Wait, in the join above, source1_entity_id and target_entity_id might not be in missed_metrics!
    # Let's fix that by selecting the entity IDs in missed_metrics:
    con.execute("""
        CREATE OR REPLACE TEMP TABLE missed_metrics AS
        SELECT 
            source1_entity_id,
            target_entity_id,
            target_source,
            name_s1_raw,
            name_target_raw,
            jaro_winkler_similarity(name_s1_normalized, name_target_normalized) AS jw_name,
            jaro_winkler_similarity(name_s1_clean, name_target_clean) AS jw_clean_name,
            abs(length(name_s1_normalized) - length(name_target_normalized)) AS len_diff_name,
            abs(length(address_s1_normalized) - length(address_target_normalized)) AS len_diff_addr,
            CASE WHEN name_s1_raw = name_target_raw THEN 1 ELSE 0 END AS raw_name_exact,
            CASE WHEN name_s1_normalized = name_target_normalized THEN 1 ELSE 0 END AS norm_name_exact,
            CASE WHEN name_s1_clean = name_target_clean AND name_s1_clean <> '' THEN 1 ELSE 0 END AS clean_name_exact,
            CASE WHEN address_s1_raw = address_target_raw THEN 1 ELSE 0 END AS raw_addr_exact,
            CASE WHEN address_s1_normalized = address_target_normalized THEN 1 ELSE 0 END AS norm_addr_exact,
            CASE WHEN address_s1_clean = address_target_clean AND address_s1_clean <> '' THEN 1 ELSE 0 END AS clean_addr_exact,
            CASE WHEN prefix4_s1 = prefix4_target AND prefix4_s1 <> '' THEN 1 ELSE 0 END AS prefix4_match,
            CASE WHEN prefix3_s1 = prefix3_target AND prefix3_s1 <> '' THEN 1 ELSE 0 END AS prefix3_match,
            CASE WHEN token1_s1 = token1_target AND token1_s1 <> '' THEN 1 ELSE 0 END AS token1_match,
            CASE WHEN root_token_s1 = root_token_target AND root_token_s1 <> '' THEN 1 ELSE 0 END AS root_token_match,
            CASE WHEN house_s1 = house_target AND house_s1 <> '' THEN 1 ELSE 0 END AS house_match,
            CASE WHEN house_norm_s1 = house_norm_target AND house_norm_s1 <> '' THEN 1 ELSE 0 END AS house_norm_match,
            CASE WHEN house_s1 = '' AND house_target = '' THEN 1 ELSE 0 END AS house_empty_both,
            CASE WHEN (house_s1 = '' AND house_target <> '') OR (house_s1 <> '' AND house_target = '') THEN 1 ELSE 0 END AS house_empty_one,
            CASE WHEN zip_s1 = zip_target AND zip_s1 <> '' THEN 1 ELSE 0 END AS zip_match,
            CASE WHEN country_s1 = country_target AND country_s1 <> '' THEN 1 ELSE 0 END AS country_match
        FROM missed_all;
    """)

    bucket_q = """
        SELECT 
            target_source,
            CASE 
                WHEN country_match = 0 THEN 'country_mismatch'
                WHEN regexp_matches(name_s1_raw || name_target_raw, '[\u0900-\u097F]') THEN 'indic_devanagari_script'
                WHEN house_norm_match = 1 AND jw_name < 0.60 THEN 'house_match_severe_name_variation'
                WHEN house_norm_match = 1 AND jw_name >= 0.60 AND jw_name < 0.85 THEN 'house_match_moderate_name_variation'
                WHEN house_empty_both = 1 THEN 'missing_house_number_both_sides'
                WHEN house_empty_one = 1 THEN 'missing_house_number_one_side'
                WHEN zip_match = 1 AND prefix3_match = 0 THEN 'zip_match_prefix_mismatch'
                WHEN jw_name >= 0.85 AND house_norm_match = 0 THEN 'high_name_similarity_house_divergence'
                ELSE 'unstructured_divergence'
            END AS error_bucket,
            COUNT(*) AS pair_count,
            ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (PARTITION BY target_source), 4) AS pct_of_missed
        FROM missed_metrics
        GROUP BY target_source, error_bucket
        ORDER BY target_source, pair_count DESC;
    """
    bucket_rows = con.execute(bucket_q).fetchall()
    bucket_tsv = os.path.join(REPORT_DIR, "V2_MISSED_PAIR_ERROR_BUCKETS.tsv")
    with open(bucket_tsv, "w") as f:
        f.write("target_source\terror_bucket\tpair_count\tpct_of_missed\tdefinition\n")
        definitions = {
            "country_mismatch": "country_s1 != country_target in ground truth",
            "indic_devanagari_script": "Non-Latin Devanagari script present in S1 or target business name",
            "house_match_severe_name_variation": "Normalized house number matches exactly, but business name Jaro-Winkler < 0.60",
            "house_match_moderate_name_variation": "Normalized house number matches exactly, business name Jaro-Winkler between 0.60 and 0.85",
            "missing_house_number_both_sides": "Neither S1 nor target has an extractable house number in address",
            "missing_house_number_one_side": "Exactly one entity lacks an extractable house number",
            "zip_match_prefix_mismatch": "5-to-6 digit postal code matches, but 3-character name prefix differs",
            "high_name_similarity_house_divergence": "Business name Jaro-Winkler >= 0.85, but house number differs or is missing",
            "unstructured_divergence": "Compound variation across both name and address tokens"
        }
        for r in bucket_rows:
            defn = definitions.get(r[1], "N/A")
            f.write(f"{r[0]}\t{r[1]}\t{r[2]}\t{r[3]}%\t{defn}\n")
    print(f"Written {bucket_tsv}")

    # Summary JSON
    summary_data = {
        "total_v2_missed": s2_cnt + s3_cnt,
        "missed_s2": s2_cnt,
        "missed_s3": s3_cnt,
        "country_breakdown": [{"target_source": r[0], "country": r[1], "count": r[2], "pct_target": r[3], "pct_total": r[4]} for r in country_rows],
        "script_analysis": [{"target_source": r[0], "total": r[1], "ascii": r[2], "devanagari": r[3], "other_non_ascii": r[4], "cross_script": r[5]} for r in script_res],
        "error_buckets": [{"target_source": r[0], "bucket": r[1], "count": r[2], "pct": r[3]} for r in bucket_rows],
        "jw_percentiles": [{"target_source": p[0], "min": p[1], "p05": p[3], "p25": p[5], "median": p[6], "p75": p[7], "p95": p[9], "max": p[11]} for p in jw_percentiles]
    }
    with open(os.path.join(REPORT_DIR, "v2_missed_pairs_stats.json"), "w") as f:
        json.dump(summary_data, f, indent=2)

    print("\nMissed pair forensic analysis complete!")

if __name__ == "__main__":
    run_analysis()
