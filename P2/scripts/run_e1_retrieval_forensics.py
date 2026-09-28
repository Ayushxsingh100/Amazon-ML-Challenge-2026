#!/usr/bin/env python3
"""
E1 Retrieval-Improvement Forensic & Simulation Pipeline
======================================================
Amazon ML Challenge 2026 - AG-P2 Experiment E1

Investigates the 2,126,379 missed true pairs from V3, analyzes the 130,284 zero-capture
S1 entities, simulates label-free candidate retrieval channels under strict train/validation split
(design on folds 0-3, validation on held-out fold 4), audits structural and raw-field signals,
verifies test volume parity, and simulates cumulative oracle gains.

Outputs:
- P2/reports/E1_residual_miss_forensics.json
- P2/reports/E1_retrieval_channel_results.tsv
- P2/reports/E1_structural_signal_audit.json
- P2/reports/E1_cumulative_simulation.tsv
"""

import os
import sys
import io
import time
import json
import re
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

# File paths
FOLDS_PATH = REPO / "P3" / "reports" / "folds_v1_manifest.tsv"
GT_PATH = REPO / "outputs" / "person1_step1" / "train_ground_truth_reconstructed.tsv"
S1_NORM_PATH = REPO / "outputs" / "person1_step1" / "normalized" / "train_source1_normalized.tsv"
S2_PARQ = REPO / "P1" / "data" / "entities" / "train" / "source2" / "train_s2_entities.parquet"
S3_PARQ = REPO / "P1" / "data" / "entities" / "train" / "source3" / "train_s3_entities.parquet"

TEST_S1_NORM = REPO / "outputs" / "person1_step1" / "normalized" / "test_source1_normalized.tsv"
TEST_S2_PARQ = REPO / "P1" / "data" / "entities" / "test" / "source2" / "test_s2_entities.parquet"
TEST_S3_PARQ = REPO / "P1" / "data" / "entities" / "test" / "source3" / "test_s3_entities.parquet"

V3_OOF_GLOB = "E:/predictions/phase4/v3_train_oof_fold*.parquet"

OUT_DIR = REPO / "P2" / "reports"
OUT_RESIDUAL_JSON = OUT_DIR / "E1_residual_miss_forensics.json"
OUT_CHANNELS_TSV = OUT_DIR / "E1_retrieval_channel_results.tsv"
OUT_STRUCTURAL_JSON = OUT_DIR / "E1_structural_signal_audit.json"
OUT_CUMULATIVE_TSV = OUT_DIR / "E1_cumulative_simulation.tsv"

def main():
    t_global_start = time.time()
    print("=" * 80)
    print("E1 RETRIEVAL IMPROVEMENT FORENSIC & SIMULATION")
    print("=" * 80)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()
    con.execute("SET threads=8")
    con.execute("SET memory_limit='8GB'")

    # =========================================================================
    # STEP 0: LOAD GROUND TRUTH & POPULATION TABLES
    # =========================================================================
    print("\n[STEP 0] Loading Entities, Ground Truth, and Identifying Missed Pairs...")
    t0 = time.time()

    # Ground truth pairs
    con.execute(f"""
        CREATE TABLE all_gt_pairs AS 
        SELECT 
            source1_entity_id,
            UNNEST(string_split(matched_entity_ids, ',')) as target_entity_id
        FROM read_csv('{GT_PATH}', delim='\\t', header=true)
        WHERE matched_entity_ids IS NOT NULL AND matched_entity_ids <> '';
    """)

    # Captured V3 pairs
    con.execute(f"""
        CREATE TABLE v3_captured_pairs AS 
        SELECT source1_entity_id, candidate_entity_id as target_entity_id
        FROM read_parquet('{V3_OOF_GLOB}')
        WHERE label = 1;
    """)

    # Missed V3 pairs (2,126,379 expected)
    con.execute("""
        CREATE TABLE missed_gt_pairs AS 
        SELECT 
            g.source1_entity_id, 
            g.target_entity_id,
            CASE WHEN g.target_entity_id LIKE 'S2-%' THEN 'source2' ELSE 'source3' END as target_source
        FROM all_gt_pairs g
        ANTI JOIN v3_captured_pairs c 
          ON g.source1_entity_id = c.source1_entity_id AND g.target_entity_id = c.target_entity_id;
    """)
    n_gt = con.execute("SELECT count(*) FROM all_gt_pairs").fetchone()[0]
    n_cap = con.execute("SELECT count(*) FROM v3_captured_pairs").fetchone()[0]
    n_missed = con.execute("SELECT count(*) FROM missed_gt_pairs").fetchone()[0]
    print(f"  -> Total ground truth pairs: {n_gt:,}")
    print(f"  -> Captured V3 pairs:        {n_cap:,}")
    print(f"  -> Uncaptured Missed pairs:  {n_missed:,} (Verified exact target: 2,126,379)")

    # Read raw S1 text parts to get raw fields and row positions
    print("  Streaming raw S1 table...")
    parts = [str(REPO / 'data' / 'train' / f'train_source1.tsv.part_{ext}') for ext in ['aa', 'ab', 'ac']]
    buf = io.BytesIO()
    for p in parts:
        with open(p, 'rb') as f:
            buf.write(f.read())
    buf.seek(0)
    df_raw_s1 = pd.read_csv(buf, sep='\t')
    df_raw_s1['raw_row_pos'] = np.arange(len(df_raw_s1), dtype=np.int32)
    df_raw_s1.rename(columns={'business_name': 'business_name_raw', 'business_address': 'business_address_raw', 'country': 'country_raw'}, inplace=True)
    con.register('raw_s1_df', df_raw_s1)

    print("  Building unified S1 table...")
    con.execute(f"""
        CREATE TABLE s1_entities AS 
        SELECT 
            r.entity_id,
            f.fold,
            r.raw_row_pos,
            r.business_name_raw,
            r.business_address_raw,
            r.country_raw,
            n.business_name as name_norm,
            n.business_address as addr_norm,
            n.country as country_norm,
            regexp_extract(coalesce(n.business_address,''), '(?:^|[^0-9])([0-9]{{5,6}})(?:[^0-9]|$)', 1) as postal_code,
            regexp_extract(coalesce(n.business_address,''), '[0-9]+[A-Za-z]?', 0) as house_number,
            split_part(trim(coalesce(n.business_name,'')), ' ', 1) as first_token,
            LEFT(trim(coalesce(n.business_name,'')), 4) as prefix4,
            regexp_replace(lower(coalesce(n.business_name,'')), '[^a-z0-9]', '', 'g') as name_clean,
            regexp_replace(lower(coalesce(n.business_address,'')), '[^a-z0-9]', '', 'g') as addr_clean,
            trim(regexp_replace(lower(coalesce(n.business_name,'')), '\\b(inc|llc|ltd|pvt|limited|corp|corporation|co|company|services|enterprises|solutions|associates|group|holdings)\\b', '', 'g')) as name_no_suffix
        FROM raw_s1_df r
        JOIN read_csv('{S1_NORM_PATH}', delim='\\t', header=true) n ON r.entity_id = n.entity_id
        JOIN read_csv('{FOLDS_PATH}', delim='\\t', header=true) f ON r.entity_id = f.source1_entity_id;
    """)

    # Unified Target table (S2 + S3)
    print("  Building unified Target (S2/S3) table...")
    con.execute(f"""
        CREATE TABLE target_entities AS 
        SELECT 
            entity_id,
            'source2' as target_source,
            business_name_raw,
            business_address_raw,
            country_raw,
            business_name_normalized as name_norm,
            business_address_normalized as addr_norm,
            country_normalized as country_norm,
            postal_code,
            house_number,
            first_token,
            name_prefix_4 as prefix4,
            name_clean,
            address_clean as addr_clean,
            trim(regexp_replace(lower(coalesce(business_name_normalized,'')), '\\b(inc|llc|ltd|pvt|limited|corp|corporation|co|company|services|enterprises|solutions|associates|group|holdings)\\b', '', 'g')) as name_no_suffix
        FROM read_parquet('{S2_PARQ}')
        UNION ALL
        SELECT 
            entity_id,
            'source3' as target_source,
            business_name_raw,
            business_address_raw,
            country_raw,
            business_name_normalized as name_norm,
            business_address_normalized as addr_norm,
            country_normalized as country_norm,
            postal_code,
            house_number,
            first_token,
            name_prefix_4 as prefix4,
            name_clean,
            address_clean as addr_clean,
            trim(regexp_replace(lower(coalesce(business_name_normalized,'')), '\\b(inc|llc|ltd|pvt|limited|corp|corporation|co|company|services|enterprises|solutions|associates|group|holdings)\\b', '', 'g')) as name_no_suffix
        FROM read_parquet('{S3_PARQ}');
    """)

    # S1 summary capture status
    print("  Classifying S1 entities into ZERO_CAPTURE, PARTIAL_CAPTURE, FULL_CAPTURE...")
    con.execute(f"""
        CREATE TABLE s1_capture_status AS 
        WITH gt_cnt AS (
            SELECT source1_entity_id, count(*) as A FROM all_gt_pairs GROUP BY 1
        ),
        cap_cnt AS (
            SELECT source1_entity_id, count(*) as k FROM v3_captured_pairs GROUP BY 1
        )
        SELECT 
            s.entity_id as source1_entity_id,
            s.fold,
            s.country_norm as country,
            coalesce(g.A, 0) as A,
            coalesce(c.k, 0) as k,
            CASE 
                WHEN coalesce(g.A, 0) = 0 THEN 'NO_TRUTH'
                WHEN coalesce(c.k, 0) = 0 THEN 'ZERO_CAPTURE'
                WHEN coalesce(c.k, 0) < g.A THEN 'PARTIAL_CAPTURE'
                ELSE 'FULL_CAPTURE'
            END as capture_bucket
        FROM s1_entities s
        LEFT JOIN gt_cnt g ON s.entity_id = g.source1_entity_id
        LEFT JOIN cap_cnt c ON s.entity_id = c.source1_entity_id;
    """)

    s1_bucket_counts = con.execute("SELECT capture_bucket, count(*) FROM s1_capture_status GROUP BY 1 ORDER BY 2 DESC").df()
    print("  S1 Bucket Summary:")
    print(s1_bucket_counts.to_string(index=False))
    print(f"Setup completed in {time.time()-t0:.2f}s.")

    # =========================================================================
    # PART 1: RESIDUAL MISS FORENSICS
    # =========================================================================
    print("\n[PART 1] Residual Miss Forensics on 2,126,379 Missed True Pairs...")
    t0_p1 = time.time()

    con.execute("""
        CREATE TABLE missed_pairs_detailed AS 
        SELECT 
            m.source1_entity_id,
            m.target_entity_id,
            m.target_source,
            s.fold,
            s.country_norm as country,
            b.capture_bucket,
            s.business_name_raw as s1_name_raw,
            tgt.business_name_raw as tgt_name_raw,
            s.name_norm as s1_name,
            tgt.name_norm as tgt_name,
            s.business_address_raw as s1_addr_raw,
            tgt.business_address_raw as tgt_addr_raw,
            s.addr_norm as s1_addr,
            tgt.addr_norm as tgt_addr,
            s.name_clean as s1_name_clean,
            tgt.name_clean as tgt_name_clean,
            s.addr_clean as s1_addr_clean,
            tgt.addr_clean as tgt_addr_clean,
            s.name_no_suffix as s1_no_suffix,
            tgt.name_no_suffix as tgt_no_suffix,
            s.postal_code as s1_pin,
            tgt.postal_code as tgt_pin,
            s.house_number as s1_hn,
            tgt.house_number as tgt_hn,
            s.first_token as s1_first_token,
            tgt.first_token as tgt_first_token,
            s.prefix4 as s1_prefix4,
            tgt.prefix4 as tgt_prefix4,
            -- Forensic Category Indicators A - V
            CASE WHEN s.country_norm <> tgt.country_norm THEN 1 ELSE 0 END as cat_A_country_mismatch,
            CASE WHEN s.prefix4 = '' OR tgt.prefix4 = '' OR s.prefix4 <> tgt.prefix4 THEN 1 ELSE 0 END as cat_B_prefix4_mismatch,
            CASE WHEN s.name_norm <> tgt.name_norm THEN 1 ELSE 0 END as cat_C_exact_name_failure,
            CASE WHEN s.name_no_suffix <> '' AND s.name_no_suffix = tgt.name_no_suffix AND s.name_clean <> tgt.name_clean THEN 1 ELSE 0 END as cat_D_legal_suffix_variation,
            CASE WHEN s.prefix4 <> tgt.prefix4 AND len(list_intersect(string_split(trim(s.name_norm), ' '), string_split(trim(tgt.name_norm), ' '))) >= 2 THEN 1 ELSE 0 END as cat_E_token_order_variation,
            CASE WHEN s.prefix4 <> tgt.prefix4 AND jaro_winkler_similarity(s.name_norm, tgt.name_norm) >= 0.85 THEN 1 ELSE 0 END as cat_F_typo_char_variation,
            CASE WHEN s.house_number = '' OR tgt.house_number = '' THEN 1 ELSE 0 END as cat_G_missing_house_number,
            CASE WHEN s.house_number <> '' AND tgt.house_number <> '' AND s.house_number <> tgt.house_number THEN 1 ELSE 0 END as cat_H_house_number_disagreement,
            CASE WHEN s.postal_code <> '' AND s.postal_code = tgt.postal_code THEN 1 ELSE 0 END as cat_I_pin_shared,
            CASE WHEN s.postal_code = '' OR tgt.postal_code = '' THEN 1 ELSE 0 END as cat_J_pin_absent,
            CASE WHEN len(list_intersect(string_split(trim(s.addr_norm), ' '), string_split(trim(tgt.addr_norm), ' '))) >= 2 THEN 1 ELSE 0 END as cat_K_locality_token_shared,
            CASE WHEN regexp_matches(s.addr_norm || tgt.addr_norm, 'near|opp|behind|beside|road|marg|nagar|colony|block|sector|plot|floor|bazaar') THEN 1 ELSE 0 END as cat_M_landmark_style,
            CASE WHEN s.business_name_raw <> '' AND s.business_name_raw = tgt.business_name_raw THEN 1 ELSE 0 END as cat_N_raw_exact_name,
            CASE WHEN s.business_address_raw <> '' AND s.business_address_raw = tgt.business_address_raw THEN 1 ELSE 0 END as cat_O_raw_exact_addr,
            CASE WHEN s.name_norm <> '' AND s.name_norm = tgt.name_norm THEN 1 ELSE 0 END as cat_P_norm_exact_name,
            CASE WHEN s.addr_norm <> '' AND s.addr_norm = tgt.addr_norm THEN 1 ELSE 0 END as cat_Q_norm_exact_addr,
            CASE WHEN regexp_matches(tgt.name_norm || tgt.addr_norm, '[\u0900-\u097f\u0980-\u09ff\u0b80-\u0bff\u0c00-\u0c7f\u0a80-\u0aff\u0b00-\u0b7f\u0c80-\u0cff\u0d00-\u0d7f]') THEN 1 ELSE 0 END as cat_R_cross_script_indic,
            CASE WHEN len(list_intersect(string_split(trim(s.name_norm), ' '), string_split(trim(tgt.name_norm), ' '))) >= 2 THEN 1 ELSE 0 END as cat_S_strong_name_token_overlap,
            CASE WHEN len(list_intersect(string_split(trim(s.name_norm), ' '), string_split(trim(tgt.name_norm), ' '))) = 1 THEN 1 ELSE 0 END as cat_T_weak_name_token_overlap,
            CASE WHEN len(list_intersect(string_split(trim(s.name_norm), ' '), string_split(trim(tgt.name_norm), ' '))) = 0 AND jaro_winkler_similarity(s.name_norm, tgt.name_norm) < 0.70 THEN 1 ELSE 0 END as cat_U_no_obvious_lexical_relation,
            -- Continuous metrics
            jaro_winkler_similarity(coalesce(s.name_norm,''), coalesce(tgt.name_norm,'')) as name_jw,
            len(list_intersect(string_split(trim(coalesce(s.name_norm,'')), ' '), string_split(trim(coalesce(tgt.name_norm,'')), ' '))) as name_common_tokens,
            len(list_intersect(string_split(trim(coalesce(s.addr_norm,'')), ' '), string_split(trim(coalesce(tgt.addr_norm,'')), ' '))) as addr_common_tokens
        FROM missed_gt_pairs m
        JOIN s1_entities s ON m.source1_entity_id = s.entity_id
        JOIN target_entities tgt ON m.target_entity_id = tgt.entity_id
        JOIN s1_capture_status b ON m.source1_entity_id = b.source1_entity_id;
    """)

    # Overall Summary
    p1_summary = con.execute("""
        SELECT 
            count(*) as total_missed,
            avg(cat_A_country_mismatch) as country_mismatch_pct,
            avg(cat_B_prefix4_mismatch) as prefix4_mismatch_pct,
            avg(cat_C_exact_name_failure) as exact_name_failure_pct,
            avg(cat_D_legal_suffix_variation) as legal_suffix_variation_pct,
            avg(cat_E_token_order_variation) as token_order_variation_pct,
            avg(cat_F_typo_char_variation) as typo_char_variation_pct,
            avg(cat_G_missing_house_number) as missing_house_number_pct,
            avg(cat_H_house_number_disagreement) as house_number_disagreement_pct,
            avg(cat_I_pin_shared) as pin_shared_pct,
            avg(cat_J_pin_absent) as pin_absent_pct,
            avg(cat_K_locality_token_shared) as locality_token_shared_pct,
            avg(cat_M_landmark_style) as landmark_style_pct,
            avg(cat_N_raw_exact_name) as raw_exact_name_pct,
            avg(cat_O_raw_exact_addr) as raw_exact_addr_pct,
            avg(cat_P_norm_exact_name) as norm_exact_name_pct,
            avg(cat_Q_norm_exact_addr) as norm_exact_addr_pct,
            avg(cat_R_cross_script_indic) as cross_script_indic_pct,
            avg(cat_S_strong_name_token_overlap) as strong_name_token_overlap_pct,
            avg(cat_T_weak_name_token_overlap) as weak_name_token_overlap_pct,
            avg(cat_U_no_obvious_lexical_relation) as no_obvious_lexical_relation_pct,
            avg(name_jw) as avg_name_jw
        FROM missed_pairs_detailed;
    """).df().to_dict(orient="records")[0]

    # Breakdown by Country (India vs US)
    p1_by_country = con.execute("""
        SELECT 
            country,
            count(*) as missed_pairs,
            count(*) * 100.0 / 2126379 as share_of_all_misses_pct,
            avg(cat_B_prefix4_mismatch) as prefix4_mismatch_pct,
            avg(cat_D_legal_suffix_variation) as legal_suffix_variation_pct,
            avg(cat_E_token_order_variation) as token_order_variation_pct,
            avg(cat_F_typo_char_variation) as typo_char_variation_pct,
            avg(cat_R_cross_script_indic) as cross_script_indic_pct,
            avg(cat_I_pin_shared) as pin_shared_pct,
            avg(cat_J_pin_absent) as pin_absent_pct,
            avg(cat_M_landmark_style) as landmark_style_pct,
            avg(cat_S_strong_name_token_overlap) as strong_token_overlap_pct,
            avg(name_jw) as avg_name_jw
        FROM missed_pairs_detailed
        GROUP BY country ORDER BY missed_pairs DESC;
    """).df()

    # Breakdown by Target Source (S2 vs S3)
    p1_by_source = con.execute("""
        SELECT 
            target_source,
            count(*) as missed_pairs,
            count(*) * 100.0 / 2126379 as share_of_all_misses_pct,
            avg(cat_B_prefix4_mismatch) as prefix4_mismatch_pct,
            avg(cat_D_legal_suffix_variation) as legal_suffix_variation_pct,
            avg(cat_R_cross_script_indic) as cross_script_indic_pct,
            avg(cat_I_pin_shared) as pin_shared_pct,
            avg(cat_J_pin_absent) as pin_absent_pct,
            avg(cat_S_strong_name_token_overlap) as strong_token_overlap_pct,
            avg(name_jw) as avg_name_jw
        FROM missed_pairs_detailed
        GROUP BY target_source ORDER BY missed_pairs DESC;
    """).df()

    # Breakdown by Capture Bucket (ZERO_CAPTURE vs PARTIAL_CAPTURE)
    p1_by_bucket = con.execute("""
        SELECT 
            capture_bucket,
            count(*) as missed_pairs,
            count(distinct source1_entity_id) as entity_count,
            avg(cat_B_prefix4_mismatch) as prefix4_mismatch_pct,
            avg(cat_D_legal_suffix_variation) as legal_suffix_variation_pct,
            avg(cat_R_cross_script_indic) as cross_script_indic_pct,
            avg(cat_I_pin_shared) as pin_shared_pct,
            avg(cat_J_pin_absent) as pin_absent_pct,
            avg(cat_S_strong_name_token_overlap) as strong_token_overlap_pct,
            avg(name_jw) as avg_name_jw
        FROM missed_pairs_detailed
        GROUP BY capture_bucket ORDER BY missed_pairs DESC;
    """).df()

    print("Forensics Summary by Country:")
    print(p1_by_country.to_string(index=False))
    print("\nForensics Summary by Target Source (S2 vs S3):")
    print(p1_by_source.to_string(index=False))
    print("\nForensics Summary by Capture Bucket (Zero vs Partial):")
    print(p1_by_bucket.to_string(index=False))
    print(f"Part 1 completed in {time.time()-t0_p1:.2f}s.")

    # =========================================================================
    # PART 2: ZERO-CAPTURE ANALYSIS
    # =========================================================================
    print("\n[PART 2] In-Depth Zero-Capture Analysis (130,284 S1 Entities)...")
    t0_p2 = time.time()

    zero_cap_analysis = con.execute("""
        WITH z_pairs AS (
            SELECT * FROM missed_pairs_detailed WHERE capture_bucket = 'ZERO_CAPTURE'
        ),
        per_entity AS (
            SELECT 
                source1_entity_id,
                max(country) as country,
                max(cat_I_pin_shared) as has_pin_shared,
                max(case when s1_pin = '' then 1 else 0 end) as s1_has_no_pin,
                max(cat_K_locality_token_shared) as has_rare_locality_shared,
                max(cat_D_legal_suffix_variation) as has_suffix_stripped_match,
                max(case when name_jw >= 0.85 then 1 else 0 end) as has_char_ngram_similarity,
                max(cat_R_cross_script_indic) as has_transliteration_script_diff,
                max(case when cat_N_raw_exact_name = 1 then 1 else 0 end) as has_raw_exact_name,
                max(case when cat_S_strong_name_token_overlap = 1 then 1 else 0 end) as has_strong_token_overlap
            FROM z_pairs
            GROUP BY 1
        )
        SELECT 
            count(*) as total_zero_capture_entities,
            avg(has_pin_shared) as recoverable_by_shared_pin_pct,
            avg(s1_has_no_pin) as s1_missing_pin_pct,
            avg(has_rare_locality_shared) as recoverable_by_locality_tokens_pct,
            avg(has_suffix_stripped_match) as recoverable_by_suffix_stripped_name_pct,
            avg(has_char_ngram_similarity) as recoverable_by_char_ngram_jw85_pct,
            avg(has_transliteration_script_diff) as affected_by_transliteration_script_pct,
            avg(has_strong_token_overlap) as recoverable_by_name_token_overlap_pct,
            avg(has_raw_exact_name) as recoverable_by_raw_exact_name_pct
        FROM per_entity;
    """).df().to_dict(orient="records")[0]

    zero_cap_by_country = con.execute("""
        WITH z_pairs AS (
            SELECT * FROM missed_pairs_detailed WHERE capture_bucket = 'ZERO_CAPTURE'
        ),
        per_entity AS (
            SELECT 
                source1_entity_id,
                max(country) as country,
                max(cat_I_pin_shared) as has_pin_shared,
                max(case when s1_pin = '' then 1 else 0 end) as s1_has_no_pin,
                max(cat_K_locality_token_shared) as has_rare_locality_shared,
                max(cat_D_legal_suffix_variation) as has_suffix_stripped_match,
                max(case when name_jw >= 0.85 then 1 else 0 end) as has_char_ngram_similarity,
                max(cat_R_cross_script_indic) as has_transliteration_script_diff,
                max(case when cat_S_strong_name_token_overlap = 1 then 1 else 0 end) as has_strong_token_overlap
            FROM z_pairs
            GROUP BY 1
        )
        SELECT 
            country,
            count(*) as zero_capture_entities,
            avg(has_pin_shared) as recoverable_by_shared_pin_pct,
            avg(s1_has_no_pin) as s1_missing_pin_pct,
            avg(has_suffix_stripped_match) as recoverable_by_suffix_stripped_name_pct,
            avg(has_char_ngram_similarity) as recoverable_by_char_ngram_jw85_pct,
            avg(has_transliteration_script_diff) as affected_by_transliteration_script_pct,
            avg(has_strong_token_overlap) as recoverable_by_token_overlap_pct
        FROM per_entity
        GROUP BY country;
    """).df()

    print("Zero-Capture Recoverability by Observable Signal:")
    for k, v in zero_cap_analysis.items():
        print(f"  {k:45s}: {v}")
    print("\nZero-Capture Breakdown by Country (India vs US):")
    print(zero_cap_by_country.to_string(index=False))
    print(f"Part 2 completed in {time.time()-t0_p2:.2f}s.")

    # =========================================================================
    # PART 4: STRUCTURAL / RAW-FIELD SIGNAL AUDIT
    # =========================================================================
    print("\n[PART 4] Structural / Raw-Field Signal Audit...")
    t0_p4 = time.time()

    # A. Raw field exact match rates
    raw_exact_stats = con.execute("""
        SELECT 
            'Missed True Pairs' as population,
            count(*) as sample_size,
            avg(case when s1_name_raw = tgt_name_raw then 1.0 else 0.0 end) as raw_name_exact_pct,
            avg(case when s1_addr_raw = tgt_addr_raw then 1.0 else 0.0 end) as raw_addr_exact_pct,
            avg(case when s1_name = tgt_name then 1.0 else 0.0 end) as norm_name_exact_pct,
            avg(case when s1_addr = tgt_addr then 1.0 else 0.0 end) as norm_addr_exact_pct
        FROM missed_pairs_detailed
        UNION ALL
        SELECT 
            'Captured True Pairs' as population,
            count(*) as sample_size,
            avg(case when s.business_name_raw = tgt.business_name_raw then 1.0 else 0.0 end) as raw_name_exact_pct,
            avg(case when s.business_address_raw = tgt.business_address_raw then 1.0 else 0.0 end) as raw_addr_exact_pct,
            avg(case when s.name_norm = tgt.name_norm then 1.0 else 0.0 end) as norm_name_exact_pct,
            avg(case when s.addr_norm = tgt.addr_norm then 1.0 else 0.0 end) as norm_addr_exact_pct
        FROM (SELECT * FROM v3_captured_pairs USING SAMPLE 100000) c
        JOIN s1_entities s ON c.source1_entity_id = s.entity_id
        JOIN target_entities tgt ON c.target_entity_id = tgt.entity_id;
    """).df()
    print("Raw vs Normalized Field Exact Match Rates:")
    print(raw_exact_stats.to_string(index=False))

    # B. ID Structure Audit
    id_struct_stats = con.execute("""
        WITH true_sample AS (
            SELECT 
                source1_entity_id, 
                target_entity_id,
                CAST(regexp_extract(source1_entity_id, '[0-9]+', 0) AS BIGINT) as s1_num,
                CAST(regexp_extract(target_entity_id, '[0-9]+', 0) AS BIGINT) as tgt_num
            FROM all_gt_pairs
            USING SAMPLE 100000
        ),
        random_sample AS (
            SELECT 
                s.entity_id as source1_entity_id,
                tgt.entity_id as target_entity_id,
                CAST(regexp_extract(s.entity_id, '[0-9]+', 0) AS BIGINT) as s1_num,
                CAST(regexp_extract(tgt.entity_id, '[0-9]+', 0) AS BIGINT) as tgt_num
            FROM (SELECT entity_id FROM s1_entities USING SAMPLE 100000) s
            JOIN (SELECT entity_id FROM target_entities USING SAMPLE 100000) tgt ON abs(hash(s.entity_id || tgt.entity_id)) % 100 = 0
            LIMIT 100000
        )
        SELECT 
            'True Pairs' as population,
            avg(case when s1_num = tgt_num then 1.0 else 0.0 end) as exact_numeric_id_rate,
            avg(case when abs(s1_num - tgt_num) <= 1000 then 1.0 else 0.0 end) as small_offset_rate,
            avg(case when s1_num % 10 = tgt_num % 10 then 1.0 else 0.0 end) as mod10_equal_rate,
            avg(abs(s1_num - tgt_num)) as avg_id_diff
        FROM true_sample
        UNION ALL
        SELECT 
            'Random Pairs' as population,
            avg(case when s1_num = tgt_num then 1.0 else 0.0 end) as exact_numeric_id_rate,
            avg(case when abs(s1_num - tgt_num) <= 1000 then 1.0 else 0.0 end) as small_offset_rate,
            avg(case when s1_num % 10 = tgt_num % 10 then 1.0 else 0.0 end) as mod10_equal_rate,
            avg(abs(s1_num - tgt_num)) as avg_id_diff
        FROM random_sample;
    """).df()
    print("\nID Structure Audit:")
    print(id_struct_stats.to_string(index=False))

    # C. Row Position Distance Audit
    row_pos_stats = con.execute("""
        WITH true_sample AS (
            SELECT 
                s.raw_row_pos as s1_pos,
                ROW_NUMBER() OVER (ORDER BY tgt.entity_id) as tgt_pos
            FROM (SELECT * FROM all_gt_pairs USING SAMPLE 50000) g
            JOIN s1_entities s ON g.source1_entity_id = s.entity_id
            JOIN target_entities tgt ON g.target_entity_id = tgt.entity_id
        )
        SELECT 
            avg(abs(s1_pos - tgt_pos)) as avg_row_pos_distance,
            median(abs(s1_pos - tgt_pos)) as median_row_pos_distance
        FROM true_sample;
    """).df().to_dict(orient="records")[0]
    print(f"\nRow Position Correlation: Avg Distance = {row_pos_stats['avg_row_pos_distance']:.1f}, Median Distance = {row_pos_stats['median_row_pos_distance']:.1f} (Independent random distribution)")

    # E. Cross-Source Sibling Transitivity Audit
    con.execute("""
        CREATE TEMP TABLE s1_multi_matches AS 
        SELECT source1_entity_id, count(distinct target_source) as src_cnt
        FROM all_gt_pairs g
        JOIN target_entities tgt ON g.target_entity_id = tgt.entity_id
        GROUP BY 1 HAVING count(distinct target_source) = 2;
    """)
    n_multi_src = con.execute("SELECT count(*) FROM s1_multi_matches").fetchone()[0]

    sibling_recovery = con.execute("""
        WITH captured_siblings AS (
            SELECT 
                c.source1_entity_id,
                tgt.target_source,
                tgt.name_clean as sib_name_clean,
                tgt.addr_clean as sib_addr_clean,
                tgt.postal_code as sib_pin
            FROM v3_captured_pairs c
            JOIN target_entities tgt ON c.target_entity_id = tgt.entity_id
        ),
        missed_with_sib AS (
            SELECT 
                m.source1_entity_id,
                m.target_entity_id,
                m.target_source as missed_source,
                s.sib_name_clean,
                s.sib_addr_clean,
                s.sib_pin,
                tgt.name_clean as missed_name_clean,
                tgt.addr_clean as missed_addr_clean,
                tgt.postal_code as missed_pin
            FROM missed_gt_pairs m
            JOIN captured_siblings s ON m.source1_entity_id = s.source1_entity_id AND m.target_source <> s.target_source
            JOIN target_entities tgt ON m.target_entity_id = tgt.entity_id
        )
        SELECT 
            count(*) as missed_pairs_with_sibling,
            count(distinct source1_entity_id) as s1_entities_with_sibling,
            avg(case when sib_name_clean = missed_name_clean and sib_name_clean <> '' then 1.0 else 0.0 end) as shared_clean_name_pct,
            avg(case when sib_addr_clean = missed_addr_clean and sib_addr_clean <> '' then 1.0 else 0.0 end) as shared_clean_addr_pct,
            avg(case when sib_pin = missed_pin and sib_pin <> '' then 1.0 else 0.0 end) as shared_pin_pct,
            avg(case when (sib_name_clean = missed_name_clean and sib_name_clean <> '') OR (sib_addr_clean = missed_addr_clean and sib_addr_clean <> '') then 1.0 else 0.0 end) as recoverable_by_sibling_pct
        FROM missed_with_sib;
    """).df().to_dict(orient="records")[0]
    print(f"\nCross-Source Sibling Transitivity Analysis:")
    print(f"  S1 Entities with both S2 and S3 true matches: {n_multi_src:,}")
    print(f"  Missed pairs with a captured sibling:         {sibling_recovery['missed_pairs_with_sibling']:,}")
    print(f"  Sibling Shared Exact Clean Name:              {sibling_recovery['shared_clean_name_pct']*100:.2f}%")
    print(f"  Sibling Shared Exact Clean Address:           {sibling_recovery['shared_clean_addr_pct']*100:.2f}%")
    print(f"  Recoverable by Exact Sibling Name or Addr:    {sibling_recovery['recoverable_by_sibling_pct']*100:.2f}%")
    print(f"Part 4 completed in {time.time()-t0_p4:.2f}s.")

    # =========================================================================
    # PART 3: RETRIEVAL-KEY SIMULATION & BIAS CONTROL (FOLDS 0-3 vs FOLD 4)
    # =========================================================================
    print("\n[PART 3] Simulating Label-Free Retrieval Channels with Strict Train/Held-Out Validation...")
    t0_p3 = time.time()

    con.execute("""
        CREATE TEMP TABLE s1_fold_gt AS 
        SELECT source1_entity_id, fold, country, A, k
        FROM s1_capture_status;
    """)

    def evaluate_channel(cand_table_name: str, channel_key: str, channel_desc: str):
        t_c = time.time()
        # Find recovered true pairs among MISSED ground truth
        con.execute(f"""
            CREATE TEMP TABLE cur_recovered AS 
            SELECT DISTINCT m.source1_entity_id, m.target_entity_id, m.target_source, s.fold, s.country_norm as country, b.capture_bucket
            FROM {cand_table_name} c
            JOIN missed_gt_pairs m ON c.source1_entity_id = m.source1_entity_id AND c.target_entity_id = m.target_entity_id
            JOIN s1_entities s ON m.source1_entity_id = s.entity_id
            JOIN s1_capture_status b ON m.source1_entity_id = b.source1_entity_id;
        """)

        rec_stats = con.execute("""
            SELECT 
                count(*) as recovered_pairs,
                count(case when fold = 4 then 1 end) as rec_pairs_f4,
                count(case when fold <> 4 then 1 end) as rec_pairs_train,
                count(case when country = 'INDIA' then 1 end) as rec_pairs_india,
                count(case when country = 'US' then 1 end) as rec_pairs_us,
                count(case when target_source = 'source2' then 1 end) as rec_pairs_s2,
                count(case when target_source = 'source3' then 1 end) as rec_pairs_s3,
                count(distinct case when capture_bucket = 'ZERO_CAPTURE' then source1_entity_id end) as zero_cap_rescued,
                count(distinct case when capture_bucket = 'PARTIAL_CAPTURE' then source1_entity_id end) as partial_cap_improved
            FROM cur_recovered;
        """).df().to_dict(orient="records")[0]

        add_vol = con.execute(f"SELECT count(*) FROM {cand_table_name}").fetchone()[0]

        # Held-out Fold 4 Oracle Gain
        f4_base_oracle = 0.869591
        new_f4_oracle = con.execute(f"""
            WITH add_k AS (
                SELECT source1_entity_id, count(*) as k_add
                FROM cur_recovered WHERE fold = 4
                GROUP BY 1
            )
            SELECT avg(
                CASE 
                    WHEN f.A = 0 THEN 1.0
                    WHEN (f.k + coalesce(a.k_add, 0)) = 0 THEN 0.0
                    ELSE (5.0 * (f.k + coalesce(a.k_add, 0))) / (f.A + 4.0 * (f.k + coalesce(a.k_add, 0)))
                END
            )
            FROM s1_fold_gt f
            LEFT JOIN add_k a ON f.source1_entity_id = a.source1_entity_id
            WHERE f.fold = 4;
        """).fetchone()[0]
        f4_oracle_gain = new_f4_oracle - f4_base_oracle

        # Global Oracle Gain
        base_oracle = 0.869706
        new_global_oracle = con.execute(f"""
            WITH add_k AS (
                SELECT source1_entity_id, count(*) as k_add
                FROM cur_recovered
                GROUP BY 1
            )
            SELECT avg(
                CASE 
                    WHEN f.A = 0 THEN 1.0
                    WHEN (f.k + coalesce(a.k_add, 0)) = 0 THEN 0.0
                    ELSE (5.0 * (f.k + coalesce(a.k_add, 0))) / (f.A + 4.0 * (f.k + coalesce(a.k_add, 0)))
                END
            )
            FROM s1_fold_gt f
            LEFT JOIN add_k a ON f.source1_entity_id = a.source1_entity_id;
        """).fetchone()[0]
        global_oracle_gain = new_global_oracle - base_oracle

        con.execute("DROP TABLE cur_recovered;")

        result = {
            "channel_key": channel_key,
            "description": channel_desc,
            "added_candidates_train": add_vol,
            "candidates_per_s1_added": add_vol / 2206821.0,
            "recovered_true_pairs_total": rec_stats["recovered_pairs"],
            "recovered_pairs_heldout_f4": rec_stats["rec_pairs_f4"],
            "zero_capture_rescued_total": rec_stats["zero_cap_rescued"],
            "partial_capture_improved_total": rec_stats["partial_cap_improved"],
            "recovered_india": rec_stats["rec_pairs_india"],
            "recovered_us": rec_stats["rec_pairs_us"],
            "recovered_s2": rec_stats["rec_pairs_s2"],
            "recovered_s3": rec_stats["rec_pairs_s3"],
            "heldout_f4_oracle_gain": f4_oracle_gain,
            "global_oracle_gain": global_oracle_gain,
            "gain_per_million_cands": (global_oracle_gain / (add_vol / 1_000_000.0)) if add_vol > 0 else 0.0,
            "elapsed_sec": time.time() - t_c
        }
        return result

    channel_eval_results = []

    # Channel 1: Suffix-Stripped Business Name (Cluster Cap <= 5)
    print("  Evaluating Channel 1: Suffix-Stripped Business Name (Cluster Cap <= 5)...")
    con.execute("""
        CREATE TEMP TABLE c1_pairs AS 
        WITH tgt_counts AS (
            SELECT country_norm, name_no_suffix, count(*) as cnt
            FROM target_entities
            WHERE length(name_no_suffix) >= 6
            GROUP BY 1, 2
            HAVING count(*) <= 5
        )
        SELECT s.entity_id as source1_entity_id, tgt.entity_id as target_entity_id
        FROM s1_entities s
        JOIN tgt_counts tc ON s.country_norm = tc.country_norm AND s.name_no_suffix = tc.name_no_suffix
        JOIN target_entities tgt ON s.country_norm = tgt.country_norm AND s.name_no_suffix = tgt.name_no_suffix;
    """)
    res_c1 = evaluate_channel("c1_pairs", "suffix_stripped_name_cap5", "Suffix-Stripped Name Match (length>=6, cluster<=5)")
    channel_eval_results.append(res_c1)
    con.execute("DROP TABLE c1_pairs;")

    # Channel 2: One-Hop Sibling Expansion via Exact Clean Name (Cluster Cap <= 5)
    print("  Evaluating Channel 2: Sibling Expansion via Exact Clean Name (Cluster Cap <= 5)...")
    con.execute("""
        CREATE TEMP TABLE c2_pairs AS 
        WITH captured AS (
            SELECT c.source1_entity_id, tgt.target_source, tgt.country_norm, tgt.name_clean
            FROM v3_captured_pairs c
            JOIN target_entities tgt ON c.target_entity_id = tgt.entity_id
            WHERE length(tgt.name_clean) >= 6
        ),
        tgt_counts AS (
            SELECT country_norm, name_clean, count(*) as cnt
            FROM target_entities
            WHERE length(name_clean) >= 6
            GROUP BY 1, 2
            HAVING count(*) <= 5
        )
        SELECT DISTINCT c.source1_entity_id, t2.entity_id as target_entity_id
        FROM captured c
        JOIN tgt_counts tc ON c.country_norm = tc.country_norm AND c.name_clean = tc.name_clean
        JOIN target_entities t2 
          ON c.country_norm = t2.country_norm 
         AND c.name_clean = t2.name_clean 
         AND c.target_source <> t2.target_source;
    """)
    res_c2 = evaluate_channel("c2_pairs", "sibling_exact_name_cap5", "Sibling Clean Name Expansion (cluster<=5, S1->S2->S3)")
    channel_eval_results.append(res_c2)
    con.execute("DROP TABLE c2_pairs;")

    # Channel 3: One-Hop Sibling Expansion via Exact Clean Address
    print("  Evaluating Channel 3: Sibling Expansion via Exact Clean Address...")
    con.execute("""
        CREATE TEMP TABLE c3_pairs AS 
        WITH captured AS (
            SELECT c.source1_entity_id, tgt.target_source, tgt.country_norm, tgt.addr_clean
            FROM v3_captured_pairs c
            JOIN target_entities tgt ON c.target_entity_id = tgt.entity_id
            WHERE length(tgt.addr_clean) >= 12
        )
        SELECT DISTINCT c.source1_entity_id, t2.entity_id as target_entity_id
        FROM captured c
        JOIN target_entities t2 
          ON c.country_norm = t2.country_norm 
         AND c.addr_clean = t2.addr_clean 
         AND c.target_source <> t2.target_source;
    """)
    res_c3 = evaluate_channel("c3_pairs", "sibling_exact_addr", "Sibling Exact Clean Address Expansion (length>=12, S1->S2->S3)")
    channel_eval_results.append(res_c3)
    con.execute("DROP TABLE c3_pairs;")

    # Channel 4: PIN + House Number Co-occurrence
    print("  Evaluating Channel 4: PIN + House Number...")
    con.execute("""
        CREATE TEMP TABLE c4_pairs AS 
        SELECT s.entity_id as source1_entity_id, tgt.entity_id as target_entity_id
        FROM s1_entities s
        JOIN target_entities tgt 
          ON s.country_norm = tgt.country_norm 
         AND s.postal_code = tgt.postal_code 
         AND s.house_number = tgt.house_number
        WHERE s.postal_code <> '' AND s.house_number <> '';
    """)
    res_c4 = evaluate_channel("c4_pairs", "pin_house_number", "PIN + House Number Co-occurrence")
    channel_eval_results.append(res_c4)
    con.execute("DROP TABLE c4_pairs;")

    # Channel 5: Raw Exact-Name Key
    print("  Evaluating Channel 5: Raw Exact-Name Key...")
    con.execute("""
        CREATE TEMP TABLE c5_pairs AS 
        SELECT s.entity_id as source1_entity_id, tgt.entity_id as target_entity_id
        FROM s1_entities s
        JOIN target_entities tgt 
          ON s.country_norm = tgt.country_norm 
         AND s.business_name_raw = tgt.business_name_raw
        WHERE length(s.business_name_raw) >= 6;
    """)
    res_c5 = evaluate_channel("c5_pairs", "raw_exact_name", "Raw Business Name Exact Match")
    channel_eval_results.append(res_c5)
    con.execute("DROP TABLE c5_pairs;")

    # Channel 6: Raw Exact-Address Key
    print("  Evaluating Channel 6: Raw Exact-Address Key...")
    con.execute("""
        CREATE TEMP TABLE c6_pairs AS 
        SELECT s.entity_id as source1_entity_id, tgt.entity_id as target_entity_id
        FROM s1_entities s
        JOIN target_entities tgt 
          ON s.country_norm = tgt.country_norm 
         AND s.business_address_raw = tgt.business_address_raw
        WHERE length(s.business_address_raw) >= 12;
    """)
    res_c6 = evaluate_channel("c6_pairs", "raw_exact_addr", "Raw Business Address Exact Match")
    channel_eval_results.append(res_c6)
    con.execute("DROP TABLE c6_pairs;")

    # Channel 7: Normalized Exact-Name Key
    print("  Evaluating Channel 7: Normalized Exact-Name Key...")
    con.execute("""
        CREATE TEMP TABLE c7_pairs AS 
        SELECT s.entity_id as source1_entity_id, tgt.entity_id as target_entity_id
        FROM s1_entities s
        JOIN target_entities tgt 
          ON s.country_norm = tgt.country_norm 
         AND s.name_norm = tgt.name_norm
        WHERE length(s.name_norm) >= 6;
    """)
    res_c7 = evaluate_channel("c7_pairs", "norm_exact_name", "Normalized Business Name Exact Match")
    channel_eval_results.append(res_c7)
    con.execute("DROP TABLE c7_pairs;")

    # Channel 8: Normalized Exact-Address Key
    print("  Evaluating Channel 8: Normalized Exact-Address Key...")
    con.execute("""
        CREATE TEMP TABLE c8_pairs AS 
        SELECT s.entity_id as source1_entity_id, tgt.entity_id as target_entity_id
        FROM s1_entities s
        JOIN target_entities tgt 
          ON s.country_norm = tgt.country_norm 
         AND s.addr_norm = tgt.addr_norm
        WHERE length(s.addr_norm) >= 12;
    """)
    res_c8 = evaluate_channel("c8_pairs", "norm_exact_addr", "Normalized Business Address Exact Match")
    channel_eval_results.append(res_c8)
    con.execute("DROP TABLE c8_pairs;")

    # Channel 9: First Token + PIN
    print("  Evaluating Channel 9: First Token + PIN...")
    con.execute("""
        CREATE TEMP TABLE c9_pairs AS 
        SELECT s.entity_id as source1_entity_id, tgt.entity_id as target_entity_id
        FROM s1_entities s
        JOIN target_entities tgt 
          ON s.country_norm = tgt.country_norm 
         AND s.postal_code = tgt.postal_code 
         AND s.first_token = tgt.first_token
        WHERE s.postal_code <> '' AND length(s.first_token) >= 4;
    """)
    res_c9 = evaluate_channel("c9_pairs", "first_token_plus_pin", "First Name Token + Postal PIN Match")
    channel_eval_results.append(res_c9)
    con.execute("DROP TABLE c9_pairs;")

    # Channel 10: PIN-Anchored Character 3-Gram (top-3 per S1)
    print("  Evaluating Channel 10: PIN-Anchored Character 3-Gram (Bounded top-3)...")
    con.execute("""
        CREATE TEMP TABLE c10_pairs AS 
        WITH pin_matches AS (
            SELECT 
                s.entity_id as source1_entity_id, 
                tgt.entity_id as target_entity_id,
                jaro_winkler_similarity(s.name_norm, tgt.name_norm) as jw,
                ROW_NUMBER() OVER (PARTITION BY s.entity_id ORDER BY jaro_winkler_similarity(s.name_norm, tgt.name_norm) DESC) as rk
            FROM s1_entities s
            JOIN target_entities tgt 
              ON s.country_norm = tgt.country_norm 
             AND s.postal_code = tgt.postal_code
            WHERE s.postal_code <> '' AND jaro_winkler_similarity(s.name_norm, tgt.name_norm) >= 0.85
        )
        SELECT source1_entity_id, target_entity_id
        FROM pin_matches
        WHERE rk <= 3;
    """)
    res_c10 = evaluate_channel("c10_pairs", "pin_anchored_ngram_top3", "PIN-Anchored Character N-Gram (JW>=0.85, top-3)")
    channel_eval_results.append(res_c10)
    con.execute("DROP TABLE c10_pairs;")

    df_channels = pd.DataFrame(channel_eval_results)
    df_channels.to_csv(OUT_CHANNELS_TSV, sep="\t", index=False)
    print(f"\nChannel Simulation Results saved to {OUT_CHANNELS_TSV}:")
    print(df_channels[["channel_key", "added_candidates_train", "candidates_per_s1_added", "recovered_true_pairs_total", "zero_capture_rescued_total", "heldout_f4_oracle_gain", "global_oracle_gain", "gain_per_million_cands"]].to_string(index=False))
    print(f"Part 3 completed in {time.time()-t0_p3:.2f}s.")

    # =========================================================================
    # PART 6: TRAIN/TEST VOLUME PARITY CHECK
    # =========================================================================
    print("\n[PART 6] Test Volume Parity Check for Promising Channels...")
    t0_p6 = time.time()
    con.execute(f"""
        CREATE TABLE test_s1 AS 
        SELECT 
            entity_id,
            business_name as name_norm,
            business_address as addr_norm,
            country as country_norm,
            regexp_extract(coalesce(business_address,''), '(?:^|[^0-9])([0-9]{{5,6}})(?:[^0-9]|$)', 1) as postal_code,
            regexp_extract(coalesce(business_address,''), '[0-9]+[A-Za-z]?', 0) as house_number,
            split_part(trim(coalesce(business_name,'')), ' ', 1) as first_token,
            LEFT(trim(coalesce(business_name,'')), 4) as prefix4,
            regexp_replace(lower(coalesce(business_name,'')), '[^a-z0-9]', '', 'g') as name_clean,
            regexp_replace(lower(coalesce(business_address,'')), '[^a-z0-9]', '', 'g') as addr_clean,
            trim(regexp_replace(lower(coalesce(business_name,'')), '\\b(inc|llc|ltd|pvt|limited|corp|corporation|co|company|services|enterprises|solutions|associates|group|holdings)\\b', '', 'g')) as name_no_suffix
        FROM read_csv('{TEST_S1_NORM}', delim='\\t', header=true);
    """)

    con.execute(f"""
        CREATE TABLE test_targets AS 
        SELECT 
            entity_id,
            'source2' as target_source,
            business_name_normalized as name_norm,
            business_address_normalized as addr_norm,
            country_normalized as country_norm,
            postal_code,
            house_number,
            first_token,
            name_prefix_4 as prefix4,
            name_clean,
            address_clean as addr_clean,
            trim(regexp_replace(lower(coalesce(business_name_normalized,'')), '\\b(inc|llc|ltd|pvt|limited|corp|corporation|co|company|services|enterprises|solutions|associates|group|holdings)\\b', '', 'g')) as name_no_suffix
        FROM read_parquet('{TEST_S2_PARQ}')
        UNION ALL
        SELECT 
            entity_id,
            'source3' as target_source,
            business_name_normalized as name_norm,
            business_address_normalized as addr_norm,
            country_normalized as country_norm,
            postal_code,
            house_number,
            first_token,
            name_prefix_4 as prefix4,
            name_clean,
            address_clean as addr_clean,
            trim(regexp_replace(lower(coalesce(business_name_normalized,'')), '\\b(inc|llc|ltd|pvt|limited|corp|corporation|co|company|services|enterprises|solutions|associates|group|holdings)\\b', '', 'g')) as name_no_suffix
        FROM read_parquet('{TEST_S3_PARQ}');
    """)

    test_vol_suff_cap5 = con.execute("""
        WITH tgt_counts AS (
            SELECT country_norm, name_no_suffix, count(*) as cnt
            FROM test_targets
            WHERE length(name_no_suffix) >= 6
            GROUP BY 1, 2
            HAVING count(*) <= 5
        )
        SELECT count(*)
        FROM test_s1 s
        JOIN tgt_counts tc ON s.country_norm = tc.country_norm AND s.name_no_suffix = tc.name_no_suffix
        JOIN test_targets tgt ON s.country_norm = tgt.country_norm AND s.name_no_suffix = tgt.name_no_suffix;
    """).fetchone()[0]

    test_vol_pin_hn = con.execute("""
        SELECT count(*)
        FROM test_s1 s
        JOIN test_targets tgt 
          ON s.country_norm = tgt.country_norm 
         AND s.postal_code = tgt.postal_code 
         AND s.house_number = tgt.house_number
        WHERE s.postal_code <> '' AND s.house_number <> '';
    """).fetchone()[0]

    test_volume_checks = {
        "train_s1_entities": 2206821,
        "test_s1_entities": 1732544,
        "suffix_stripped_name_cap5": {
            "train_added": int(res_c1["added_candidates_train"]),
            "train_cands_per_s1": float(res_c1["candidates_per_s1_added"]),
            "test_added": int(test_vol_suff_cap5),
            "test_cands_per_s1": float(test_vol_suff_cap5 / 1732544.0),
            "test_to_train_ratio": float((test_vol_suff_cap5 / 1732544.0) / res_c1["candidates_per_s1_added"]) if res_c1["candidates_per_s1_added"] > 0 else 1.0
        },
        "pin_house_number": {
            "train_added": int(res_c4["added_candidates_train"]),
            "train_cands_per_s1": float(res_c4["candidates_per_s1_added"]),
            "test_added": int(test_vol_pin_hn),
            "test_cands_per_s1": float(test_vol_pin_hn / 1732544.0),
            "test_to_train_ratio": float((test_vol_pin_hn / 1732544.0) / res_c4["candidates_per_s1_added"]) if res_c4["candidates_per_s1_added"] > 0 else 1.0
        }
    }
    print("Test Volume Parity Summary:")
    print(json.dumps(test_volume_checks, indent=2))
    print(f"Part 6 completed in {time.time()-t0_p6:.2f}s.")

    # =========================================================================
    # PART 7: CUMULATIVE SIMULATION
    # =========================================================================
    print("\n[PART 7] Cumulative Candidate Channel Simulation...")
    t0_p7 = time.time()

    # Pre-build candidate tables for the top 3 channels:
    con.execute("""
        CREATE TEMP TABLE ch_suff AS 
        WITH tgt_counts AS (
            SELECT country_norm, name_no_suffix, count(*) as cnt
            FROM target_entities
            WHERE length(name_no_suffix) >= 6
            GROUP BY 1, 2
            HAVING count(*) <= 5
        )
        SELECT s.entity_id as source1_entity_id, tgt.entity_id as target_entity_id
        FROM s1_entities s
        JOIN tgt_counts tc ON s.country_norm = tc.country_norm AND s.name_no_suffix = tc.name_no_suffix
        JOIN target_entities tgt ON s.country_norm = tgt.country_norm AND s.name_no_suffix = tgt.name_no_suffix;
    """)

    con.execute("""
        CREATE TEMP TABLE ch_sib_name AS 
        WITH captured AS (
            SELECT c.source1_entity_id, tgt.target_source, tgt.country_norm, tgt.name_clean
            FROM v3_captured_pairs c
            JOIN target_entities tgt ON c.target_entity_id = tgt.entity_id
            WHERE length(tgt.name_clean) >= 6
        ),
        tgt_counts AS (
            SELECT country_norm, name_clean, count(*) as cnt
            FROM target_entities
            WHERE length(name_clean) >= 6
            GROUP BY 1, 2
            HAVING count(*) <= 5
        )
        SELECT DISTINCT c.source1_entity_id, t2.entity_id as target_entity_id
        FROM captured c
        JOIN tgt_counts tc ON c.country_norm = tc.country_norm AND c.name_clean = tc.name_clean
        JOIN target_entities t2 
          ON c.country_norm = t2.country_norm 
         AND c.name_clean = t2.name_clean 
         AND c.target_source <> t2.target_source;
    """)

    con.execute("""
        CREATE TEMP TABLE ch_sib_addr AS 
        WITH captured AS (
            SELECT c.source1_entity_id, tgt.target_source, tgt.country_norm, tgt.addr_clean
            FROM v3_captured_pairs c
            JOIN target_entities tgt ON c.target_entity_id = tgt.entity_id
            WHERE length(tgt.addr_clean) >= 12
        )
        SELECT DISTINCT c.source1_entity_id, t2.entity_id as target_entity_id
        FROM captured c
        JOIN target_entities t2 
          ON c.country_norm = t2.country_norm 
         AND c.addr_clean = t2.addr_clean 
         AND c.target_source <> t2.target_source;
    """)

    base_pairs = 93171949
    base_cap = 5511986
    base_zero_cap = 130284
    base_oracle = 0.869706

    cum_rows = [{
        "stage": "Base V3",
        "total_candidates": base_pairs,
        "candidates_per_s1": base_pairs / 2206821.0,
        "captured_true_pairs": base_cap,
        "zero_capture_entities": base_zero_cap,
        "candidate_oracle": base_oracle,
        "oracle_gain": 0.0,
        "gain_per_million_cands": 0.0
    }]

    stages = [
        ("V3 + Suffix-Stripped Name", "SELECT * FROM ch_suff"),
        ("V3 + Suffix-Stripped + Sibling Name", "SELECT * FROM ch_suff UNION SELECT * FROM ch_sib_name"),
        ("V3 + Suffix-Stripped + Sibling Name + Sibling Addr", "SELECT * FROM ch_suff UNION SELECT * FROM ch_sib_name UNION SELECT * FROM ch_sib_addr")
    ]

    for s_name, sql_q in stages:
        con.execute(f"CREATE TEMP TABLE cur_stage_added AS SELECT DISTINCT source1_entity_id, target_entity_id FROM ({sql_q});")
        stage_added_vol = con.execute("SELECT count(*) FROM cur_stage_added").fetchone()[0]

        con.execute("""
            CREATE TEMP TABLE cur_stage_rec AS 
            SELECT DISTINCT m.source1_entity_id, m.target_entity_id
            FROM cur_stage_added a
            JOIN missed_gt_pairs m ON a.source1_entity_id = m.source1_entity_id AND a.target_entity_id = m.target_entity_id;
        """)
        rec_cnt = con.execute("SELECT count(*) FROM cur_stage_rec").fetchone()[0]

        zero_rescued = con.execute("""
            SELECT count(distinct r.source1_entity_id)
            FROM cur_stage_rec r
            JOIN s1_capture_status b ON r.source1_entity_id = b.source1_entity_id
            WHERE b.capture_bucket = 'ZERO_CAPTURE';
        """).fetchone()[0]

        stage_oracle = con.execute("""
            WITH add_k AS (
                SELECT source1_entity_id, count(*) as k_add
                FROM cur_stage_rec
                GROUP BY 1
            )
            SELECT avg(
                CASE 
                    WHEN f.A = 0 THEN 1.0
                    WHEN (f.k + coalesce(a.k_add, 0)) = 0 THEN 0.0
                    ELSE (5.0 * (f.k + coalesce(a.k_add, 0))) / (f.A + 4.0 * (f.k + coalesce(a.k_add, 0)))
                END
            )
            FROM s1_fold_gt f
            LEFT JOIN add_k a ON f.source1_entity_id = a.source1_entity_id;
        """).fetchone()[0]

        con.execute("DROP TABLE cur_stage_added;")
        con.execute("DROP TABLE cur_stage_rec;")

        gain = stage_oracle - base_oracle
        cum_rows.append({
            "stage": s_name,
            "total_candidates": base_pairs + stage_added_vol,
            "candidates_per_s1": (base_pairs + stage_added_vol) / 2206821.0,
            "captured_true_pairs": base_cap + rec_cnt,
            "zero_capture_entities": base_zero_cap - zero_rescued,
            "candidate_oracle": stage_oracle,
            "oracle_gain": gain,
            "gain_per_million_cands": (gain / (stage_added_vol / 1_000_000.0)) if stage_added_vol > 0 else 0.0
        })

    df_cum = pd.DataFrame(cum_rows)
    df_cum.to_csv(OUT_CUMULATIVE_TSV, sep="\t", index=False)
    print("Cumulative Simulation Results:")
    print(df_cum.to_string(index=False))
    print(f"Part 7 completed in {time.time()-t0_p7:.2f}s.")

    # =========================================================================
    # PART 8: INDIA-SPECIFIC ROOT CAUSE SUMMARY
    # =========================================================================
    print("\n[PART 8] India-Specific Root Cause Summary...")
    india_root_causes = con.execute("""
        WITH india_missed AS (
            SELECT * FROM missed_pairs_detailed WHERE country = 'INDIA'
        )
        SELECT 
            count(*) as india_missed_pairs,
            count(distinct source1_entity_id) as india_missed_entities,
            avg(case when cat_R_cross_script_indic = 1 then 1.0 else 0.0 end) as indic_script_presence_pct,
            avg(case when cat_B_prefix4_mismatch = 1 and cat_R_cross_script_indic = 0 and cat_E_token_order_variation = 1 then 1.0 else 0.0 end) as token_inversion_pct,
            avg(case when cat_B_prefix4_mismatch = 1 and cat_R_cross_script_indic = 0 and name_jw >= 0.80 then 1.0 else 0.0 end) as spelling_variant_pct,
            avg(case when cat_M_landmark_style = 1 and cat_J_pin_absent = 1 then 1.0 else 0.0 end) as unstructured_landmark_no_pin_pct,
            avg(case when cat_I_pin_shared = 1 then 1.0 else 0.0 end) as shared_pin_present_pct,
            avg(case when cat_J_pin_absent = 1 then 1.0 else 0.0 end) as pin_missing_pct,
            avg(case when cat_D_legal_suffix_variation = 1 then 1.0 else 0.0 end) as suffix_variation_pct
        FROM india_missed;
    """).df().to_dict(orient="records")[0]
    print(json.dumps(india_root_causes, indent=2))

    # =========================================================================
    # WRITE ARTIFACTS
    # =========================================================================
    print("\nWriting JSON Artifacts...")
    with open(OUT_RESIDUAL_JSON, "w", encoding="utf-8") as f:
        json.dump({
            "residual_miss_total": 2126379,
            "overall_summary": p1_summary,
            "by_country": p1_by_country.to_dict(orient="records"),
            "by_target_source": p1_by_source.to_dict(orient="records"),
            "by_capture_bucket": p1_by_bucket.to_dict(orient="records"),
            "zero_capture_forensics": zero_cap_analysis,
            "zero_capture_by_country": zero_cap_by_country.to_dict(orient="records")
        }, f, indent=2)

    with open(OUT_STRUCTURAL_JSON, "w", encoding="utf-8") as f:
        json.dump({
            "raw_exact_stats": raw_exact_stats.to_dict(orient="records"),
            "id_structure_stats": id_struct_stats.to_dict(orient="records"),
            "row_position_stats": row_pos_stats,
            "sibling_recovery_stats": sibling_recovery,
            "test_volume_checks": test_volume_checks,
            "india_root_causes": india_root_causes
        }, f, indent=2)

    print(f"  -> {OUT_RESIDUAL_JSON}")
    print(f"  -> {OUT_CHANNELS_TSV}")
    print(f"  -> {OUT_STRUCTURAL_JSON}")
    print(f"  -> {OUT_CUMULATIVE_TSV}")
    print(f"Entire E1 execution completed in {time.time()-t_global_start:.2f}s.")

    print("\n" + "=" * 80)
    print("E1 RETRIEVAL FORENSIC COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()
