#!/usr/bin/env python3
"""
P1/scripts/phase7_missed_gt_analysis.py

Task P1-A: Complete Missed Ground-Truth Forensics
Categorizes all 2,126,379 missed ground truth pairs from V3 candidate generation.
Outputs:
- P1/reports/phase7_missed_gt_summary.json
- P1/reports/phase7_missed_gt_breakdown.tsv
"""

import os
import sys
import time
import json
import duckdb
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

con = duckdb.connect()
con.execute("SET threads=4")
con.execute("SET memory_limit='8GB'")
con.execute("SET preserve_insertion_order=false")

print("=" * 70)
print("PHASE 7 TASK P1-A: MISSED GROUND TRUTH FORENSICS")
print("=" * 70)

t0 = time.time()

# 1. Identify Missed GT for S2 and S3
print("[1/4] Extracting missed ground truth pairs...")

con.execute("""
CREATE OR REPLACE TABLE missed_gt_pairs AS
WITH gt_raw AS (
    SELECT source1_entity_id, UNNEST(STRING_SPLIT(matched_entity_ids, ',')) as target_id
    FROM read_csv('outputs/person1_step1/train_ground_truth_reconstructed.tsv', delim='\\t', header=true)
),
cands_s2 AS (
    SELECT DISTINCT source1_entity_id, matched_entity_id as target_id
    FROM read_csv('P1/data/candidates/v3/train_candidate_pairs_s2_v3.tsv', delim='\\t', header=true)
),
cands_s3 AS (
    SELECT DISTINCT source1_entity_id, matched_entity_id as target_id
    FROM read_csv('P1/data/candidates/v3/train_candidate_pairs_s3_v3.tsv', delim='\\t', header=true)
),
all_cands AS (
    SELECT * FROM cands_s2
    UNION ALL
    SELECT * FROM cands_s3
)
SELECT gt.source1_entity_id, gt.target_id,
       CASE WHEN gt.target_id LIKE 'S2-%' THEN 'S2' ELSE 'S3' END AS target_source
FROM gt_raw gt
LEFT JOIN all_cands c ON gt.source1_entity_id = c.source1_entity_id AND gt.target_id = c.target_id
WHERE c.target_id IS NULL;
""")

missed_count = con.execute("SELECT count(*) FROM missed_gt_pairs").fetchone()[0]
print(f"Total missed GT pairs: {missed_count:,} (Identified in {time.time()-t0:.2f}s)")

# 2. Join with Entity Attributes
print("[2/4] Joining with canonical entity tables...")
con.execute("""
CREATE OR REPLACE TABLE missed_gt_enriched AS
WITH s1_ent AS (
    SELECT entity_id, business_name_normalized as s1_name, business_address_normalized as s1_addr, country_normalized as s1_country
    FROM read_parquet('P1/data/entities/train/source1/train_s1_entities.parquet')
),
tgt_ent AS (
    SELECT entity_id, business_name_normalized as tgt_name, business_address_normalized as tgt_addr, country_normalized as tgt_country
    FROM read_parquet('P1/data/entities/train/source2/train_s2_entities.parquet')
    UNION ALL
    SELECT entity_id, business_name_normalized as tgt_name, business_address_normalized as tgt_addr, country_normalized as tgt_country
    FROM read_parquet('P1/data/entities/train/source3/train_s3_entities.parquet')
)
SELECT 
    m.source1_entity_id,
    m.target_id,
    m.target_source,
    s1.s1_name,
    tgt.tgt_name,
    s1.s1_addr,
    tgt.tgt_addr,
    s1.s1_country,
    tgt.tgt_country
FROM missed_gt_pairs m
JOIN s1_ent s1 ON m.source1_entity_id = s1.entity_id
JOIN tgt_ent tgt ON m.target_id = tgt.entity_id;
""")

# 3. Categorize Missed Pairs Deterministically
print("[3/4] Categorizing failure mechanisms...")
con.execute("""
CREATE OR REPLACE TABLE categorized_misses AS
SELECT 
    source1_entity_id,
    target_id,
    target_source,
    s1_country,
    tgt_country,
    CASE
        -- 1. Missing data
        WHEN s1_name IS NULL OR s1_name = '' OR tgt_name IS NULL OR tgt_name = '' THEN 'MISSING_NAME'
        WHEN s1_addr IS NULL OR s1_addr = '' OR tgt_addr IS NULL OR tgt_addr = '' THEN 'MISSING_ADDRESS'
        
        -- 2. Cross-Script / Non-Latin Indic Characters
        -- Check if either name contains non-ascii characters (> chr(127))
        WHEN regexp_matches(s1_name, '[^\\x00-\\x7F]') OR regexp_matches(tgt_name, '[^\\x00-\\x7F]') 
          OR regexp_matches(s1_addr, '[^\\x00-\\x7F]') OR regexp_matches(tgt_addr, '[^\\x00-\\x7F]') THEN 'CROSS_SCRIPT_INDIC'
        
        -- 3. Name Match but House Number Mismatch or Missing
        WHEN (LEFT(s1_name, 4) = LEFT(tgt_name, 4) AND LENGTH(s1_name) >= 4)
             OR (SPLIT_PART(s1_name, ' ', 1) = SPLIT_PART(tgt_name, ' ', 1) AND s1_name <> '') THEN
            CASE 
                WHEN regexp_extract(s1_addr, '[0-9]+[A-Za-z]?', 0) = '' OR regexp_extract(tgt_addr, '[0-9]+[A-Za-z]?', 0) = '' THEN 'SAME_NAME_MISSING_HOUSE'
                WHEN regexp_extract(s1_addr, '[0-9]+[A-Za-z]?', 0) <> regexp_extract(tgt_addr, '[0-9]+[A-Za-z]?', 0) THEN 'SAME_NAME_DIFF_HOUSE'
                ELSE 'SAME_NAME_BLOCKED_BY_FANOUT'
            END
            
        -- 4. High Address Similarity but Name Variation
        WHEN jaro_winkler_similarity(COALESCE(s1_addr, ''), COALESCE(tgt_addr, '')) >= 80.0 THEN 'ADDRESS_MATCH_NAME_VARIATION'
        
        -- 5. High Name Similarity but Relaxed Address
        WHEN jaro_winkler_similarity(COALESCE(s1_name, ''), COALESCE(tgt_name, '')) >= 80.0 THEN 'NAME_SIMILAR_ADDRESS_MISMATCH'
        
        -- 6. Token Overlap in Name (Reordered / Suffix variation)
        WHEN jaccard(COALESCE(s1_name, ''), COALESCE(tgt_name, '')) >= 0.40 THEN 'NAME_TOKEN_OVERLAP'
        
        -- 7. Other / Severe variation
        ELSE 'SEVERE_VARIATION_OTHER'
    END AS failure_category
FROM missed_gt_enriched;
""")

# 4. Generate Reports and Summaries
print("[4/4] Writing failure breakdown report...")
breakdown = con.execute("""
SELECT 
    failure_category,
    COUNT(*) as missed_count,
    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM categorized_misses), 2) as pct,
    SUM(CASE WHEN target_source = 'S2' THEN 1 ELSE 0 END) as s2_count,
    SUM(CASE WHEN target_source = 'S3' THEN 1 ELSE 0 END) as s3_count
FROM categorized_misses
GROUP BY failure_category
ORDER BY missed_count DESC
""").fetchall()

print("\n" + "=" * 80)
print(f"{'FAILURE CATEGORY':<35} | {'COUNT':<10} | {'PCT (%)':<8} | {'S2':<9} | {'S3':<9}")
print("-" * 80)
for cat, cnt, pct, s2_c, s3_c in breakdown:
    print(f"{cat:<35} | {cnt:<10,} | {pct:<8.2f} | {s2_c:<9,} | {s3_c:<9,}")
print("=" * 80)

# Export summary TSV
out_tsv = REPO_ROOT / "P1" / "reports" / "phase7_missed_gt_breakdown.tsv"
con.execute(f"""
COPY (
    SELECT 
        failure_category,
        COUNT(*) as missed_count,
        ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM categorized_misses), 2) as pct,
        SUM(CASE WHEN target_source = 'S2' THEN 1 ELSE 0 END) as s2_count,
        SUM(CASE WHEN target_source = 'S3' THEN 1 ELSE 0 END) as s3_count
    FROM categorized_misses
    GROUP BY failure_category
    ORDER BY missed_count DESC
) TO '{out_tsv}' (HEADER TRUE, DELIMITER '\\t');
""")
print(f"\nSaved failure breakdown to: {out_tsv}")

# Export sample TSV (first 5,000 for forensic inspection)
sample_tsv = REPO_ROOT / "P1" / "reports" / "phase7_missed_gt_sample5k.tsv"
con.execute(f"""
COPY (
    SELECT 
        c.source1_entity_id,
        c.target_id,
        c.target_source,
        c.failure_category,
        e.s1_name,
        e.tgt_name,
        e.s1_addr,
        e.tgt_addr,
        e.s1_country,
        e.tgt_country
    FROM categorized_misses c
    JOIN missed_gt_enriched e ON c.source1_entity_id = e.source1_entity_id AND c.target_id = e.target_id
    LIMIT 5000
) TO '{sample_tsv}' (HEADER TRUE, DELIMITER '\\t');
""")
print(f"Saved 5,000 forensic missed pairs to: {sample_tsv}")

# Export full analysis TSV required by Section 5
full_tsv = REPO_ROOT / "P1" / "reports" / "phase7_missed_gt_analysis.tsv"
print(f"Exporting full 2.12M missed GT analysis to {full_tsv}...")
con.execute(f"""
COPY (
    SELECT 
        c.source1_entity_id,
        c.target_id,
        c.target_source,
        c.failure_category,
        e.s1_name,
        e.tgt_name,
        e.s1_addr,
        e.tgt_addr,
        e.s1_country,
        e.tgt_country
    FROM categorized_misses c
    JOIN missed_gt_enriched e ON c.source1_entity_id = e.source1_entity_id AND c.target_id = e.target_id
) TO '{full_tsv}' (HEADER TRUE, DELIMITER '\\t');
""")
print(f"Saved full missed GT analysis to: {full_tsv}")

print(f"Total analysis runtime: {time.time()-t0:.2f}s")

