#!/usr/bin/env python3
"""
P1/scripts/phase7_blocking_ablation.py

Task P1-C & Section 9: Blocking Strategy Ablation
Tests complementary candidate blocking passes on Fold 0 to measure:
- new_candidates_added
- new_GT_pairs_recovered
- incremental_recall
- marginal efficiency (GT recovered per 1k candidates)
- fanout statistics

Outputs:
- P1/reports/phase7_blocking_ablation.tsv
"""

import time
import duckdb
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

con = duckdb.connect()
con.execute("SET threads=4")
con.execute("SET memory_limit='8GB'")
con.execute("SET preserve_insertion_order=false")

print("=" * 80)
print("PHASE 7 TASK P1-C: BLOCKING STRATEGY ABLATION ON FOLD 0")
print("=" * 80)

t0 = time.time()

# 1. Prepare Fold 0 Ground Truth and V3 baseline
print("[1/5] Setting up Fold 0 Ground Truth & V3 candidates...")

con.execute("""
CREATE OR REPLACE TABLE f0_manifest AS
SELECT source1_entity_id 
FROM read_csv('P3/reports/folds_v1_manifest.tsv', delim='\\t', header=true) 
WHERE fold = 0;

CREATE OR REPLACE TABLE f0_gt AS
WITH gt_raw AS (
    SELECT source1_entity_id, UNNEST(STRING_SPLIT(matched_entity_ids, ',')) as target_id
    FROM read_csv('outputs/person1_step1/train_ground_truth_reconstructed.tsv', delim='\\t', header=true)
)
SELECT gt.source1_entity_id, gt.target_id,
       CASE WHEN gt.target_id LIKE 'S2-%' THEN 'S2' ELSE 'S3' END as target_source
FROM gt_raw gt
JOIN f0_manifest m ON gt.source1_entity_id = m.source1_entity_id;

CREATE OR REPLACE TABLE f0_v3_cands AS
WITH v3_s2 AS (
    SELECT c.source1_entity_id, c.matched_entity_id as target_id
    FROM read_csv('P1/data/candidates/v3/train_candidate_pairs_s2_v3.tsv', delim='\\t', header=true) c
    JOIN f0_manifest m ON c.source1_entity_id = m.source1_entity_id
),
v3_s3 AS (
    SELECT c.source1_entity_id, c.matched_entity_id as target_id
    FROM read_csv('P1/data/candidates/v3/train_candidate_pairs_s3_v3.tsv', delim='\\t', header=true) c
    JOIN f0_manifest m ON c.source1_entity_id = m.source1_entity_id
)
SELECT * FROM v3_s2
UNION ALL
SELECT * FROM v3_s3;
""")

f0_gt_total = con.execute("SELECT count(*) FROM f0_gt").fetchone()[0]
f0_v3_count = con.execute("SELECT count(*) FROM f0_v3_cands").fetchone()[0]
f0_v3_recovered = con.execute("""
    SELECT count(*) 
    FROM f0_gt gt
    JOIN f0_v3_cands c ON gt.source1_entity_id = c.source1_entity_id AND gt.target_id = c.target_id
""").fetchone()[0]

v3_recall = f0_v3_recovered / f0_gt_total * 100.0

print(f"Fold 0 GT Total: {f0_gt_total:,}")
print(f"Fold 0 V3 Candidates: {f0_v3_count:,}")
print(f"Fold 0 V3 Captured GT: {f0_v3_recovered:,} ({v3_recall:.4f}% recall)\n")

# Prepare entity tables for Fold 0
print("[2/5] Preparing entity tables with blocking keys...")
con.execute("""
CREATE OR REPLACE TABLE f0_s1 AS
SELECT 
    entity_id,
    country_normalized as country,
    business_name_normalized as name,
    business_address_normalized as addr,
    -- Normalized name without corporate suffixes
    regexp_replace(business_name_normalized, '\\b(inc|llc|ltd|limited|pvt|co|corp|services|company|group)\\b', '', 'g') as name_clean,
    -- First token
    split_part(trim(business_name_normalized), ' ', 1) as token1,
    -- 4-char prefix
    left(trim(business_name_normalized), 4) as prefix4,
    -- Clean address prefix 20 chars
    left(regexp_replace(lower(coalesce(business_address_normalized,'')), '[^a-z0-9]', '', 'g'), 20) as addr_clean20,
    -- Postal code
    regexp_extract(coalesce(business_address_normalized,''), '(?:^|[^0-9])([0-9]{5,6})(?:[^0-9]|$)', 1) as postal
FROM read_parquet('P1/data/entities/train/source1/train_s1_entities.parquet')
WHERE entity_id IN (SELECT source1_entity_id FROM f0_manifest);

CREATE OR REPLACE TABLE targets AS
SELECT 
    entity_id,
    country_normalized as country,
    business_name_normalized as name,
    business_address_normalized as addr,
    regexp_replace(business_name_normalized, '\\b(inc|llc|ltd|limited|pvt|co|corp|services|company|group)\\b', '', 'g') as name_clean,
    split_part(trim(business_name_normalized), ' ', 1) as token1,
    left(trim(business_name_normalized), 4) as prefix4,
    left(regexp_replace(lower(coalesce(business_address_normalized,'')), '[^a-z0-9]', '', 'g'), 20) as addr_clean20,
    regexp_extract(coalesce(business_address_normalized,''), '(?:^|[^0-9])([0-9]{5,6})(?:[^0-9]|$)', 1) as postal
FROM read_parquet('P1/data/entities/train/source2/train_s2_entities.parquet')
UNION ALL
SELECT 
    entity_id,
    country_normalized as country,
    business_name_normalized as name,
    business_address_normalized as addr,
    regexp_replace(business_name_normalized, '\\b(inc|llc|ltd|limited|pvt|co|corp|services|company|group)\\b', '', 'g') as name_clean,
    split_part(trim(business_name_normalized), ' ', 1) as token1,
    left(trim(business_name_normalized), 4) as prefix4,
    left(regexp_replace(lower(coalesce(business_address_normalized,'')), '[^a-z0-9]', '', 'g'), 20) as addr_clean20,
    regexp_extract(coalesce(business_address_normalized,''), '(?:^|[^0-9])([0-9]{5,6})(?:[^0-9]|$)', 1) as postal
FROM read_parquet('P1/data/entities/train/source3/train_s3_entities.parquet');
""")

results = []
results.append({
    "strategy": "V3_BASELINE",
    "total_cands": f0_v3_count,
    "new_cands": 0,
    "recovered_gt": f0_v3_recovered,
    "new_gt": 0,
    "recall": v3_recall,
    "marginal_eff": 0.0
})

def eval_strategy(name: str, query_sql: str, max_fanout: int = 50):
    t_start = time.time()
    print(f"\n--- Testing Strategy: {name} (fanout <= {max_fanout}) ---")
    
    con.execute(f"""
    CREATE OR REPLACE TEMP TABLE strat_raw AS
    {query_sql};
    
    -- Enforce S1 fanout cap
    CREATE OR REPLACE TEMP TABLE strat_filtered AS
    SELECT r.source1_entity_id, r.target_id
    FROM strat_raw r
    JOIN (
        SELECT source1_entity_id 
        FROM strat_raw 
        GROUP BY source1_entity_id 
        HAVING count(*) <= {max_fanout}
    ) k ON r.source1_entity_id = k.source1_entity_id;
    
    -- Deduplicate against existing V3
    CREATE OR REPLACE TEMP TABLE strat_new AS
    SELECT DISTINCT s.source1_entity_id, s.target_id
    FROM strat_filtered s
    LEFT JOIN f0_v3_cands v ON s.source1_entity_id = v.source1_entity_id AND s.target_id = v.target_id
    WHERE v.target_id IS NULL;
    """)
    
    new_cands = con.execute("SELECT count(*) FROM strat_new").fetchone()[0]
    new_gt = con.execute("""
        SELECT count(*)
        FROM f0_gt gt
        JOIN strat_new n ON gt.source1_entity_id = n.source1_entity_id AND gt.target_id = n.target_id
    """).fetchone()[0]
    
    total_cands = f0_v3_count + new_cands
    total_recovered = f0_v3_recovered + new_gt
    new_recall = total_recovered / f0_gt_total * 100.0
    marginal_eff = (new_gt / new_cands * 1000.0) if new_cands > 0 else 0.0
    
    print(f"  New Candidates Added: {new_cands:,}")
    print(f"  New GT Pairs Recovered: {new_gt:,}")
    print(f"  New Total Recall: {new_recall:.4f}% (+{new_recall - v3_recall:.4f}%)")
    print(f"  Marginal Efficiency: {marginal_eff:.2f} GT pairs per 1,000 new candidates")
    print(f"  Elapsed: {time.time() - t_start:.2f}s")
    
    results.append({
        "strategy": name,
        "total_cands": total_cands,
        "new_cands": new_cands,
        "recovered_gt": total_recovered,
        "new_gt": new_gt,
        "recall": new_recall,
        "marginal_eff": marginal_eff
    })

# STRATEGY A: Clean Address Prefix (Address Block for Cross-Script Indic)
# Matches on same country + first 15 chars of alphanumeric address (handles cross-script name variations!)
eval_strategy(
    "STRAT_A_ADDR_CLEAN15",
    """
    SELECT s1.entity_id as source1_entity_id, tgt.entity_id as target_id
    FROM f0_s1 s1
    JOIN targets tgt 
      ON s1.country = tgt.country 
     AND left(s1.addr_clean20, 15) = left(tgt.addr_clean20, 15)
    WHERE length(left(s1.addr_clean20, 15)) >= 12
    """,
    max_fanout=30
)

# STRATEGY B: Clean Name Core without House (for Missing/Different House)
# Matches on country + name without corporate suffixes (length >= 8)
eval_strategy(
    "STRAT_B_NAME_CLEAN_NO_HOUSE",
    """
    SELECT s1.entity_id as source1_entity_id, tgt.entity_id as target_id
    FROM f0_s1 s1
    JOIN targets tgt 
      ON s1.country = tgt.country 
     AND regexp_replace(lower(s1.name_clean), '[^a-z0-9]', '', 'g') = regexp_replace(lower(tgt.name_clean), '[^a-z0-9]', '', 'g')
    WHERE length(regexp_replace(lower(s1.name_clean), '[^a-z0-9]', '', 'g')) >= 8
    """,
    max_fanout=30
)

# STRATEGY C: Postal Code + 4-char Name Prefix
# Matches on country + postal code (len >= 5) + first 4 chars of name
eval_strategy(
    "STRAT_C_POSTAL_PREFIX4",
    """
    SELECT s1.entity_id as source1_entity_id, tgt.entity_id as target_id
    FROM f0_s1 s1
    JOIN targets tgt 
      ON s1.country = tgt.country 
     AND s1.postal = tgt.postal
     AND s1.prefix4 = tgt.prefix4
    WHERE length(s1.postal) >= 5 AND length(s1.prefix4) = 4
    """,
    max_fanout=30
)

# Export Summary TSV
print("\n[5/5] Exporting ablation summary...")
import csv
out_tsv = REPO_ROOT / "P1" / "reports" / "phase7_blocking_ablation.tsv"
with open(out_tsv, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["strategy", "total_cands", "new_cands", "recovered_gt", "new_gt", "recall", "marginal_eff"], delimiter="\t")
    writer.writeheader()
    for r in results:
        writer.writerow(r)

print(f"Saved ablation report to: {out_tsv}")
print(f"Total ablation runtime: {time.time()-t0:.2f}s")
