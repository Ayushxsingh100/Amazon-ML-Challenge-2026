import os
import time
import duckdb

repo_root = "c:/NEW AMAZON"
con = duckdb.connect()
con.execute("PRAGMA threads=8;")
con.execute("PRAGMA memory_limit='24GB';")

print("--- Testing True Pairs Evaluation ---")
t0 = time.time()

s1_path = os.path.join(repo_root, "Amazon-ML-Challenge-2026", "outputs", "person1_step1", "normalized", "train_source1_normalized.tsv")
s2_path = os.path.join(repo_root, "Amazon-ML-Challenge-2026", "outputs", "person1_step1", "normalized", "train_source2_normalized.tsv")
s3_path = os.path.join(repo_root, "Amazon-ML-Challenge-2026", "outputs", "person1_step1", "normalized", "train_source3_normalized.tsv")
gt_path = os.path.join(repo_root, "Amazon-ML-Challenge-2026", "outputs", "person1_step1", "train_ground_truth_reconstructed.tsv")

con.execute(f"""
CREATE TEMP TABLE s1 AS
SELECT 
    entity_id as s1_id,
    regexp_replace(lower(strip_accents(coalesce(business_name, ''))), '[^a-z0-9]', '', 'g') as name,
    regexp_replace(lower(strip_accents(coalesce(business_address, ''))), '[^a-z0-9]', '', 'g') as addr,
    regexp_replace(lower(strip_accents(coalesce(country, ''))), '[^a-z0-9]', '', 'g') as country,
    regexp_extract(coalesce(business_address, ''), '(?:^|[^0-9])([0-9]{{5,6}})(?:[^0-9]|$)', 1) as postcode,
    regexp_extract(coalesce(business_address, ''), '[0-9]+[A-Za-z]?', 0) as house,
    regexp_extract(coalesce(business_address, '') || ' ' || coalesce(business_name, ''), '(?:^|[^0-9])([0-9]{{10}})(?:[^0-9]|$)', 1) as phone_10,
    regexp_extract(lower(coalesce(business_name, '') || ' ' || coalesce(business_address, '')), '([a-z0-9-]+(\\.[a-z0-9-]+)*\\.(?:com|org|net|in|co|io|biz|info|us|gov))', 1) as domain
FROM read_csv('{s1_path.replace(os.sep, "/")}', delim='\t', header=True, all_varchar=True);
""")

con.execute("""
CREATE TEMP TABLE s1_feat AS
SELECT 
    s1_id,
    name,
    addr,
    country,
    CASE WHEN postcode != '' AND house != '' THEN postcode || '_' || lower(house) ELSE '' END as post_house,
    CASE WHEN name != '' AND postcode != '' THEN name || '_' || postcode ELSE '' END as name_post,
    CASE WHEN length(phone_10) >= 8 THEN right(phone_10, 8) ELSE '' END as phone_last8,
    domain
FROM s1;
""")

print("Loading target S2 and S3...")
con.execute(f"""
CREATE TEMP TABLE target_feat AS
SELECT 
    entity_id as tgt_id,
    regexp_replace(lower(strip_accents(coalesce(business_name, ''))), '[^a-z0-9]', '', 'g') as name,
    regexp_replace(lower(strip_accents(coalesce(business_address, ''))), '[^a-z0-9]', '', 'g') as addr,
    regexp_replace(lower(strip_accents(coalesce(country, ''))), '[^a-z0-9]', '', 'g') as country,
    CASE 
        WHEN regexp_extract(coalesce(business_address, ''), '(?:^|[^0-9])([0-9]{{5,6}})(?:[^0-9]|$)', 1) != '' 
         AND regexp_extract(coalesce(business_address, ''), '[0-9]+[A-Za-z]?', 0) != ''
        THEN regexp_extract(coalesce(business_address, ''), '(?:^|[^0-9])([0-9]{{5,6}})(?:[^0-9]|$)', 1) || '_' || lower(regexp_extract(coalesce(business_address, ''), '[0-9]+[A-Za-z]?', 0))
        ELSE ''
    END as post_house,
    CASE 
        WHEN regexp_replace(lower(strip_accents(coalesce(business_name, ''))), '[^a-z0-9]', '', 'g') != ''
         AND regexp_extract(coalesce(business_address, ''), '(?:^|[^0-9])([0-9]{{5,6}})(?:[^0-9]|$)', 1) != ''
        THEN regexp_replace(lower(strip_accents(coalesce(business_name, ''))), '[^a-z0-9]', '', 'g') || '_' || regexp_extract(coalesce(business_address, ''), '(?:^|[^0-9])([0-9]{{5,6}})(?:[^0-9]|$)', 1)
        ELSE ''
    END as name_post,
    CASE 
        WHEN length(regexp_extract(coalesce(business_address, '') || ' ' || coalesce(business_name, ''), '(?:^|[^0-9])([0-9]{{10}})(?:[^0-9]|$)', 1)) >= 8
        THEN right(regexp_extract(coalesce(business_address, '') || ' ' || coalesce(business_name, ''), '(?:^|[^0-9])([0-9]{{10}})(?:[^0-9]|$)', 1), 8)
        ELSE ''
    END as phone_last8,
    regexp_extract(lower(coalesce(business_name, '') || ' ' || coalesce(business_address, '')), '([a-z0-9-]+(\\.[a-z0-9-]+)*\\.(?:com|org|net|in|co|io|biz|info|us|gov))', 1) as domain
FROM read_csv('{s2_path.replace(os.sep, "/")}', delim='\t', header=True, all_varchar=True)
UNION ALL
SELECT 
    entity_id as tgt_id,
    regexp_replace(lower(strip_accents(coalesce(business_name, ''))), '[^a-z0-9]', '', 'g') as name,
    regexp_replace(lower(strip_accents(coalesce(business_address, ''))), '[^a-z0-9]', '', 'g') as addr,
    regexp_replace(lower(strip_accents(coalesce(country, ''))), '[^a-z0-9]', '', 'g') as country,
    CASE 
        WHEN regexp_extract(coalesce(business_address, ''), '(?:^|[^0-9])([0-9]{{5,6}})(?:[^0-9]|$)', 1) != '' 
         AND regexp_extract(coalesce(business_address, ''), '[0-9]+[A-Za-z]?', 0) != ''
        THEN regexp_extract(coalesce(business_address, ''), '(?:^|[^0-9])([0-9]{{5,6}})(?:[^0-9]|$)', 1) || '_' || lower(regexp_extract(coalesce(business_address, ''), '[0-9]+[A-Za-z]?', 0))
        ELSE ''
    END as post_house,
    CASE 
        WHEN regexp_replace(lower(strip_accents(coalesce(business_name, ''))), '[^a-z0-9]', '', 'g') != ''
         AND regexp_extract(coalesce(business_address, ''), '(?:^|[^0-9])([0-9]{{5,6}})(?:[^0-9]|$)', 1) != ''
        THEN regexp_replace(lower(strip_accents(coalesce(business_name, ''))), '[^a-z0-9]', '', 'g') || '_' || regexp_extract(coalesce(business_address, ''), '(?:^|[^0-9])([0-9]{{5,6}})(?:[^0-9]|$)', 1)
        ELSE ''
    END as name_post,
    CASE 
        WHEN length(regexp_extract(coalesce(business_address, '') || ' ' || coalesce(business_name, ''), '(?:^|[^0-9])([0-9]{{10}})(?:[^0-9]|$)', 1)) >= 8
        THEN right(regexp_extract(coalesce(business_address, '') || ' ' || coalesce(business_name, ''), '(?:^|[^0-9])([0-9]{{10}})(?:[^0-9]|$)', 1), 8)
        ELSE ''
    END as phone_last8,
    regexp_extract(lower(coalesce(business_name, '') || ' ' || coalesce(business_address, '')), '([a-z0-9-]+(\\.[a-z0-9-]+)*\\.(?:com|org|net|in|co|io|biz|info|us|gov))', 1) as domain
FROM read_csv('{s3_path.replace(os.sep, "/")}', delim='\t', header=True, all_varchar=True);
""")

print(f"Loaded target entities in {time.time() - t0:.2f}s")

# Load GT pairs
print("Exploding ground truth...")
con.execute(f"""
CREATE TEMP TABLE gt_exploded AS
WITH split_pairs AS (
    SELECT 
        source1_entity_id as s1_id,
        unnest(string_split(matched_entity_ids, ',')) as tgt_id
    FROM read_csv('{gt_path.replace(os.sep, "/")}', delim='\t', header=True, all_varchar=True)
    WHERE matched_entity_ids IS NOT NULL AND TRIM(matched_entity_ids) != ''
)
SELECT s1_id, TRIM(tgt_id) as tgt_id
FROM split_pairs;
""")

gt_total = con.execute("SELECT COUNT(*) FROM gt_exploded").fetchone()[0]
print(f"Total True Pairs: {gt_total}")

# Join GT with S1 and Target
print("Evaluating True Pairs...")
true_stats = con.execute("""
WITH joined AS (
    SELECT 
        g.s1_id,
        g.tgt_id,
        CASE WHEN s.name != '' AND s.name = t.name THEN 1 ELSE 0 END as eq_name,
        CASE WHEN s.addr != '' AND s.addr = t.addr THEN 1 ELSE 0 END as eq_addr,
        CASE WHEN s.country != '' AND s.country = t.country THEN 1 ELSE 0 END as eq_country,
        CASE WHEN s.post_house != '' AND s.post_house = t.post_house THEN 1 ELSE 0 END as eq_post_house,
        CASE WHEN s.name_post != '' AND s.name_post = t.name_post THEN 1 ELSE 0 END as eq_name_post,
        CASE WHEN s.phone_last8 != '' AND s.phone_last8 = t.phone_last8 THEN 1 ELSE 0 END as eq_phone,
        CASE WHEN s.domain != '' AND s.domain = t.domain THEN 1 ELSE 0 END as eq_domain
    FROM gt_exploded g
    JOIN s1_feat s ON g.s1_id = s.s1_id
    JOIN target_feat t ON g.tgt_id = t.tgt_id
)
SELECT 
    COUNT(*) as total_true_evaluated,
    ROUND(SUM(eq_name)::DOUBLE / COUNT(*) * 100, 4) as true_pct_name,
    ROUND(SUM(eq_addr)::DOUBLE / COUNT(*) * 100, 4) as true_pct_addr,
    ROUND(SUM(eq_country)::DOUBLE / COUNT(*) * 100, 4) as true_pct_country,
    ROUND(SUM(eq_post_house)::DOUBLE / COUNT(*) * 100, 4) as true_pct_post_house,
    ROUND(SUM(eq_name_post)::DOUBLE / COUNT(*) * 100, 4) as true_pct_name_post,
    ROUND(SUM(eq_phone)::DOUBLE / COUNT(*) * 100, 4) as true_pct_phone,
    ROUND(SUM(eq_domain)::DOUBLE / COUNT(*) * 100, 4) as true_pct_domain
FROM joined;
""").df()

print("True Pairs Stats:")
print(true_stats.to_string(index=False))

# Now evaluate Non-Matching V3 Candidates:
# We take a sample of e.g. 5,000,000 non-matching candidate pairs (or fold 0 negatives)
print("\nEvaluating Non-Matching V3 Candidates (Sample of 5,000,000 from OOF fold 0)...")
neg_stats = con.execute("""
WITH neg_sample AS (
    SELECT source1_entity_id as s1_id, candidate_entity_id as tgt_id
    FROM 'E:/predictions/phase4/v3_train_oof_fold0.parquet'
    WHERE label = 0
    USING SAMPLE 5000000
),
joined_neg AS (
    SELECT 
        n.s1_id,
        n.tgt_id,
        CASE WHEN s.name != '' AND s.name = t.name THEN 1 ELSE 0 END as eq_name,
        CASE WHEN s.addr != '' AND s.addr = t.addr THEN 1 ELSE 0 END as eq_addr,
        CASE WHEN s.country != '' AND s.country = t.country THEN 1 ELSE 0 END as eq_country,
        CASE WHEN s.post_house != '' AND s.post_house = t.post_house THEN 1 ELSE 0 END as eq_post_house,
        CASE WHEN s.name_post != '' AND s.name_post = t.name_post THEN 1 ELSE 0 END as eq_name_post,
        CASE WHEN s.phone_last8 != '' AND s.phone_last8 = t.phone_last8 THEN 1 ELSE 0 END as eq_phone,
        CASE WHEN s.domain != '' AND s.domain = t.domain THEN 1 ELSE 0 END as eq_domain
    FROM neg_sample n
    JOIN s1_feat s ON n.s1_id = s.s1_id
    JOIN target_feat t ON n.tgt_id = t.tgt_id
)
SELECT 
    COUNT(*) as total_neg_evaluated,
    ROUND(SUM(eq_name)::DOUBLE / COUNT(*) * 100, 4) as neg_pct_name,
    ROUND(SUM(eq_addr)::DOUBLE / COUNT(*) * 100, 4) as neg_pct_addr,
    ROUND(SUM(eq_country)::DOUBLE / COUNT(*) * 100, 4) as neg_pct_country,
    ROUND(SUM(eq_post_house)::DOUBLE / COUNT(*) * 100, 4) as neg_pct_post_house,
    ROUND(SUM(eq_name_post)::DOUBLE / COUNT(*) * 100, 4) as neg_pct_name_post,
    ROUND(SUM(eq_phone)::DOUBLE / COUNT(*) * 100, 4) as neg_pct_phone,
    ROUND(SUM(eq_domain)::DOUBLE / COUNT(*) * 100, 4) as neg_pct_domain
FROM joined_neg;
""").df()

print("Non-Matching Pairs Stats:")
print(neg_stats.to_string(index=False))

print(f"\nElapsed time: {time.time() - t0:.2f}s")
