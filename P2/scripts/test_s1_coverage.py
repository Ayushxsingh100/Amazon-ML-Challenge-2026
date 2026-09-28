import os
import time
import duckdb

repo_root = "c:/NEW AMAZON"
con = duckdb.connect()
con.execute("PRAGMA threads=8;")
con.execute("PRAGMA memory_limit='24GB';")

print("--- Testing Shortcut Hunt Logic ---")
t0 = time.time()

s1_path = os.path.join(repo_root, "Amazon-ML-Challenge-2026", "outputs", "person1_step1", "normalized", "train_source1_normalized.tsv")
s2_path = os.path.join(repo_root, "Amazon-ML-Challenge-2026", "outputs", "person1_step1", "normalized", "train_source2_normalized.tsv")
s3_path = os.path.join(repo_root, "Amazon-ML-Challenge-2026", "outputs", "person1_step1", "normalized", "train_source3_normalized.tsv")
gt_path = os.path.join(repo_root, "Amazon-ML-Challenge-2026", "outputs", "person1_step1", "train_ground_truth_reconstructed.tsv")

print("Loading sample of S1 and S2...")
con.execute(f"""
CREATE TEMP TABLE s1 AS
SELECT 
    entity_id as s1_id,
    regexp_replace(lower(strip_accents(coalesce(business_name, ''))), '[^a-z0-9]', '', 'g') as name_norm,
    regexp_replace(lower(strip_accents(coalesce(business_address, ''))), '[^a-z0-9]', '', 'g') as addr_norm,
    regexp_replace(lower(strip_accents(coalesce(country, ''))), '[^a-z0-9]', '', 'g') as country_norm,
    regexp_extract(coalesce(business_address, ''), '(?:^|[^0-9])([0-9]{{5,6}})(?:[^0-9]|$)', 1) as postcode,
    regexp_extract(coalesce(business_address, ''), '[0-9]+[A-Za-z]?', 0) as house,
    regexp_extract(coalesce(business_address, '') || ' ' || coalesce(business_name, ''), '(?:^|[^0-9])([0-9]{{10}})(?:[^0-9]|$)', 1) as phone_10,
    regexp_extract(lower(coalesce(business_name, '') || ' ' || coalesce(business_address, '')), '([a-z0-9-]+(\\.[a-z0-9-]+)*\\.(?:com|org|net|in|co|io|biz|info|us|gov))', 1) as domain
FROM read_csv('{s1_path.replace(os.sep, "/")}', delim='\t', header=True, all_varchar=True);
""")

s1_count = con.execute("SELECT COUNT(*) FROM s1").fetchone()[0]
print(f"Total S1 entities: {s1_count}")

# Check % of S1 rows where each field is non-empty
print("Computing S1 non-empty percentages...")
con.execute("""
CREATE TEMP TABLE s1_features AS
SELECT 
    s1_id,
    name_norm,
    addr_norm,
    country_norm,
    CASE WHEN postcode != '' AND house != '' THEN postcode || '_' || lower(house) ELSE '' END as postcode_house,
    CASE WHEN name_norm != '' AND postcode != '' THEN name_norm || '_' || postcode ELSE '' END as name_postcode,
    CASE WHEN length(phone_10) >= 8 THEN right(phone_10, 8) ELSE '' END as phone_last8,
    domain
FROM s1;
""")

s1_coverage = con.execute("""
SELECT 
    ROUND(COUNT(CASE WHEN name_norm != '' THEN 1 END)::DOUBLE / COUNT(*) * 100, 4) as name_cov,
    ROUND(COUNT(CASE WHEN addr_norm != '' THEN 1 END)::DOUBLE / COUNT(*) * 100, 4) as addr_cov,
    ROUND(COUNT(CASE WHEN country_norm != '' THEN 1 END)::DOUBLE / COUNT(*) * 100, 4) as country_cov,
    ROUND(COUNT(CASE WHEN postcode_house != '' THEN 1 END)::DOUBLE / COUNT(*) * 100, 4) as post_house_cov,
    ROUND(COUNT(CASE WHEN name_postcode != '' THEN 1 END)::DOUBLE / COUNT(*) * 100, 4) as name_post_cov,
    ROUND(COUNT(CASE WHEN phone_last8 != '' THEN 1 END)::DOUBLE / COUNT(*) * 100, 4) as phone_cov,
    ROUND(COUNT(CASE WHEN domain != '' THEN 1 END)::DOUBLE / COUNT(*) * 100, 4) as domain_cov
FROM s1_features;
""").df()

print("S1 Coverage:")
print(s1_coverage.to_string(index=False))
