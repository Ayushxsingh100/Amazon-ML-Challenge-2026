import os
import duckdb

repo_root = "c:/NEW AMAZON"
con = duckdb.connect()
con.execute("PRAGMA threads=8;")
con.execute("PRAGMA memory_limit='24GB';")

print("--- Testing Regex Extraction on S1, S2, S3 ---")

s1_path = os.path.join(repo_root, "Amazon-ML-Challenge-2026", "outputs", "person1_step1", "normalized", "train_source1_normalized.tsv")
s2_path = os.path.join(repo_root, "Amazon-ML-Challenge-2026", "outputs", "person1_step1", "normalized", "train_source2_normalized.tsv")
s3_path = os.path.join(repo_root, "Amazon-ML-Challenge-2026", "outputs", "person1_step1", "normalized", "train_source3_normalized.tsv")

con.execute(f"""
CREATE TEMP TABLE s1_sample AS
SELECT * FROM read_csv('{s1_path.replace(os.sep, "/")}', delim='\t', header=True, all_varchar=True)
LIMIT 10000;
""")

# Test domain extraction
domains = con.execute("""
SELECT 
    business_name, 
    business_address,
    regexp_extract(lower(business_name), '([a-z0-9-]+(\\.[a-z0-9-]+)*\\.(com|org|net|in|co|io|biz|info|us|gov))', 1) as dom_name,
    regexp_extract(lower(business_address), '([a-z0-9-]+(\\.[a-z0-9-]+)*\\.(com|org|net|in|co|io|biz|info|us|gov))', 1) as dom_addr,
    regexp_extract(business_address, '(?:^|[^0-9])([0-9]{5,6})(?:[^0-9]|$)', 1) as postcode,
    regexp_extract(business_address, '[0-9]+[A-Za-z]?', 0) as house_number,
    regexp_extract(business_address || ' ' || business_name, '(?:^|[^0-9])([0-9]{10})(?:[^0-9]|$)', 1) as phone_10
FROM s1_sample;
""").df()

print("Sample regex extractions:")
print("Non-empty dom_name:", len(domains[domains['dom_name'] != '']))
print("Non-empty dom_addr:", len(domains[domains['dom_addr'] != '']))
print("Non-empty postcode:", len(domains[domains['postcode'] != '']))
print("Non-empty house_number:", len(domains[domains['house_number'] != '']))
print("Non-empty phone_10:", len(domains[domains['phone_10'] != '']))
