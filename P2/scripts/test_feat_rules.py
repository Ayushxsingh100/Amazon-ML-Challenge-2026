import os
import time
import duckdb

REPO = r"c:\NEW AMAZON\Amazon-ML-Challenge-2026"
FEAT_S2 = os.path.join(REPO, "P2", "data", "features", "cands_BCD_v1_features_train_s2.tsv")

con = duckdb.connect()
con.execute("SET threads=4")
con.execute("SET memory_limit='4GB'")

t0 = time.time()
print("Scanning sample of features...")
res = con.execute(f"""
    SELECT 
        count(*),
        sum(case when address_exact_match=1 and name_exact_match=0 and (first_token_match=0 or address_first_number_match=0) and (prefix4_match=0 or address_first_number_match=0) then 1 else 0 end) as b_only,
        sum(case when address_exact_match=0 and name_exact_match=1 and (first_token_match=0 or address_first_number_match=0) and (prefix4_match=0 or address_first_number_match=0) then 1 else 0 end) as d_only,
        sum(case when address_exact_match=0 and name_exact_match=0 and first_token_match=1 and address_first_number_match=1 and prefix4_match=0 then 1 else 0 end) as c_only
    FROM read_csv('{FEAT_S2}', delim='\\t', header=true)
    LIMIT 1000000
""").fetchall()
print(f"Sample res: {res} in {time.time() - t0:.2f}s")
