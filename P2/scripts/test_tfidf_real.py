import os
import time
import duckdb
from sklearn.feature_extraction.text import TfidfVectorizer

repo_root = "c:/NEW AMAZON"
s2_test_path = repo_root + "/Amazon-ML-Challenge-2026/outputs/person1_step1/normalized/test_source2_normalized.tsv"

con = duckdb.connect()
con.execute("PRAGMA threads=8;")

print("Loading 200,000 rows from test_source2...")
t0 = time.time()
df = con.execute(f"""
SELECT 
    entity_id,
    trim(regexp_replace(regexp_replace(lower(strip_accents(coalesce(business_name, '') || ' ' || coalesce(business_address, ''))), '[^a-z0-9 ]', ' ', 'g'), '\\s+', ' ', 'g')) as text
FROM read_csv('{s2_test_path}', delim='\t', header=True, all_varchar=True)
WHERE country = 'FRANCE'
LIMIT 200000;
""").df()
print(f"Loaded {len(df)} rows in {time.time() - t0:.2f}s")

t1 = time.time()
vec = TfidfVectorizer(analyzer='char', ngram_range=(3, 3), min_df=2, dtype='float32')
X = vec.fit_transform(df['text'])
print(f"TF-IDF fit_transform in {time.time() - t1:.2f}s, shape: {X.shape}, nnz: {X.nnz}, mem: {X.data.nbytes / 1e6:.1f} MB")
