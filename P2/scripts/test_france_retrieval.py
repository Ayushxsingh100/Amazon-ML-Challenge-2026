import os
import time
import duckdb
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

repo_root = "c:/NEW AMAZON"
s1_test_path = repo_root + "/Amazon-ML-Challenge-2026/outputs/person1_step1/normalized/test_source1_normalized.tsv"
s2_test_path = repo_root + "/Amazon-ML-Challenge-2026/outputs/person1_step1/normalized/test_source2_normalized.tsv"
s3_test_path = repo_root + "/Amazon-ML-Challenge-2026/outputs/person1_step1/normalized/test_source3_normalized.tsv"
v5_path = repo_root + "/Amazon-ML-Challenge-2026/P2/data/matching_results_v5.tsv"

con = duckdb.connect()
con.execute("PRAGMA threads=8;")
con.execute("PRAGMA memory_limit='24GB';")

print("--- Testing Retrieval for France ---")
t0 = time.time()

# 1. Load France S1 targets searching S3 (23,637 targets)
df_q = con.execute(f"""
WITH v5_targets AS (
    SELECT 
        v.source1_entity_id as s1_id,
        v.matched_entity_ids as matched_ids
    FROM read_csv('{v5_path}', delim='\t', header=True, all_varchar=True) v
    WHERE v.matched_entity_ids IS NOT NULL AND TRIM(v.matched_entity_ids) != ''
      AND v.matched_entity_ids LIKE '%S2-%' AND v.matched_entity_ids NOT LIKE '%S3-%'
)
SELECT 
    s.entity_id as s1_id,
    trim(regexp_replace(regexp_replace(lower(strip_accents(coalesce(s.business_name, '') || ' ' || coalesce(s.business_address, ''))), '[^a-z0-9 ]', ' ', 'g'), '\\s+', ' ', 'g')) as text
FROM read_csv('{s1_test_path}', delim='\t', header=True, all_varchar=True) s
JOIN v5_targets t ON s.entity_id = t.s1_id
WHERE s.country = 'FRANCE';
""").df()

print(f"Loaded {len(df_q)} query targets in {time.time() - t0:.2f}s")

# 2. Load France S3 docs (731,615 docs)
t1 = time.time()
df_d = con.execute(f"""
SELECT 
    entity_id as s3_id,
    trim(regexp_replace(regexp_replace(lower(strip_accents(coalesce(business_name, '') || ' ' || coalesce(business_address, ''))), '[^a-z0-9 ]', ' ', 'g'), '\\s+', ' ', 'g')) as text
FROM read_csv('{s3_test_path}', delim='\t', header=True, all_varchar=True)
WHERE country = 'FRANCE';
""").df()
print(f"Loaded {len(df_d)} doc targets in {time.time() - t1:.2f}s")

# 3. Fit TF-IDF on S3 docs
t2 = time.time()
vec = TfidfVectorizer(analyzer='char', ngram_range=(3, 3), min_df=2, dtype=np.float32)
D = vec.fit_transform(df_d['text'])
print(f"Fit TF-IDF on S3 docs in {time.time() - t2:.2f}s, shape: {D.shape}")

# Transpose D for cosine similarity dot product
Dt = D.T.tocsc()

# 4. Transform queries and retrieve top 5
t3 = time.time()
Q = vec.transform(df_q['text'])
print(f"Transformed queries in {time.time() - t3:.2f}s, shape: {Q.shape}")

# Chunk queries in batches of 2000
chunk_size = 2000
num_q = Q.shape[0]
pairs = []

t4 = time.time()
for start_idx in range(0, num_q, chunk_size):
    end_idx = min(start_idx + chunk_size, num_q)
    Q_chunk = Q[start_idx:end_idx]
    sims = Q_chunk @ Dt  # csr matrix
    
    # Extract top 5 for each row
    for i in range(end_idx - start_idx):
        q_id = df_q['s1_id'].iloc[start_idx + i]
        s = sims.indptr[i]
        e = sims.indptr[i+1]
        data = sims.data[s:e]
        indices = sims.indices[s:e]
        if len(data) == 0:
            continue
        if len(data) <= 5:
            top_cand_indices = indices[np.argsort(-data)]
        else:
            part = np.argpartition(-data, 5)[:5]
            top_cand_indices = indices[part[np.argsort(-data[part])]]
        
        for cand_idx in top_cand_indices:
            cand_id = df_d['s3_id'].iloc[cand_idx]
            pairs.append((q_id, cand_id))

print(f"Retrieved {len(pairs)} pairs in {time.time() - t4:.2f}s")
print(f"Total time for France S3: {time.time() - t0:.2f}s")
