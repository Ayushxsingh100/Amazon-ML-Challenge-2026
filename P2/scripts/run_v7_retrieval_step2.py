import os
import re
import time
import unicodedata
import duckdb
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
import sparse_dot_topn as sdt

# Paths
REPO_ROOT = "c:/NEW AMAZON"
P1_OUTPUTS = os.path.join(REPO_ROOT, "Amazon-ML-Challenge-2026", "outputs", "person1_step1", "normalized")
P2_DIR = os.path.join(REPO_ROOT, "Amazon-ML-Challenge-2026", "P2")

S1_PATH = os.path.join(P1_OUTPUTS, "test_source1_normalized.tsv")
S2_PATH = os.path.join(P1_OUTPUTS, "test_source2_normalized.tsv")
S3_PATH = os.path.join(P1_OUTPUTS, "test_source3_normalized.tsv")
V5_PATH = os.path.join(P2_DIR, "data", "matching_results_v5.tsv")
V3_S2_PRED = os.path.join(P2_DIR, "predictions", "phase4", "test_predictions_s2.tsv")
V3_S3_PRED = os.path.join(P2_DIR, "predictions", "phase4", "test_predictions_s3.tsv")

# Regex & Normalizer
LEGAL_RE = re.compile(r'\b(?:ltd|llc|inc|pvt|gmbh|sarl|sas|sa|srl|bv|co)\b')
NON_ALPHANUM_RE = re.compile(r'[^a-z0-9 ]')

def normalize_business_name(name):
    if not name or not isinstance(name, str):
        return ""
    norm = unicodedata.normalize('NFKD', name)
    s = ''.join(c for c in norm if not unicodedata.combining(c)).lower()
    s = s.replace('.', '')
    s = NON_ALPHANUM_RE.sub(' ', s)
    s = LEGAL_RE.sub(' ', s)
    return ' '.join(s.split())

def run_retrieval_for_country_source(country, missing_source, con):
    print(f"\n==========================================")
    print(f"Retrieval for Country: {country}, Missing Source: {missing_source}")
    print(f"==========================================")
    t0 = time.time()
    
    # 1. Determine target condition
    if missing_source == 'S3':
        cond = "matched_entity_ids LIKE '%S2-%' AND matched_entity_ids NOT LIKE '%S3-%'"
        doc_path = S3_PATH
        v3_pred_path = V3_S3_PRED
    elif missing_source == 'S2':
        cond = "matched_entity_ids LIKE '%S3-%' AND matched_entity_ids NOT LIKE '%S2-%'"
        doc_path = S2_PATH
        v3_pred_path = V3_S2_PRED
    else:
        raise ValueError(f"Unknown missing source: {missing_source}")

    # 2. Load Queries (Target S1s)
    t_load = time.time()
    df_queries = con.execute(f"""
    WITH targets AS (
        SELECT source1_entity_id as s1_id
        FROM read_csv('{V5_PATH}', delim='\t', header=True, all_varchar=True)
        WHERE matched_entity_ids IS NOT NULL AND TRIM(matched_entity_ids) != ''
          AND {cond}
    )
    SELECT s.entity_id as s1_id, s.business_name
    FROM read_csv('{S1_PATH}', delim='\t', header=True, all_varchar=True) s
    JOIN targets t ON s.entity_id = t.s1_id
    WHERE s.country = '{country}';
    """).df()
    print(f"Loaded {len(df_queries):,} query targets in {time.time() - t_load:.2f}s")
    
    if len(df_queries) == 0:
        return []

    # 3. Load Documents (Missing Source)
    t_doc = time.time()
    df_docs = con.execute(f"""
    SELECT entity_id as cand_id, business_name
    FROM read_csv('{doc_path}', delim='\t', header=True, all_varchar=True)
    WHERE country = '{country}';
    """).df()
    print(f"Loaded {len(df_docs):,} doc records in {time.time() - t_doc:.2f}s")

    # 4. Normalize business_name
    t_norm = time.time()
    docs_clean = [normalize_business_name(x) for x in df_docs['business_name']]
    queries_clean = [normalize_business_name(x) for x in df_queries['business_name']]
    print(f"Normalized text in {time.time() - t_norm:.2f}s")

    # 5. Fit word unigram + bigram TF-IDF with max_df=0.001
    t_tfidf = time.time()
    vec = TfidfVectorizer(
        analyzer='word',
        ngram_range=(1, 2),
        max_df=0.001,
        dtype=np.float32
    )
    D = vec.fit_transform(docs_clean)
    Q = vec.transform(queries_clean)
    print(f"TF-IDF fitted & transformed in {time.time() - t_tfidf:.2f}s. Vocab size: {len(vec.vocabulary_):,}")
    print(f"Docs matrix: {D.shape}, nnz={D.nnz:,}; Queries matrix: {Q.shape}, nnz={Q.nnz:,}")

    # 6. Chunked sparse dot product (10,000 queries per chunk)
    t_dot = time.time()
    Dt = D.T.tocsr()
    chunk_size = 10000
    num_queries = Q.shape[0]
    raw_pairs = []
    
    for start_idx in range(0, num_queries, chunk_size):
        end_idx = min(start_idx + chunk_size, num_queries)
        Q_chunk = Q[start_idx:end_idx]
        
        # sparse_dot_topn
        # res has shape (chunk_size, num_docs)
        res = sdt.sp_matmul_topn(Q_chunk, Dt, top_n=5, sort=True, n_threads=8)
        
        # Extract pairs with similarity > 0
        for i in range(end_idx - start_idx):
            q_idx = start_idx + i
            q_id = df_queries['s1_id'].iloc[q_idx]
            
            s = res.indptr[i]
            e = res.indptr[i+1]
            if s == e:
                continue
            
            col_indices = res.indices[s:e]
            scores = res.data[s:e]
            
            for doc_idx, sim_score in zip(col_indices, scores):
                if sim_score > 0:
                    cand_id = df_docs['cand_id'].iloc[doc_idx]
                    raw_pairs.append((q_id, cand_id, float(sim_score)))
                    
    print(f"Retrieved {len(raw_pairs):,} raw top-5 pairs in {time.time() - t_dot:.2f}s")

    # 7. Drop pairs already in V3 test candidates
    t_filter = time.time()
    import pandas as pd
    df_raw = pd.DataFrame(raw_pairs, columns=['s1_id', 'cand_id', 'sim_score'])
    con.register("retrieved_pairs", df_raw)
    
    df_new = con.execute(f"""
    SELECT r.s1_id, r.cand_id, r.sim_score
    FROM retrieved_pairs r
    LEFT JOIN (
        SELECT source1_entity_id, candidate_entity_id 
        FROM read_csv('{v3_pred_path}', delim='\\t', header=True, all_varchar=True)
    ) v3
      ON r.s1_id = v3.source1_entity_id AND r.cand_id = v3.candidate_entity_id
    WHERE v3.candidate_entity_id IS NULL;
    """).df()
    con.unregister("retrieved_pairs")
    
    print(f"Dropped pairs in V3 test candidates: {len(raw_pairs) - len(df_new):,} pairs dropped.")
    print(f"New pairs found: {len(df_new):,} pairs (from {df_new['s1_id'].nunique():,} unique S1s) in {time.time() - t_filter:.2f}s")
    print(f"Total time for {country} {missing_source}: {time.time() - t0:.2f}s")
    
    return df_new

if __name__ == "__main__":
    con = duckdb.connect()
    con.execute("PRAGMA threads=8;")
    con.execute("PRAGMA memory_limit='24GB';")
    
    total_start = time.time()
    
    # Run France
    print("Starting France Retrieval...")
    france_s3 = run_retrieval_for_country_source("FRANCE", "S3", con)
    france_s2 = run_retrieval_for_country_source("FRANCE", "S2", con)
    
    total_france_time = time.time() - total_start
    total_france_new_pairs = len(france_s3) + len(france_s2)
    
    print("\n==========================================")
    print(f"FRANCE RETRIEVAL COMPLETE:")
    print(f"Time taken: {total_france_time:.2f}s ({total_france_time/60:.2f} min)")
    print(f"New S3 pairs: {len(france_s3):,}")
    print(f"New S2 pairs: {len(france_s2):,}")
    print(f"Total new pairs found: {total_france_new_pairs:,}")
    print("==========================================")
