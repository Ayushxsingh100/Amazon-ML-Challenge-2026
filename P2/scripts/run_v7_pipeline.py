import os
import re
import sys
import time
import unicodedata
import tempfile
import gc
from pathlib import Path

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd
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
V7_PATH = os.path.join(P2_DIR, "data", "matching_results_v7.tsv")
V3_S2_PRED = os.path.join(P2_DIR, "predictions", "phase4", "test_predictions_s2.tsv")
V3_S3_PRED = os.path.join(P2_DIR, "predictions", "phase4", "test_predictions_s3.tsv")
MODEL_DIR = os.path.join(P2_DIR, "models", "phase4")

# Regex & Normalizer
LEGAL_RE = re.compile(r'\b(?:ltd|llc|inc|pvt|gmbh|sarl|sas|sa|srl|bv|co)\b')
NON_ALPHANUM_RE = re.compile(r'[^a-z0-9 ]')

def normalize_business_name(name):
    if not isinstance(name, str):
        return ""
    norm = unicodedata.normalize('NFKD', name)
    s = ''.join(c for c in norm if not unicodedata.combining(c)).lower()
    s = s.replace('.', '')
    s = NON_ALPHANUM_RE.sub(' ', s)
    s = LEGAL_RE.sub(' ', s)
    return ' '.join(s.split())

def load_lf_model(p: str) -> lgb.Booster:
    with open(p, "r", encoding="utf-8") as f:
        text = f.read().replace("\r\n", "\n")
    with tempfile.NamedTemporaryFile("w", delete=False, newline="\n", suffix=".txt") as tmp:
        tmp.write(text)
        tmp_path = tmp.name
    try:
        bst = lgb.Booster(model_file=tmp_path)
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
    return bst

def run_retrieval_for_country_source(country, missing_source, con):
    print(f"\n==========================================")
    print(f"Retrieval: Country={country}, Missing Source={missing_source}")
    print(f"==========================================")
    t0 = time.time()
    
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

    # Load Queries
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
        return pd.DataFrame(columns=['s1_id', 'cand_id', 'sim_score'])

    # Load Docs
    t_doc = time.time()
    df_docs = con.execute(f"""
    SELECT entity_id as cand_id, business_name
    FROM read_csv('{doc_path}', delim='\t', header=True, all_varchar=True)
    WHERE country = '{country}';
    """).df()
    print(f"Loaded {len(df_docs):,} doc records in {time.time() - t_doc:.2f}s")

    # Normalize business_name
    t_norm = time.time()
    docs_clean = [normalize_business_name(x) for x in df_docs['business_name']]
    queries_clean = [normalize_business_name(x) for x in df_queries['business_name']]
    print(f"Normalized text in {time.time() - t_norm:.2f}s")

    # Fit word unigram + bigram TF-IDF with max_df=0.001
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

    # Chunked sparse dot product (10,000 queries per chunk)
    t_dot = time.time()
    Dt = D.T.tocsr()
    chunk_size = 10000
    num_queries = Q.shape[0]
    raw_pairs = []
    
    for start_idx in range(0, num_queries, chunk_size):
        end_idx = min(start_idx + chunk_size, num_queries)
        Q_chunk = Q[start_idx:end_idx]
        
        res = sdt.sp_matmul_topn(Q_chunk, Dt, top_n=5, sort=True, n_threads=8)
        
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

    # Free memory
    del D, Q, Dt, docs_clean, queries_clean
    gc.collect()

    # Drop pairs already in V3 test candidates
    t_filter = time.time()
    df_raw = pd.DataFrame(raw_pairs, columns=['s1_id', 'cand_id', 'sim_score'])
    del raw_pairs
    gc.collect()

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
    
    del df_raw
    gc.collect()
    
    print(f"New pairs found: {len(df_new):,} pairs (from {df_new['s1_id'].nunique():,} unique S1s) in {time.time() - t_filter:.2f}s")
    print(f"Total time for {country} {missing_source}: {time.time() - t0:.2f}s")
    
    return df_new

def main():
    con = duckdb.connect()
    con.execute("PRAGMA threads=8;")
    con.execute("PRAGMA memory_limit='24GB';")
    
    total_start = time.time()
    print("=================================================================")
    print("STEP 2': Word-Level Retrieval for France, US, and India")
    print("=================================================================")
    
    retrieval_stats = {}
    all_dfs = []
    
    # France first
    t_fr0 = time.time()
    fr_s3 = run_retrieval_for_country_source("FRANCE", "S3", con)
    fr_s2 = run_retrieval_for_country_source("FRANCE", "S2", con)
    t_fr = time.time() - t_fr0
    fr_pairs = len(fr_s3) + len(fr_s2)
    retrieval_stats["FRANCE"] = {"time_sec": t_fr, "new_s3": len(fr_s3), "new_s2": len(fr_s2), "total": fr_pairs}
    all_dfs.extend([fr_s3, fr_s2])
    
    print(f"\n>>> FRANCE SUMMARY: {fr_pairs:,} new pairs in {t_fr:.2f}s ({t_fr/60:.2f} min)")
    
    if t_fr > 600:
        print("France took > 10 min. Stopping retrieval and building V7 from France alone.")
    else:
        print("France took < 10 min. Running US and India...")
        
        # US
        t_us0 = time.time()
        us_s3 = run_retrieval_for_country_source("US", "S3", con)
        us_s2 = run_retrieval_for_country_source("US", "S2", con)
        t_us = time.time() - t_us0
        us_pairs = len(us_s3) + len(us_s2)
        retrieval_stats["US"] = {"time_sec": t_us, "new_s3": len(us_s3), "new_s2": len(us_s2), "total": us_pairs}
        all_dfs.extend([us_s3, us_s2])
        print(f"\n>>> US SUMMARY: {us_pairs:,} new pairs in {t_us:.2f}s ({t_us/60:.2f} min)")
        
        # India
        t_in0 = time.time()
        in_s3 = run_retrieval_for_country_source("INDIA", "S3", con)
        in_s2 = run_retrieval_for_country_source("INDIA", "S2", con)
        t_in = time.time() - t_in0
        in_pairs = len(in_s3) + len(in_s2)
        retrieval_stats["INDIA"] = {"time_sec": t_in, "new_s3": len(in_s3), "new_s2": len(in_s2), "total": in_pairs}
        all_dfs.extend([in_s3, in_s2])
        print(f"\n>>> INDIA SUMMARY: {in_pairs:,} new pairs in {t_in:.2f}s ({t_in/60:.2f} min)")

    # Combine all retrieved new pairs
    df_all_new = pd.concat(all_dfs, ignore_index=True)
    del all_dfs
    gc.collect()
    
    total_retrieval_time = time.time() - total_start
    print(f"\n=================================================================")
    print(f"STEP 2' COMPLETE:")
    print(f"Total retrieved candidate pairs: {len(df_all_new):,}")
    print(f"Total retrieval time: {total_retrieval_time:.2f}s ({total_retrieval_time/60:.2f} min)")
    print(f"=================================================================")
    
    # STEP 3: Feature Engineering + Model Prediction
    print("\n=================================================================")
    print("STEP 3: Scoring New Pairs with 5-Fold LightGBM Ensemble")
    print("=================================================================")
    t_feat0 = time.time()
    
    con.register("all_new_pairs", df_all_new[["s1_id", "cand_id"]])
    
    print("Loading normalized entity tables into DuckDB...")
    con.execute(f"""
    CREATE TEMP TABLE s1_entities AS 
    SELECT entity_id, business_name, business_address, country 
    FROM read_csv('{S1_PATH}', delim='\t', header=True, all_varchar=True);
    
    CREATE TEMP TABLE tgt_entities AS 
    SELECT entity_id, business_name, business_address, country 
    FROM read_csv('{S2_PATH}', delim='\t', header=True, all_varchar=True)
    UNION ALL
    SELECT entity_id, business_name, business_address, country 
    FROM read_csv('{S3_PATH}', delim='\t', header=True, all_varchar=True);
    """)
    
    print(f"Computing 19 canonical features for {len(df_all_new):,} pairs...")
    FEATURE_SQL = """
        p.s1_id,
        p.cand_id,
        CASE WHEN s1.business_name <> '' AND s1.business_name = tgt.business_name THEN 1.0 ELSE 0.0 END AS name_exact_match,
        jaro_winkler_similarity(COALESCE(s1.business_name,''), COALESCE(tgt.business_name,'')) / 100.0 AS name_jaro_winkler,
        CASE WHEN LENGTH(COALESCE(s1.business_name,'')) >= 2 AND LENGTH(COALESCE(tgt.business_name,'')) >= 2 
             THEN jaccard(COALESCE(s1.business_name,''), COALESCE(tgt.business_name,'')) ELSE 0.0 END AS name_jaccard,
        CASE WHEN LENGTH(s1.business_name) >= 4 AND LENGTH(tgt.business_name) >= 4
                  AND LEFT(s1.business_name, 4) = LEFT(tgt.business_name, 4) THEN 1.0 ELSE 0.0 END AS prefix4_match,
        CASE WHEN s1.business_name <> '' AND tgt.business_name <> ''
                  AND SPLIT_PART(s1.business_name, ' ', 1) = SPLIT_PART(tgt.business_name, ' ', 1) THEN 1.0 ELSE 0.0 END AS first_token_match,
        CAST(ABS(LENGTH(COALESCE(s1.business_name,'')) - LENGTH(COALESCE(tgt.business_name,''))) AS DOUBLE) AS name_len_diff,
        CASE WHEN LENGTH(s1.business_name) > 0 AND LENGTH(tgt.business_name) > 0
             THEN CAST(LEAST(LENGTH(s1.business_name), LENGTH(tgt.business_name)) AS DOUBLE) / GREATEST(LENGTH(s1.business_name), LENGTH(tgt.business_name))
             ELSE 0.0 END AS name_len_ratio,
        CASE WHEN s1.business_address <> '' AND s1.business_address = tgt.business_address THEN 1.0 ELSE 0.0 END AS address_exact_match,
        jaro_winkler_similarity(COALESCE(s1.business_address,''), COALESCE(tgt.business_address,'')) / 100.0 AS address_jaro_winkler,
        CASE WHEN LENGTH(COALESCE(s1.business_address,'')) >= 2 AND LENGTH(COALESCE(tgt.business_address,'')) >= 2 
             THEN jaccard(COALESCE(s1.business_address,''), COALESCE(tgt.business_address,'')) ELSE 0.0 END AS address_jaccard,
        CAST(ABS(LENGTH(COALESCE(s1.business_address,'')) - LENGTH(COALESCE(tgt.business_address,''))) AS DOUBLE) AS address_len_diff,
        CASE WHEN LENGTH(s1.business_address) > 0 AND LENGTH(tgt.business_address) > 0
             THEN CAST(LEAST(LENGTH(s1.business_address), LENGTH(tgt.business_address)) AS DOUBLE) / GREATEST(LENGTH(s1.business_address), LENGTH(tgt.business_address))
             ELSE 0.0 END AS address_len_ratio,
        CASE WHEN regexp_extract(COALESCE(s1.business_address,''), '[0-9]+[A-Za-z]?', 0) <> ''
                  AND regexp_extract(COALESCE(tgt.business_address,''), '[0-9]+[A-Za-z]?', 0) <> ''
                  AND regexp_extract(s1.business_address, '[0-9]+[A-Za-z]?', 0) = regexp_extract(tgt.business_address, '[0-9]+[A-Za-z]?', 0)
             THEN 1.0 ELSE 0.0 END AS address_first_number_match,
        CASE WHEN s1.country <> '' AND s1.country = tgt.country THEN 1.0 ELSE 0.0 END AS country_match,
        CAST(LENGTH(COALESCE(s1.business_name,'')) AS DOUBLE) AS s1_name_len,
        CAST(LENGTH(COALESCE(tgt.business_name,'')) AS DOUBLE) AS tgt_name_len,
        CAST(LENGTH(COALESCE(s1.business_address,'')) AS DOUBLE) AS s1_addr_len,
        CAST(LENGTH(COALESCE(tgt.business_address,'')) AS DOUBLE) AS tgt_addr_len,
        CASE WHEN p.cand_id LIKE 'S3-%' THEN 1.0 ELSE 0.0 END AS source_is_s3
    """
    df_features = con.execute(f"""
    SELECT {FEATURE_SQL}
    FROM all_new_pairs p
    JOIN s1_entities s1 ON p.s1_id = s1.entity_id
    JOIN tgt_entities tgt ON p.cand_id = tgt.entity_id;
    """).df()
    print(f"Features computed in {time.time() - t_feat0:.2f}s. Shape: {df_features.shape}")
    
    feature_cols = [
        "name_exact_match", "name_jaro_winkler", "name_jaccard", "prefix4_match",
        "first_token_match", "name_len_diff", "name_len_ratio",
        "address_exact_match", "address_jaro_winkler", "address_jaccard",
        "address_len_diff", "address_len_ratio", "address_first_number_match",
        "country_match", "s1_name_len", "tgt_name_len", "s1_addr_len",
        "tgt_addr_len", "source_is_s3"
    ]
    
    t_model0 = time.time()
    print("Loading 5-fold LightGBM boosters...")
    models = [load_lf_model(os.path.join(MODEL_DIR, f"lgb_fold{fold}.txt")) for fold in range(5)]
    
    X = df_features[feature_cols].values
    print(f"Predicting ensemble scores for {len(X):,} rows...")
    preds = np.zeros(len(X), dtype=np.float64)
    for fold, bst in enumerate(models):
        t_fold = time.time()
        preds += bst.predict(X) / 5.0
        print(f"  Fold {fold} predict done in {time.time() - t_fold:.2f}s")
        
    df_features["score"] = preds
    print(f"Model prediction finished in {time.time() - t_model0:.2f}s")
    
    # Score distribution of newly retrieved pairs
    print("\nScore distribution of new pairs:")
    print(pd.cut(df_features["score"], bins=[-np.inf, 0.05, 0.2, 0.5, 0.8, 0.95, 0.98, 0.995, 1.0]).value_counts().sort_index())
    
    # Register scored pairs
    con.register("scored_new_pairs", df_features[["s1_id", "cand_id", "score"]])
    
    # STEP 4: Build V7
    print("\n=================================================================")
    print("STEP 4: Building V7 (Cutoff >= 0.995, unused in V5, max 1/row)")
    print("=================================================================")
    t_v7_0 = time.time()
    
    # 1. Load V5 and identify used candidate IDs
    print("Loading V5 and extracting used candidate IDs...")
    con.execute(f"""
    CREATE TEMP TABLE v5_raw AS
    SELECT 
        row_number() over () as row_id,
        source1_entity_id as s1_id,
        COALESCE(matched_entity_ids, '') as v5_matched_ids
    FROM read_csv('{V5_PATH}', delim='\t', header=True, all_varchar=True);
    
    CREATE TEMP TABLE v5_used_ids AS
    WITH split_ids AS (
        SELECT unnest(string_split(v5_matched_ids, ',')) as cand_id
        FROM v5_raw
        WHERE v5_matched_ids IS NOT NULL AND TRIM(v5_matched_ids) != ''
    )
    SELECT DISTINCT TRIM(cand_id) as cand_id FROM split_ids;
    """)
    
    v5_used_count = con.execute("SELECT COUNT(*) FROM v5_used_ids").fetchone()[0]
    print(f"V5 has {v5_used_count:,} unique candidate IDs already used.")
    
    # 2. Filter qualifying additions:
    # - score >= 0.995
    # - cand_id NOT IN (v5_used_ids)
    # - global conflict resolution: highest score per cand_id
    # - at most 1 new pair per s1_id: highest score per s1_id
    print("Selecting qualifying additions...")
    con.execute("""
    CREATE TEMP TABLE v7_qualifying_pairs AS
    SELECT s1_id, cand_id, score
    FROM scored_new_pairs
    WHERE score >= 0.995
      AND cand_id NOT IN (SELECT cand_id FROM v5_used_ids);
      
    CREATE TEMP TABLE v7_deduped_cand AS
    WITH ranked_cand AS (
        SELECT 
            s1_id,
            cand_id,
            score,
            ROW_NUMBER() OVER (PARTITION BY cand_id ORDER BY score DESC, s1_id ASC) as rnk
        FROM v7_qualifying_pairs
    )
    SELECT s1_id, cand_id, score
    FROM ranked_cand
    WHERE rnk = 1;
    
    CREATE TEMP TABLE v7_additions AS
    WITH ranked_s1 AS (
        SELECT 
            s1_id,
            cand_id,
            score,
            ROW_NUMBER() OVER (PARTITION BY s1_id ORDER BY score DESC, cand_id ASC) as rnk
        FROM v7_deduped_cand
    )
    SELECT s1_id, cand_id, score
    FROM ranked_s1
    WHERE rnk = 1;
    """)
    
    v7_pairs_added = con.execute("SELECT COUNT(*) FROM v7_additions").fetchone()[0]
    v7_rows_changed = con.execute("SELECT COUNT(DISTINCT s1_id) FROM v7_additions").fetchone()[0]
    print(f"V7 Qualifying Pairs (score >= 0.995, unused in V5): {con.execute('SELECT COUNT(*) FROM v7_qualifying_pairs').fetchone()[0]:,}")
    print(f"V7 After Candidate Conflict Resolution: {con.execute('SELECT COUNT(*) FROM v7_deduped_cand').fetchone()[0]:,}")
    print(f"V7 Final Pairs Added (max 1/row): {v7_pairs_added:,}")
    print(f"V7 Rows Changed: {v7_rows_changed:,}")
    
    # Breakdown of added pairs by country & source
    print("\nBreakdown of added pairs by source (S2 vs S3):")
    print(con.execute("""
    SELECT 
        CASE WHEN cand_id LIKE 'S2-%' THEN 'S2' ELSE 'S3' END as source,
        COUNT(*) as count
    FROM v7_additions
    GROUP BY source;
    """).df())
    
    # Merge additions into V5
    print("\nMerging additions into V5...")
    con.execute("""
    CREATE TEMP TABLE v7_full_rows AS
    SELECT 
        r.row_id,
        r.s1_id,
        CASE 
            WHEN a.cand_id IS NOT NULL AND a.cand_id != '' THEN
                r.v5_matched_ids || ',' || a.cand_id
            ELSE
                r.v5_matched_ids
        END as v7_matched_ids
    FROM v5_raw r
    LEFT JOIN v7_additions a ON r.s1_id = a.s1_id;
    """)
    
    # STEP 5: Write V7 and Validate
    print("\n=================================================================")
    print("STEP 5: Writing and Validating matching_results_v7.tsv")
    print("=================================================================")
    t_w0 = time.time()
    v7_rows = con.execute("SELECT s1_id, v7_matched_ids FROM v7_full_rows ORDER BY row_id ASC").fetchall()
    
    with open(V7_PATH, "w", encoding="utf-8", newline="\n") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for s1, m in v7_rows:
            f.write(f"{s1}\t{m}\n")
    print(f"Wrote {V7_PATH} in {time.time() - t_w0:.2f}s")
    
    # Validation
    print("Validating V7 submission file...")
    line_count = 0
    has_dup = False
    empty_count = 0
    with open(V7_PATH, "r", encoding="utf-8") as f:
        header = f.readline().rstrip("\n")
        assert header == "source1_entity_id\tmatched_entity_ids", f"Invalid header: {header}"
        for line in f:
            line_count += 1
            parts = line.rstrip("\n").split("\t")
            assert len(parts) == 2, f"Line {line_count} does not have 2 fields"
            ids_str = parts[1]
            if not ids_str:
                empty_count += 1
            else:
                ids_list = ids_str.split(",")
                if len(ids_list) != len(set(ids_list)):
                    has_dup = True
                    break
                    
    assert line_count == 1732544, f"Row count mismatch: expected 1732544, got {line_count}"
    assert not has_dup, "Duplicate ids found within a row"
    print(f"Validation PASSED:")
    print(f"  Total rows: {line_count:,}")
    print(f"  Empty rows: {empty_count:,}")
    print(f"  Pairs added: {v7_pairs_added:,}")
    print(f"  Rows changed: {v7_rows_changed:,}")
    print(f"  Total time from start: {time.time() - total_start:.2f}s ({(time.time() - total_start)/60:.2f} min)")

if __name__ == "__main__":
    main()
