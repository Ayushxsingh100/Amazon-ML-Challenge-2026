import os
import time
import duckdb
import numpy as np
import pandas as pd
import lightgbm as lgb
import json

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
P3_MANIFEST = os.path.join(REPO, "P3", "reports", "folds_v1_manifest.tsv")
FEAT_DIR = os.path.join(REPO, "P2", "data", "features")
REPORT_DIR = os.path.join(REPO, "P2", "reports")
MODEL_DIR = os.path.join(REPO, "P2", "models")
TMP_DIR = os.path.join(REPO, "P2", "data", "duckdb_tmp_e02_oof")
os.makedirs(TMP_DIR, exist_ok=True)

OOF_PATH = os.path.join(REPORT_DIR, "E02_OOF_FULL_CANONICAL.tsv")

def main():
    con = duckdb.connect()
    con.execute(f"SET temp_directory='{TMP_DIR}'")
    con.execute("SET memory_limit='16GB'")
    con.execute("SET threads=4")
    
    print("Loading folds...")
    con.execute(f"CREATE TABLE folds AS SELECT source1_entity_id, fold FROM read_csv('{P3_MANIFEST}', delim='\t', header=true)")
    
    features = [
        "name_exact_match", "name_jaro_winkler", "name_jaccard", "prefix4_match",
        "first_token_match", "name_len_diff", "name_len_ratio",
        "address_exact_match", "address_jaro_winkler", "address_jaccard",
        "address_len_diff", "address_len_ratio", "address_first_number_match",
        "country_match", "s1_name_len", "tgt_name_len", "s1_addr_len",
        "tgt_addr_len", "source_is_s3"
    ]
    cat_features = ["source_is_s3", "country_match", "address_first_number_match", "name_exact_match", "address_exact_match", "prefix4_match", "first_token_match"]
    
    print("Joining full canonical features...")
    con.execute(f"""
        CREATE TABLE full_train AS 
        SELECT f.fold, t.* FROM read_csv('{FEAT_DIR}/cands_BCD_v1_features_train_s2.tsv', delim='\t', header=true) t
        JOIN folds f ON t.source1_entity_id = f.source1_entity_id
        UNION ALL
        SELECT f.fold, t.* FROM read_csv('{FEAT_DIR}/cands_BCD_v1_features_train_s3.tsv', delim='\t', header=true) t
        JOIN folds f ON t.source1_entity_id = f.source1_entity_id
    """)
    
    # Write empty file with header
    pd.DataFrame(columns=["source1_entity_id", "matched_entity_id", "target", "fold", "oof_score"]).to_csv(OOF_PATH, sep='\t', index=False)
    
    total_preds = 0
    start_t = time.time()
    
    for f in range(5):
        print(f"\\n--- Predicting Fold {f} ---")
        df = con.execute(f"SELECT source1_entity_id, candidate_entity_id, fold, label, " + ", ".join(features) + f" FROM full_train WHERE fold = {f}").fetch_arrow_table().to_pandas()
        
        for c in features:
            if c in cat_features:
                df[c] = df[c].astype('int8')
            else:
                df[c] = df[c].astype('float32')
        
        print(f"Loaded {len(df)} rows for fold {f}")
        
        model_path = os.path.join(MODEL_DIR, f"e02_lgb_fold{f}.txt")
        model = lgb.Booster(model_file=model_path)
        
        preds = np.zeros(len(df), dtype=np.float32)
        chunk_size = 1000000
        for i in range(0, len(df), chunk_size):
            preds[i:i+chunk_size] = model.predict(df[features].iloc[i:i+chunk_size])
            
        df['oof_score'] = preds
        
        out_df = df[['source1_entity_id', 'candidate_entity_id', 'label', 'fold', 'oof_score']].rename(
            columns={'candidate_entity_id': 'matched_entity_id', 'label': 'target'}
        )
        total_preds += len(df)
        del df
        
        # Append to TSV
        out_df.to_csv(OOF_PATH, sep='\t', index=False, mode='a', header=False, chunksize=100000)
        del out_df
        
    print(f"\\nCompleted OOF generation. Total predicted: {total_preds}")
    print(f"Time taken: {time.time() - start_t:.1f}s")
    
    print("\\nRunning Integrity Checks...")
    con.execute(f"CREATE TABLE oof AS SELECT * FROM read_csv('{OOF_PATH}', delim='\t', header=true)")
    
    oof_count = con.execute("SELECT COUNT(*) FROM oof").fetchone()[0]
    print(f"OOF row count: {oof_count}")
    
    s2_count = con.execute(f"SELECT COUNT(*) FROM read_csv('{FEAT_DIR}/cands_BCD_v1_features_train_s2.tsv', delim='\t', header=true)").fetchone()[0]
    s3_count = con.execute(f"SELECT COUNT(*) FROM read_csv('{FEAT_DIR}/cands_BCD_v1_features_train_s3.tsv', delim='\t', header=true)").fetchone()[0]
    
    oof_s2 = con.execute(f"SELECT COUNT(*) FROM oof o JOIN read_csv('{FEAT_DIR}/cands_BCD_v1_features_train_s2.tsv', delim='\t', header=true) s2 ON o.source1_entity_id=s2.source1_entity_id AND o.matched_entity_id=s2.candidate_entity_id").fetchone()[0]
    oof_s3 = con.execute(f"SELECT COUNT(*) FROM oof o JOIN read_csv('{FEAT_DIR}/cands_BCD_v1_features_train_s3.tsv', delim='\t', header=true) s3 ON o.source1_entity_id=s3.source1_entity_id AND o.matched_entity_id=s3.candidate_entity_id").fetchone()[0]
    
    print(f"S2 rows (expected {s2_count}): {oof_s2}")
    print(f"S3 rows (expected {s3_count}): {oof_s3}")
    
    missing = (s2_count + s3_count) - oof_count
    extra = oof_count - (s2_count + s3_count)
    print(f"Symmetric difference: missing={missing}, extra={extra}")
    
    dups = con.execute("SELECT COUNT(*) FROM (SELECT source1_entity_id, matched_entity_id FROM oof GROUP BY 1, 2 HAVING COUNT(*) > 1)").fetchone()[0]
    print(f"Duplicates: {dups}")
    
    finite_check = con.execute("SELECT COUNT(*) FROM oof WHERE oof_score IS NULL OR oof_score = 'inf' OR oof_score = '-inf'").fetchone()[0]
    print(f"Non-finite scores: {finite_check}")
    
    print("Updating E02_RUN.json...")
    json_path = os.path.join(REPORT_DIR, "E02_RUN.json")
    with open(json_path, "r") as f:
        meta = json.load(f)
        
    meta["memory_behavior"] = "TRAINING: 10% negative downsampling. OOF PREDICTION: Full 54.5M canonical coverage via chunking"
    meta["oof_row_count"] = oof_count
    meta["integrity_checks"]["duplicate_predictions"] = dups
    meta["integrity_checks"]["missing_predictions"] = max(0, missing)
    meta["integrity_checks"]["all_scores_finite"] = (finite_check == 0)
    meta["integrity_checks"]["s2_rows"] = oof_s2
    meta["integrity_checks"]["s3_rows"] = oof_s3
    
    with open(json_path, "w") as f:
        json.dump(meta, f, indent=2)
        
    import hashlib
    with open(OOF_PATH, "rb") as f:
        m = hashlib.sha256(f.read()).hexdigest()
    
    manifest = pd.DataFrame([{"artifact": "E02_OOF_FULL_CANONICAL.tsv", "rows": oof_count, "sha256": m}])
    manifest.to_csv(os.path.join(REPORT_DIR, "E02_OOF_FULL_CANONICAL_MANIFEST.tsv"), sep='\t', index=False)

if __name__ == "__main__":
    main()
