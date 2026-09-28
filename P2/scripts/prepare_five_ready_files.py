import os
import shutil
import time
import duckdb

def main():
    repo_root = "c:/NEW AMAZON/Amazon-ML-Challenge-2026"
    ready_dir = os.path.join(repo_root, "ready_files")
    os.makedirs(ready_dir, exist_ok=True)
    
    con = duckdb.connect()
    con.execute("PRAGMA threads=8;")
    con.execute("PRAGMA memory_limit='24GB';")
    
    print("=================================================================")
    print("Preparing the Five Ready Files in:", ready_dir)
    print("=================================================================")
    
    # -------------------------------------------------------------------------
    # 1. Validation Candidate Pairs Scored (Model did NOT train on these entities)
    # Fold 0 OOF: scored by lgb_fold0.txt (trained on Folds 1, 2, 3, 4)
    # -------------------------------------------------------------------------
    print("\n[1/5] Preparing validation candidate pairs with model scores (Fold 0 OOF)...")
    t0 = time.time()
    f0_src_parquet = "E:/predictions/phase4/v3_train_oof_fold0.parquet"
    out_val_tsv = os.path.join(ready_dir, "validation_candidate_pairs_scored.tsv")
    out_val_parquet = os.path.join(ready_dir, "validation_candidate_pairs_scored.parquet")
    
    # Copy parquet directly (instant, compact)
    shutil.copyfile(f0_src_parquet, out_val_parquet)
    print(f"  Saved Parquet: {out_val_parquet} ({os.path.getsize(out_val_parquet)/1e6:.2f} MB)")
    
    # Export TSV: source1_entity_id, candidate_entity_id, score, label
    con.execute(f"""
    COPY (
        SELECT source1_entity_id, candidate_entity_id, score, label
        FROM '{f0_src_parquet}'
    ) TO '{out_val_tsv.replace(os.sep, "/")}' (HEADER, DELIMITER '\t');
    """)
    val_count = con.execute(f"SELECT count(*) FROM '{out_val_parquet}'").fetchone()[0]
    print(f"  Saved TSV: {out_val_tsv} ({os.path.getsize(out_val_tsv)/1e6:.2f} MB)")
    print(f"  Rows: {val_count:,} pairs in {time.time() - t0:.2f}s")
    
    # -------------------------------------------------------------------------
    # 2. Train Ground Truth
    # -------------------------------------------------------------------------
    print("\n[2/5] Preparing train ground truth...")
    t0 = time.time()
    gt_src = os.path.join(repo_root, "outputs", "person1_step1", "train_ground_truth_reconstructed.tsv")
    out_gt = os.path.join(ready_dir, "train_ground_truth.tsv")
    shutil.copyfile(gt_src, out_gt)
    gt_count = con.execute(f"SELECT count(*) FROM read_csv('{out_gt.replace(os.sep, "/")}', delim='\t', header=True)").fetchone()[0]
    print(f"  Saved TSV: {out_gt} ({os.path.getsize(out_gt)/1e6:.2f} MB)")
    print(f"  Rows: {gt_count:,} in {time.time() - t0:.2f}s")
    
    # -------------------------------------------------------------------------
    # 3. List of Validation S1 IDs
    # -------------------------------------------------------------------------
    print("\n[3/5] Preparing list of validation S1 IDs (Fold 0)...")
    t0 = time.time()
    manifest_src = os.path.join(repo_root, "P3", "reports", "folds_v1_manifest.tsv")
    out_s1_ids = os.path.join(ready_dir, "validation_s1_ids.tsv")
    
    con.execute(f"""
    COPY (
        SELECT source1_entity_id
        FROM read_csv('{manifest_src.replace(os.sep, "/")}', delim='\t', header=True)
        WHERE fold = 0
        ORDER BY source1_entity_id
    ) TO '{out_s1_ids.replace(os.sep, "/")}' (HEADER, DELIMITER '\t');
    """)
    s1_count = con.execute(f"SELECT count(*) FROM read_csv('{out_s1_ids.replace(os.sep, "/")}', delim='\t', header=True)").fetchone()[0]
    print(f"  Saved TSV: {out_s1_ids} ({os.path.getsize(out_s1_ids)/1e6:.2f} MB)")
    print(f"  Validation S1 Entities: {s1_count:,} in {time.time() - t0:.2f}s")
    
    # -------------------------------------------------------------------------
    # 4. Test Candidate Pairs with Scores
    # -------------------------------------------------------------------------
    print("\n[4/5] Preparing test candidate pairs with scores (Union of S2 and S3)...")
    t0 = time.time()
    test_s2 = os.path.join(repo_root, "P2", "predictions", "phase4", "test_predictions_s2.tsv")
    test_s3 = os.path.join(repo_root, "P2", "predictions", "phase4", "test_predictions_s3.tsv")
    out_test_tsv = os.path.join(ready_dir, "test_candidate_pairs_scored.tsv")
    out_test_parquet = os.path.join(ready_dir, "test_candidate_pairs_scored.parquet")
    
    con.execute(f"""
    CREATE TEMP TABLE test_all_scored AS
    SELECT source1_entity_id, candidate_entity_id, model_score as score
    FROM read_csv('{test_s2.replace(os.sep, "/")}', delim='\t', header=True, columns={{'source1_entity_id': 'VARCHAR', 'candidate_entity_id': 'VARCHAR', 'model_score': 'DOUBLE', 'predicted_match_label': 'TINYINT'}})
    UNION ALL
    SELECT source1_entity_id, candidate_entity_id, model_score as score
    FROM read_csv('{test_s3.replace(os.sep, "/")}', delim='\t', header=True, columns={{'source1_entity_id': 'VARCHAR', 'candidate_entity_id': 'VARCHAR', 'model_score': 'DOUBLE', 'predicted_match_label': 'TINYINT'}});
    
    COPY test_all_scored TO '{out_test_parquet.replace(os.sep, "/")}' (FORMAT PARQUET);
    COPY test_all_scored TO '{out_test_tsv.replace(os.sep, "/")}' (HEADER, DELIMITER '\t');
    """)
    test_count = con.execute("SELECT count(*) FROM test_all_scored").fetchone()[0]
    print(f"  Saved Parquet: {out_test_parquet} ({os.path.getsize(out_test_parquet)/1e6:.2f} MB)")
    print(f"  Saved TSV: {out_test_tsv} ({os.path.getsize(out_test_tsv)/1e6:.2f} MB)")
    print(f"  Rows: {test_count:,} test candidate pairs in {time.time() - t0:.2f}s")
    
    # -------------------------------------------------------------------------
    # 5. Current matching_results.tsv (V7)
    # -------------------------------------------------------------------------
    print("\n[5/5] Preparing current matching_results.tsv (Latest V7 submission)...")
    t0 = time.time()
    v7_src = os.path.join(repo_root, "P2", "data", "matching_results_v7.tsv")
    out_sub = os.path.join(ready_dir, "matching_results.tsv")
    shutil.copyfile(v7_src, out_sub)
    
    # Also update output/matching_results.tsv so the standard top-level location is intact
    out_top = os.path.join(repo_root, "output", "matching_results.tsv")
    shutil.copyfile(v7_src, out_top)
    
    sub_count = con.execute(f"SELECT count(*) FROM read_csv('{out_sub.replace(os.sep, "/")}', delim='\t', header=True, all_varchar=True)").fetchone()[0]
    empty_count = con.execute(f"SELECT count(*) FROM read_csv('{out_sub.replace(os.sep, "/")}', delim='\t', header=True, all_varchar=True) WHERE matched_entity_ids IS NULL OR TRIM(matched_entity_ids) = ''").fetchone()[0]
    print(f"  Saved TSV: {out_sub} ({os.path.getsize(out_sub)/1e6:.2f} MB)")
    print(f"  Also updated: {out_top}")
    print(f"  Total submission rows: {sub_count:,} (Empty rows: {empty_count:,}) in {time.time() - t0:.2f}s")
    
    print("\n=================================================================")
    print("ALL FIVE FILES SUCCESSFULLY READY!")
    print("=================================================================")

if __name__ == "__main__":
    main()
