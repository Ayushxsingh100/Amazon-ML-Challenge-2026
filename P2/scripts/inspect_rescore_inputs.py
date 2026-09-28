import duckdb

con = duckdb.connect()

print("--- OOF Parquet Schema ---")
print(con.execute("DESCRIBE SELECT * FROM 'E:/predictions/phase4/v3_train_oof_fold0.parquet'").df())

print("\n--- OOF Parquet Head ---")
print(con.execute("SELECT * FROM 'E:/predictions/phase4/v3_train_oof_fold0.parquet' LIMIT 5").df())

print("\n--- BEST TSV Head ---")
print(con.execute("SELECT * FROM read_csv('Amazon-ML-Challenge-2026/P2/data/matching_results_v1_conservative.tsv', delim='\t', header=True) LIMIT 5").df())

print("\n--- TEST PRED S2 Head ---")
print(con.execute("SELECT * FROM read_csv('Amazon-ML-Challenge-2026/P2/predictions/phase4/test_predictions_s2.tsv', delim='\t', header=True) LIMIT 5").df())
