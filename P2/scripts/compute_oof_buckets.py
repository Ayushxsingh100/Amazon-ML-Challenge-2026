import duckdb
import time

con = duckdb.connect()
con.execute("PRAGMA threads=8;")
con.execute("PRAGMA memory_limit='20GB';")

print("--- Computing OOF Precision by Score Bucket ---")
t0 = time.time()

# Let's verify row count across the 5 folds
row_count = con.execute("SELECT COUNT(*) FROM 'E:/predictions/phase4/v3_train_oof_fold*.parquet'").fetchone()[0]
print(f"Total OOF rows: {row_count}")

# Score buckets:
# 0-0.02, 0.02-0.05, 0.05-0.1, 0.1-0.2, 0.2-0.5, 0.5-0.8, 0.8-0.9, 0.9-0.95, 0.95-0.98, 0.98-1.0
query = """
SELECT 
    bucket,
    bucket_order,
    COUNT(*) as total_pairs,
    SUM(label) as true_positives,
    ROUND(SUM(label)::DOUBLE / COUNT(*), 6) as precision
FROM (
    SELECT 
        score,
        label,
        CASE 
            WHEN score >= 0.00 AND score < 0.02 THEN '0-0.02'
            WHEN score >= 0.02 AND score < 0.05 THEN '0.02-0.05'
            WHEN score >= 0.05 AND score < 0.10 THEN '0.05-0.1'
            WHEN score >= 0.10 AND score < 0.20 THEN '0.1-0.2'
            WHEN score >= 0.20 AND score < 0.50 THEN '0.2-0.5'
            WHEN score >= 0.50 AND score < 0.80 THEN '0.5-0.8'
            WHEN score >= 0.80 AND score < 0.90 THEN '0.8-0.9'
            WHEN score >= 0.90 AND score < 0.95 THEN '0.9-0.95'
            WHEN score >= 0.95 AND score < 0.98 THEN '0.95-0.98'
            WHEN score >= 0.98 AND score <= 1.000001 THEN '0.98-1.0'
            ELSE 'OTHER'
        END as bucket,
        CASE 
            WHEN score >= 0.00 AND score < 0.02 THEN 1
            WHEN score >= 0.02 AND score < 0.05 THEN 2
            WHEN score >= 0.05 AND score < 0.10 THEN 3
            WHEN score >= 0.10 AND score < 0.20 THEN 4
            WHEN score >= 0.20 AND score < 0.50 THEN 5
            WHEN score >= 0.50 AND score < 0.80 THEN 6
            WHEN score >= 0.80 AND score < 0.90 THEN 7
            WHEN score >= 0.90 AND score < 0.95 THEN 8
            WHEN score >= 0.95 AND score < 0.98 THEN 9
            WHEN score >= 0.98 AND score <= 1.000001 THEN 10
            ELSE 11
        END as bucket_order
    FROM 'E:/predictions/phase4/v3_train_oof_fold*.parquet'
)
GROUP BY bucket, bucket_order
ORDER BY bucket_order;
"""

df_buckets = con.execute(query).df()
print(df_buckets.to_string(index=False))
print(f"Time: {time.time() - t0:.2f}s")
