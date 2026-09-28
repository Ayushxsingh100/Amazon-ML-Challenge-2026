import os
import duckdb

repo_root = "c:/NEW AMAZON"
oof_pattern = "E:/predictions/phase4/v3_train_oof_fold*.parquet"

con = duckdb.connect()
con.execute("PRAGMA threads=8;")
con.execute("PRAGMA memory_limit='20GB';")

print("--- Step 3: OUR_OOF Precision for High-Score Sub-Buckets ---")
query = f"""
SELECT 
    bucket,
    bucket_order,
    COUNT(*) as total_pairs,
    SUM(label) as true_positives,
    ROUND(SUM(label)::DOUBLE / COUNT(*), 6) as precision,
    ROUND(SUM(label)::DOUBLE / COUNT(*) * 100, 4) as precision_pct
FROM (
    SELECT 
        score,
        label,
        CASE 
            WHEN score >= 0.980 AND score < 0.990 THEN '0.98-0.99'
            WHEN score >= 0.990 AND score < 0.995 THEN '0.99-0.995'
            WHEN score >= 0.995 AND score < 0.999 THEN '0.995-0.999'
            WHEN score >= 0.999 AND score <= 1.000001 THEN '0.999-1.0'
            ELSE 'OTHER'
        END as bucket,
        CASE 
            WHEN score >= 0.980 AND score < 0.990 THEN 1
            WHEN score >= 0.990 AND score < 0.995 THEN 2
            WHEN score >= 0.995 AND score < 0.999 THEN 3
            WHEN score >= 0.999 AND score <= 1.000001 THEN 4
            ELSE 5
        END as bucket_order
    FROM '{oof_pattern}'
    WHERE score >= 0.98
)
GROUP BY bucket, bucket_order
ORDER BY bucket_order;
"""

df_buckets = con.execute(query).df()
print(df_buckets.to_string(index=False))
