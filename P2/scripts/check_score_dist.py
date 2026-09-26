import os
import duckdb

REPO = r"c:\NEW AMAZON\Amazon-ML-Challenge-2026"
OOF_F0 = os.path.join(REPO, "P2", "reports", "E02_OOF_FULL_CANONICAL_fold0.tsv")

con = duckdb.connect()
con.execute(f"""
    SELECT 
        count(*) as total_rows,
        sum(case when oof_score >= 0.05 then 1 else 0 end) as gte_005,
        sum(case when oof_score >= 0.10 then 1 else 0 end) as gte_010,
        sum(case when oof_score >= 0.30 then 1 else 0 end) as gte_030,
        sum(case when oof_score >= 0.50 then 1 else 0 end) as gte_050,
        sum(case when oof_score >= 0.70 then 1 else 0 end) as gte_070,
        sum(case when oof_score >= 0.90 then 1 else 0 end) as gte_090
    FROM read_csv('{OOF_F0}', delim='\\t', header=true)
""")
print(con.fetchall())
