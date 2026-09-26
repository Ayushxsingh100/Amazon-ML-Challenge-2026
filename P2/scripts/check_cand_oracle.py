import os
import duckdb

REPO = r"c:\NEW AMAZON\Amazon-ML-Challenge-2026"
FOLDS_PATH = os.path.join(REPO, "P3", "reports", "folds_v1_manifest.tsv")
GT_PATH = os.path.join(REPO, "outputs", "person1_step1", "train_ground_truth_reconstructed.tsv")
OOF_PATH = os.path.join(REPO, "P2", "reports", "E02_OOF_FULL_CANONICAL.tsv")

con = duckdb.connect()
con.execute("SET threads=4")
con.execute("SET memory_limit='4GB'")

con.execute(f"CREATE TABLE s1_all AS SELECT source1_entity_id, fold FROM read_csv('{FOLDS_PATH}', delim='\\t', header=true)")
con.execute(f"""
    CREATE TABLE gt_counts AS 
    SELECT 
        source1_entity_id,
        CASE WHEN matched_entity_ids IS NULL OR matched_entity_ids = '' THEN 0 ELSE len(string_split(matched_entity_ids, ',')) END as A,
        len(list_filter(string_split(coalesce(matched_entity_ids, ''), ','), x -> x LIKE 'S2-%')) as A_s2,
        len(list_filter(string_split(coalesce(matched_entity_ids, ''), ','), x -> x LIKE 'S3-%')) as A_s3
    FROM read_csv('{GT_PATH}', delim='\\t', header=true)
""")

con.execute(f"""
    CREATE TABLE cand_pos AS
    SELECT 
        source1_entity_id,
        sum(target) as K,
        sum(case when matched_entity_id like 'S2-%' and target=1 then 1 else 0 end) as K_s2,
        sum(case when matched_entity_id like 'S3-%' and target=1 then 1 else 0 end) as K_s3
    FROM read_csv('{OOF_PATH}', delim='\\t', header=true)
    GROUP BY source1_entity_id
""")

con.execute("""
    CREATE TABLE s1_cand_oracle AS
    SELECT 
        s.source1_entity_id,
        s.fold,
        coalesce(g.A, 0) as A,
        coalesce(g.A_s2, 0) as A_s2,
        coalesce(g.A_s3, 0) as A_s3,
        coalesce(c.K, 0) as K,
        coalesce(c.K_s2, 0) as K_s2,
        coalesce(c.K_s3, 0) as K_s3,
        -- Joint Candidate Oracle F0.5
        CASE 
            WHEN coalesce(g.A, 0) = 0 THEN 1.0
            WHEN coalesce(c.K, 0) = 0 THEN 0.0
            ELSE (5.0 * c.K) / (g.A + 4.0 * c.K)
        END as cand_oracle_f05,
        -- S2-only Candidate Oracle F0.5
        CASE 
            WHEN coalesce(g.A_s2, 0) = 0 THEN 1.0
            WHEN coalesce(c.K_s2, 0) = 0 THEN 0.0
            ELSE (5.0 * c.K_s2) / (g.A_s2 + 4.0 * c.K_s2)
        END as cand_oracle_f05_s2,
        -- S3-only Candidate Oracle F0.5
        CASE 
            WHEN coalesce(g.A_s3, 0) = 0 THEN 1.0
            WHEN coalesce(c.K_s3, 0) = 0 THEN 0.0
            ELSE (5.0 * c.K_s3) / (g.A_s3 + 4.0 * c.K_s3)
        END as cand_oracle_f05_s3
    FROM s1_all s
    LEFT JOIN gt_counts g ON s.source1_entity_id = g.source1_entity_id
    LEFT JOIN cand_pos c ON s.source1_entity_id = c.source1_entity_id
""")

print("Overall Candidate Oracle:")
res_overall = con.execute("""
    SELECT 
        count(*),
        avg(cand_oracle_f05),
        avg(cand_oracle_f05_s2),
        avg(cand_oracle_f05_s3)
    FROM s1_cand_oracle
""").fetchone()
print(f"  Joint: {res_overall[1]:.9f}")
print(f"  S2:    {res_overall[2]:.9f}")
print(f"  S3:    {res_overall[3]:.9f}")

print("\nPer-fold Candidate Oracle:")
folds = con.execute("""
    SELECT 
        fold,
        count(*),
        avg(cand_oracle_f05),
        avg(cand_oracle_f05_s2),
        avg(cand_oracle_f05_s3)
    FROM s1_cand_oracle
    GROUP BY fold
    ORDER BY fold
""").fetchall()
for f in folds:
    print(f"  Fold {f[0]}: Joint={f[2]:.9f}, S2={f[3]:.9f}, S3={f[4]:.9f}")
