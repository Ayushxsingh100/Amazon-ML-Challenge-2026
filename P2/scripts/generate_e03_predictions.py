import os
import time
import duckdb
import hashlib

REPO = r"c:\NEW AMAZON\Amazon-ML-Challenge-2026"
REPORT_DIR = os.path.join(REPO, "P2", "reports")
OOF_IN = os.path.join(REPORT_DIR, "E02_OOF_FULL_CANONICAL.tsv")
OOF_OUT = os.path.join(REPORT_DIR, "E03_OOF_PREDICTIONS.tsv")

con = duckdb.connect()
con.execute("SET threads=4")
con.execute("SET memory_limit='4GB'")

t0 = time.time()
print("Creating fold threshold mapping...")
# All 5 folds selected 0.900
con.execute("""
    CREATE TABLE fold_thresh (
        fold INT,
        selected_threshold FLOAT
    );
    INSERT INTO fold_thresh VALUES (0, 0.900), (1, 0.900), (2, 0.900), (3, 0.900), (4, 0.900);
""")

print("Streaming canonical OOF to generate E03_OOF_PREDICTIONS.tsv...")
con.execute(f"""
    COPY (
        SELECT 
            o.source1_entity_id,
            o.matched_entity_id,
            o.target,
            o.fold,
            o.oof_score,
            t.selected_threshold,
            CASE WHEN o.oof_score >= t.selected_threshold THEN 1 ELSE 0 END AS selected_prediction
        FROM read_csv('{OOF_IN}', delim='\\t', header=true) o
        JOIN fold_thresh t ON o.fold = t.fold
    ) TO '{OOF_OUT}' (DELIMITER '\\t', HEADER TRUE)
""")

print(f"File written in {time.time() - t0:.2f}s")

# Integrity checks
print("Running integrity checks...")
t_chk = time.time()
res = con.execute(f"""
    SELECT 
        count(*) AS total_rows,
        count(distinct source1_entity_id) AS distinct_s1,
        sum(case when fold is null then 1 else 0 end) AS null_folds,
        sum(case when oof_score is null then 1 else 0 end) AS null_scores,
        sum(case when selected_threshold != 0.900 then 1 else 0 end) AS bad_thresh,
        sum(selected_prediction) AS total_predicted_matches,
        sum(case when selected_prediction = 1 and target = 1 then 1 else 0 end) AS tp,
        sum(case when selected_prediction = 1 and target = 0 then 1 else 0 end) AS fp
    FROM read_csv('{OOF_OUT}', delim='\\t', header=true)
""").fetchone()

total_rows, dist_s1, null_folds, null_scores, bad_thresh, tot_p, tp, fp = res
print(f"Total Rows (expected 54,592,725): {total_rows}")
print(f"Distinct S1 entities (expected 2,206,821): {dist_s1}")
print(f"Null folds: {null_folds}, Null scores: {null_scores}, Bad thresholds: {bad_thresh}")
print(f"Total Predicted Matches: {tot_p} (TP={tp}, FP={fp})")
print(f"Checks completed in {time.time() - t_chk:.2f}s")

assert total_rows == 54592725, f"Row count mismatch: {total_rows}"
assert null_folds == 0 and null_scores == 0 and bad_thresh == 0
assert tot_p == 4497417, f"Predicted matches mismatch: {tot_p} vs expected 4497417"
assert tp == 4327648, f"TP mismatch: {tp} vs expected 4327648"
assert fp == 169769, f"FP mismatch: {fp} vs expected 169769"
print("ALL INTEGRITY CHECKS PASSED PERFECTLY!")
