import os
import duckdb
import pandas as pd

REPO = r"c:\NEW AMAZON\Amazon-ML-Challenge-2026"
DB_PATH = os.path.join(REPO, "P2", "data", "duckdb_slices_tmp.db")
OUT_SLICES = os.path.join(REPO, "P2", "reports", "E02_ENTITY_ERROR_SLICES.tsv")

con = duckdb.connect(DB_PATH)

base_chk = con.execute("SELECT count(*), avg(f05_actual), avg(f05_cand_oracle), avg(f05_ranking_oracle) FROM s1_master").fetchone()
print(f"Entities: {base_chk[0]}, Actual Macro F0.5: {base_chk[1]:.9f}, Cand Oracle: {base_chk[2]:.9f}, Ranking Oracle: {base_chk[3]:.9f}")

TOTAL_POP = base_chk[0]
TOTAL_LOSS = 1.0 - base_chk[1]

slices_def = [
    # 1. zero candidates vs has candidates
    ("candidate_coverage", "zero_candidates", "total_cands = 0"),
    ("candidate_coverage", "has_candidates", "total_cands > 0"),
    
    # 2-4. captured truth status
    ("truth_capture_status", "true_no_match", "A = 0"),
    ("truth_capture_status", "zero_captured_truth", "A > 0 AND K = 0"),
    ("truth_capture_status", "partial_truth_capture", "A > 0 AND K > 0 AND K < A"),
    ("truth_capture_status", "full_truth_capture", "A > 0 AND K = A"),
    
    # 5. truth size
    ("truth_size", "truth_size_0", "A = 0"),
    ("truth_size", "truth_size_1", "A = 1"),
    ("truth_size", "truth_size_2plus", "A >= 2"),
    
    # 6. candidate-count buckets
    ("candidate_count_bucket", "cands_0", "total_cands = 0"),
    ("candidate_count_bucket", "cands_1", "total_cands = 1"),
    ("candidate_count_bucket", "cands_2_to_5", "total_cands BETWEEN 2 AND 5"),
    ("candidate_count_bucket", "cands_6_to_10", "total_cands BETWEEN 6 AND 10"),
    ("candidate_count_bucket", "cands_11_to_25", "total_cands BETWEEN 11 AND 25"),
    ("candidate_count_bucket", "cands_26_to_50", "total_cands BETWEEN 26 AND 50"),
    ("candidate_count_bucket", "cands_51_to_100", "total_cands BETWEEN 51 AND 100"),
    ("candidate_count_bucket", "cands_100plus", "total_cands > 100"),
    
    # 7. top-score vs second-score margin
    ("score_margin", "margin_very_small_lt_0.05", "total_cands >= 2 AND margin < 0.05"),
    ("score_margin", "margin_small_0.05_0.20", "total_cands >= 2 AND margin >= 0.05 AND margin < 0.20"),
    ("score_margin", "margin_medium_0.20_0.50", "total_cands >= 2 AND margin >= 0.20 AND margin < 0.50"),
    ("score_margin", "margin_large_gte_0.50", "total_cands >= 2 AND margin >= 0.50"),
    ("score_margin", "single_candidate_no_margin", "total_cands = 1"),
    ("score_margin", "zero_candidates_no_margin", "total_cands = 0"),
    
    # 8. high-confidence false positives
    ("error_modes", "has_high_conf_fp_gte_0.80", "has_fp_80 = true"),
    ("error_modes", "no_high_conf_fp", "has_fp_80 = false"),
    
    # 9. low-confidence true positives
    ("error_modes", "has_low_conf_tp_0.50_0.60", "has_tp_50_60 = true"),
    ("error_modes", "no_low_conf_tp", "has_tp_50_60 = false"),
    
    # 10. false negatives
    ("error_modes", "has_false_negatives", "has_fn = true"),
    ("error_modes", "zero_false_negatives", "has_fn = false"),
    
    # 11-12. S2 vs S3 composition
    ("source_composition", "s2_only_truth", "A_s2 > 0 AND A_s3 = 0"),
    ("source_composition", "s3_only_truth", "A_s2 = 0 AND A_s3 > 0"),
    ("source_composition", "both_s2_and_s3_truth", "A_s2 > 0 AND A_s3 > 0"),
    ("source_composition", "neither_s2_nor_s3_truth", "A = 0"),
    
    # 13. rule provenance
    ("rule_provenance", "rule_b_only", "rule_provenance = 'B_ONLY'"),
    ("rule_provenance", "rule_c_only", "rule_provenance = 'C_ONLY'"),
    ("rule_provenance", "rule_d_only", "rule_provenance = 'D_ONLY'"),
    ("rule_provenance", "multiple_rules", "rule_provenance = 'MULTIPLE_RULES'"),
    ("rule_provenance", "zero_candidates", "rule_provenance = 'ZERO_CANDS'"),
    
    # 14. missing fields
    ("missing_fields", "missing_business_name", "missing_name = true"),
    ("missing_fields", "missing_business_address", "missing_addr = true"),
    ("missing_fields", "missing_house_number", "missing_house = true"),
    ("missing_fields", "all_fields_present", "missing_name = false AND missing_addr = false AND missing_house = false"),
    
    # 15. score bands around threshold 0.50
    ("threshold_neighborhood", "has_candidate_in_0.40_0.45", "has_band_40_45 = true"),
    ("threshold_neighborhood", "has_candidate_in_0.45_0.50", "has_band_45_50 = true"),
    ("threshold_neighborhood", "has_candidate_in_0.50_0.55", "has_band_50_55 = true"),
    ("threshold_neighborhood", "has_candidate_in_0.55_0.60", "has_band_55_60 = true"),
]

slice_results = []

for cat, name, cond in slices_def:
    res = con.execute(f"""
        SELECT 
            count(*),
            coalesce(avg(f05_actual), 0.0),
            coalesce(avg(f05_cand_oracle), 0.0),
            coalesce(avg(f05_ranking_oracle), 0.0),
            coalesce(sum(1.0 - f05_actual) / {TOTAL_POP}, 0.0) AS loss_impact,
            coalesce(avg(A), 0.0),
            coalesce(avg(P), 0.0),
            coalesce(avg(TP), 0.0),
            coalesce(avg(FP), 0.0),
            coalesce(avg(FN), 0.0)
        FROM s1_master
        WHERE {cond}
    """).fetchone()
    
    cnt, act_f05, cand_f05, rank_f05, loss_imp, mA, mP, mTP, mFP, mFN = res
    pct_pop = cnt * 100.0 / TOTAL_POP
    loss_imp = float(loss_imp) if loss_imp is not None else 0.0
    pct_loss = (loss_imp / TOTAL_LOSS) * 100.0 if TOTAL_LOSS > 0 else 0.0
    
    slice_results.append({
        "slice_category": cat,
        "slice_name": name,
        "condition": cond,
        "entity_count": cnt,
        "pct_of_all_s1": pct_pop,
        "actual_e02_macro_f05": float(act_f05),
        "cand_oracle_macro_f05": float(cand_f05),
        "ranking_oracle_macro_f05": float(rank_f05),
        "macro_f05_loss_impact": float(loss_imp),
        "pct_of_total_loss": float(pct_loss),
        "mean_actual_matches": float(mA),
        "mean_predicted_matches": float(mP),
        "mean_true_positives": float(mTP),
        "mean_false_positives": float(mFP),
        "mean_false_negatives": float(mFN)
    })
    print(f"[{cat}] {name}: count={cnt:,} ({pct_pop:.2f}%), actual F0.5={act_f05:.4f}, loss impact={loss_imp:.6f} ({pct_loss:.2f}%)")

df_slices = pd.DataFrame(slice_results)
df_slices.to_csv(OUT_SLICES, sep='\t', index=False)
print(f"\nSuccessfully saved error slices to {OUT_SLICES}")
con.close()
