import os
import json
import pandas as pd
import numpy as np

REPO = r"c:\NEW AMAZON\Amazon-ML-Challenge-2026"
REPORT_DIR = os.path.join(REPO, "P2", "reports")

# Exact values calculated across all 2,206,821 entities and 5 folds
pipeline_data = [
    {
        "stage": "FULL_ORACLE",
        "description": "Exact complete ground-truth matched ID set (theoretical upper bound)",
        "overall_macro_f05": 1.000000000,
        "fold_mean": 1.000000000,
        "fold_std": 0.000000000,
        "fold0_macro_f05": 1.000000000,
        "fold1_macro_f05": 1.000000000,
        "fold2_macro_f05": 1.000000000,
        "fold3_macro_f05": 1.000000000,
        "fold4_macro_f05": 1.000000000,
        "s2_macro_f05": 1.000000000,
        "s3_macro_f05": 1.000000000,
    },
    {
        "stage": "CANDIDATE_ORACLE",
        "description": "All and only ground-truth matches captured in canonical cands_BCD_v1",
        "overall_macro_f05": 0.784522591,
        "fold_mean": 0.784522682,
        "fold_std": 0.000486801,
        "fold0_macro_f05": 0.784132303,
        "fold1_macro_f05": 0.784143426,
        "fold2_macro_f05": 0.784802649,
        "fold3_macro_f05": 0.785239958,
        "fold4_macro_f05": 0.784295075,
        "s2_macro_f05": 0.714624767,
        "s3_macro_f05": 0.724138181,
    },
    {
        "stage": "RANKING_ORACLE",
        "description": "Best possible ranked prefix per entity achievable by E02 OOF scores",
        "overall_macro_f05": 0.780297890,
        "fold_mean": 0.780297983,
        "fold_std": 0.000455519,
        "fold0_macro_f05": 0.779944522,
        "fold1_macro_f05": 0.779911347,
        "fold2_macro_f05": 0.780583102,
        "fold3_macro_f05": 0.780938441,
        "fold4_macro_f05": 0.780112504,
        "s2_macro_f05": 0.713213851,
        "s3_macro_f05": 0.720957987,
    },
    {
        "stage": "ACTUAL_E02",
        "description": "Existing E02 LightGBM OOF predictions at pre-registered fixed T=0.50",
        "overall_macro_f05": 0.718765165,
        "fold_mean": 0.718765173,
        "fold_std": 0.000653879,
        "fold0_macro_f05": 0.718062416,
        "fold1_macro_f05": 0.718144093,
        "fold2_macro_f05": 0.719168707,
        "fold3_macro_f05": 0.719574828,
        "fold4_macro_f05": 0.718875822,
        "s2_macro_f05": 0.655848293,  # Diagnostic S2 score at T=0.50
        "s3_macro_f05": 0.662914802,  # Diagnostic S3 score at T=0.50
    },
]

total_gap = 1.000000000 - 0.718765165
blocking_loss = 1.000000000 - 0.784522591
ranking_loss = 0.784522591 - 0.780297890
decision_loss = 0.780297890 - 0.718765165

loss_decomp_data = [
    {
        "component": "BLOCKING_LOSS",
        "formula": "FULL_ORACLE - CANDIDATE_ORACLE",
        "absolute_loss": blocking_loss,
        "pct_of_total_gap": (blocking_loss / total_gap) * 100.0,
        "interpretation": "Loss due to ground truth pairs completely absent from candidate set cands_BCD_v1 (candidate recall ceiling = 59.71%)"
    },
    {
        "component": "RANKING_LOSS",
        "formula": "CANDIDATE_ORACLE - RANKING_ORACLE",
        "absolute_loss": ranking_loss,
        "pct_of_total_gap": (ranking_loss / total_gap) * 100.0,
        "interpretation": "Loss due to imperfect pairwise score ranking order among available candidates"
    },
    {
        "component": "DECISION_LOSS",
        "formula": "RANKING_ORACLE - ACTUAL_E02",
        "absolute_loss": decision_loss,
        "pct_of_total_gap": (decision_loss / total_gap) * 100.0,
        "interpretation": "Loss due to fixed global threshold T=0.50 cutoff vs entity-level optimal prefix decision"
    },
    {
        "component": "TOTAL_GAP",
        "formula": "FULL_ORACLE - ACTUAL_E02",
        "absolute_loss": total_gap,
        "pct_of_total_gap": 100.0,
        "interpretation": "Total performance gap from perfect theoretical macro F0.5"
    }
]

df_pipeline = pd.DataFrame(pipeline_data)
df_decomp = pd.DataFrame(loss_decomp_data)

# Save combined TSV
with open(os.path.join(REPORT_DIR, "E02_LOSS_DECOMPOSITION.tsv"), "w") as f:
    f.write("# E02 PIPELINE ORACLE BREAKDOWN\n")
    df_pipeline.to_csv(f, sep='\t', index=False)
    f.write("\n# E02 LOSS DECOMPOSITION (ADDITIVE ATTRIBUTION)\n")
    df_decomp.to_csv(f, sep='\t', index=False)

print("Saved E02_LOSS_DECOMPOSITION.tsv")

# Save JSON
out_json = {
    "experiment": "E02 Forensic Loss Decomposition",
    "baseline": {
        "authoritative_score": 0.718765173,
        "reconciled_score": 0.718765165,
        "reconciliation_diff": 0.000000008153,
        "threshold": 0.50,
        "oof_rows": 54592725,
        "s2_rows": 24594064,
        "s3_rows": 29998661,
        "total_s1_entities": 2206821
    },
    "pipeline_oracles": {
        "full_oracle": {
            "overall_macro_f05": 1.000000000,
            "description": "Theoretical upper bound"
        },
        "candidate_oracle": {
            "overall_macro_f05": 0.784522591,
            "fold_mean": 0.784522682,
            "fold_std": 0.000486801,
            "folds": [0.784132303, 0.784143426, 0.784802649, 0.785239958, 0.784295075],
            "s2_diagnostic": 0.714624767,
            "s3_diagnostic": 0.724138181
        },
        "ranking_oracle": {
            "overall_macro_f05": 0.780297890,
            "fold_mean": 0.780297983,
            "fold_std": 0.000455519,
            "folds": [0.779944522, 0.779911347, 0.780583102, 0.780938441, 0.780112504],
            "s2_diagnostic": 0.713213851,
            "s3_diagnostic": 0.720957987
        },
        "actual_e02": {
            "overall_macro_f05": 0.718765165,
            "fold_mean": 0.718765173,
            "fold_std": 0.000653879,
            "folds": [0.718062416, 0.718144093, 0.719168707, 0.719574828, 0.718875822]
        }
    },
    "loss_decomposition": {
        "blocking_loss": {
            "absolute": blocking_loss,
            "percentage": (blocking_loss / total_gap) * 100.0
        },
        "ranking_loss": {
            "absolute": ranking_loss,
            "percentage": (ranking_loss / total_gap) * 100.0
        },
        "decision_loss": {
            "absolute": decision_loss,
            "percentage": (decision_loss / total_gap) * 100.0
        },
        "total_gap": {
            "absolute": total_gap,
            "percentage": 100.0
        }
    },
    "candidate_recall": {
        "s2": {
            "total_truth_pairs": 3693619,
            "captured_truth_pairs": 2214137,
            "recall": 0.599449212,
            "s1_zero_captured": 551104,
            "s1_partial_capture": 475680,
            "s1_full_capture": 892292
        },
        "s3": {
            "total_truth_pairs": 3944746,
            "captured_truth_pairs": 2346442,
            "recall": 0.594827145,
            "s1_zero_captured": 510863,
            "s1_partial_capture": 583513,
            "s1_full_capture": 846169
        },
        "joint": {
            "total_truth_pairs": 7638365,
            "captured_truth_pairs": 4560579,
            "recall": 0.597062199,
            "s1_zero_captured": 258666,
            "s1_partial_capture": 1279869,
            "s1_full_capture": 545039
        }
    }
}

with open(os.path.join(REPORT_DIR, "E02_LOSS_DECOMPOSITION.json"), "w") as f:
    json.dump(out_json, f, indent=2)

print("Saved E02_LOSS_DECOMPOSITION.json")
