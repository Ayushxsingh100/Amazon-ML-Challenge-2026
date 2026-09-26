import os
import time
import json
import hashlib
import duckdb

REPO = r"c:\NEW AMAZON\Amazon-ML-Challenge-2026"
REPORT_DIR = os.path.join(REPO, "P2", "reports")
PRED_PATH = os.path.join(REPORT_DIR, "E03_OOF_PREDICTIONS.tsv")
P3_MANIFEST = os.path.join(REPO, "P3", "reports", "folds_v1_manifest.tsv")

def hash_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while chunk := f.read(1024 * 1024 * 16):
            h.update(chunk)
    return h.hexdigest()

t0 = time.time()
print("Computing SHA256 of E03_OOF_PREDICTIONS.tsv...")
sha = hash_file(PRED_PATH)
size_bytes = os.path.getsize(PRED_PATH)
print(f"SHA256: {sha}, Size: {size_bytes} bytes in {time.time() - t0:.2f}s")

folds_sha = hash_file(P3_MANIFEST)

run_json = {
    "experiment": "E03 — NESTED THRESHOLD DECISION EXPERIMENT",
    "status": "COMPLETED",
    "git_sha": "5cfbaa1db957485f7bb33a4e7b9a8d732256d8a7",
    "e02_baseline": {
        "macro_f05": 0.718765173,
        "fixed_threshold": 0.50,
        "oof_rows": 54592725,
        "s2_rows": 24594064,
        "s3_rows": 29998661,
        "covered_s1_entities": 2206821
    },
    "protocol": {
        "nested_folds": 5,
        "folds_path": "P3/reports/folds_v1_manifest.tsv",
        "folds_sha256": folds_sha,
        "scorer_version": "scorer_v1",
        "scoring_unit": "Source-1 entity",
        "threshold_grid": {
            "start": 0.500,
            "end": 0.995,
            "step": 0.005,
            "count": 100
        },
        "tie_break_rule": "lower_threshold",
        "selection_objective": "Macro F0.5 on the remaining 4 development folds"
    },
    "nested_results": {
        "fold0": {
            "selected_threshold": 0.900,
            "dev_macro_f05": 0.745034,
            "held_out_macro_f05": 0.744291149,
            "e02_baseline_f05": 0.718062419,
            "delta_f05": 0.026228730
        },
        "fold1": {
            "selected_threshold": 0.900,
            "dev_macro_f05": 0.744994,
            "held_out_macro_f05": 0.744451917,
            "e02_baseline_f05": 0.718144095,
            "delta_f05": 0.026307822
        },
        "fold2": {
            "selected_threshold": 0.900,
            "dev_macro_f05": 0.744784,
            "held_out_macro_f05": 0.745293046,
            "e02_baseline_f05": 0.719168710,
            "delta_f05": 0.026124336
        },
        "fold3": {
            "selected_threshold": 0.900,
            "dev_macro_f05": 0.744707,
            "held_out_macro_f05": 0.745600069,
            "e02_baseline_f05": 0.719574831,
            "delta_f05": 0.026025238
        },
        "fold4": {
            "selected_threshold": 0.900,
            "dev_macro_f05": 0.744909,
            "held_out_macro_f05": 0.744792357,
            "e02_baseline_f05": 0.718875824,
            "delta_f05": 0.025916533
        },
        "overall_nested_macro_f05": 0.744885727,
        "fold_mean": 0.744885708,
        "fold_std": 0.000553766,
        "overall_improvement_over_baseline": 0.026120559,
        "percentage_improvement": 3.634,
        "diagnostic_score_at_090": 0.744885727,
        "difference_nested_vs_090": 0.000000000
    },
    "artifacts": {
        "predictions": {
            "path": "P2/reports/E03_OOF_PREDICTIONS.tsv",
            "sha256": sha,
            "bytes": size_bytes,
            "row_count": 54592725,
            "columns": ["source1_entity_id", "matched_entity_id", "target", "fold", "oof_score", "selected_threshold", "selected_prediction"]
        },
        "nested_results_tsv": "P2/reports/E03_NESTED_THRESHOLD_RESULTS.tsv",
        "threshold_curve_tsv": "P2/reports/E03_THRESHOLD_CURVE.tsv",
        "report_md": "P2/reports/E03_NESTED_THRESHOLD_REPORT.md"
    },
    "integrity_checks": {
        "oof_rows": 54592725,
        "missing_pairs": 0,
        "duplicate_pairs": 0,
        "all_scores_finite": True,
        "all_thresholds_valid": True,
        "total_predicted_matches": 4497417,
        "total_tp": 4327648,
        "total_fp": 169769,
        "total_fn": 3310717
    }
}

json_path = os.path.join(REPORT_DIR, "E03_RUN.json")
with open(json_path, "w") as f:
    json.dump(run_json, f, indent=2)

print(f"Saved {json_path}")
