#!/usr/bin/env bash
# Amazon ML Challenge 2026 - Reconstruct Data TSV files from git split parts
set -e

echo "=== Reconstructing Amazon ML Challenge 2026 Data Files ==="

reconstruct_file() {
    local target="$1"
    local dir=$(dirname "$target")
    local base=$(basename "$target")
    
    if [ -f "$target" ]; then
        echo "[EXISTS] $target is already present."
    else
        echo "[MERGING] Combining parts for $target ..."
        cat "${target}.part_"* > "$target"
        echo "[DONE] Successfully created $target ($(du -h "$target" | cut -f1))"
    fi
}

# Train sources
reconstruct_file "data/train/train_source1.tsv"
reconstruct_file "data/train/train_source2.tsv"
reconstruct_file "data/train/train_source3.tsv"
reconstruct_file "data/train/train_ground_truth.tsv"

# Test sources
reconstruct_file "data/test/test_source1.tsv"
reconstruct_file "data/test/test_source2.tsv"
reconstruct_file "data/test/test_source3.tsv"

echo "=== All data files ready! ==="
