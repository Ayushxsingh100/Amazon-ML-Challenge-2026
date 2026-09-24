# Amazon ML Challenge 2026: Entity Resolution & Record Linkage

A scalable, high-recall entity resolution and candidate-generation pipeline for multi-source business record matching.

---

## 📁 Repository Structure

```text
amazon-ml-2026/
│
├── data/
│   ├── train/
│   │   ├── train_source1.tsv.part_*
│   │   ├── train_source2.tsv.part_*
│   │   ├── train_source3.tsv.part_*
│   │   └── train_ground_truth.tsv.part_*
│   │
│   └── test/
│       ├── test_source1.tsv.part_*
│       ├── test_source2.tsv.part_*
│       └── test_source3.tsv.part_*
│
├── notebooks/
│   ├── 01_data_audit.ipynb
│   ├── 02_normalization.ipynb
│   ├── 03_blocking.ipynb
│   └── 04_blocking_evaluation.ipynb
│
├── src/
│   ├── __init__.py
│   ├── loading.py
│   ├── normalization.py
│   ├── blocking.py
│   ├── evaluation.py
│   └── features.py
│
├── outputs/
├── reconstruct_data.sh
├── check.py
└── understand.py
```

---

## ⚡ Quick Start

### 1. Reconstructing Data Files
The dataset files contain ~26.4 million records (>2.5 GB total). To comply with GitHub's file limits, data files are stored in chunked parts (`*.tsv.part_*`).

To assemble all TSV files, run:
```bash
./reconstruct_data.sh
```
*(Alternatively, calling `src.load_source()` will automatically detect and stitch parts if the `.tsv` file is missing).*

### 2. Running Jupyter Notebooks
Navigate to `notebooks/` and follow the progression:
1. **`01_data_audit.ipynb`**: Complete profiling of row counts, missingness, country distribution, and ground-truth cardinality.
2. **`02_normalization.ipynb`**: Unicode diacritic stripping, legal entity standardization (`Inc`, `LLC`, `Pvt Ltd`, `SARL`), and address normalization.
3. **`03_blocking.ipynb`**: Scalable candidate generation using inverted indexing with IDF weights, stop-token pruning, and Top-$K$ budget.
4. **`04_blocking_evaluation.ipynb`**: Rigorous evaluation of Candidate Recall (Pair Completeness), Reduction Ratio, sample candidate pair inspection, and false-negative error analysis.

---

## 🛠️ Python Package (`src/`)

- **`src.loading`**: Memory-efficient stream/chunk loader and ground-truth parser.
- **`src.normalization`**: Multi-lingual text cleaning, company suffix normalization, and address standardization.
- **`src.blocking`**: `StandardBlocker` and `TokenInvertedIndexBlocker` for high-recall candidate reduction.
- **`src.evaluation`**: Standard entity resolution metrics (`compute_blocking_metrics`, `analyze_missed_pairs`).
- **`src.features`**: Pairwise feature engineering (token Jaccard, trigram similarity, sequence matcher, street number concordance).
