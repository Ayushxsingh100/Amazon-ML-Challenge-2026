# Amazon ML Challenge 2026: Entity Resolution & Record Linkage

A scalable, high-precision, and high-recall entity resolution and candidate-generation pipeline for multi-source business record matching.

---

## 📌 Problem Overview

In multi-source business record matching, entities (companies, suppliers, merchants) are represented across distinct, noisy data sources with varying schemas, missing attributes, non-standardized legal entities, and multi-lingual transliterations. The goal is to accurately match Source 1 entities with their corresponding records in Source 2 and Source 3 while optimizing for **Macro F0.5 score** (prioritizing high precision with strong recall).

---

## 📁 Repository Structure

```text
Amazon-ML-Challenge-2026/
├── src/                               # Core reusable library
│   ├── __init__.py
│   ├── loading.py                     # High-performance chunked data loader
│   ├── normalization.py               # Text cleaning, legal suffixes & address normalizer
│   ├── blocking.py                    # Scalable inverted index & candidate blockers
│   ├── features.py                    # Pairwise similarity & lexical feature engineering
│   └── evaluation.py                  # Evaluation metrics & error analysis
│
├── notebooks/                         # Exploratory data analysis & prototyping
│   ├── 01_data_audit.ipynb            # Profiling row counts, cardinality, missingness
│   ├── 02_normalization.ipynb         # Legal entity & address normalization
│   ├── 03_blocking.ipynb              # Inverted index blocking & candidate generation
│   └── 04_blocking_evaluation.ipynb   # Blocking recall & candidate reduction evaluation
│
├── P1/                                # Candidate Retrieval & Blocking Engine
│   ├── scripts/                       # Candidate generation & audit scripts
│   ├── manifests/                     # Reproducibility specs & artifact registry
│   └── reports/                       # Retrieval recall baselines & error breakdowns
│
├── P2/                                # Modeling, Ranking & Submission Pipeline
│   ├── scripts/                       # LightGBM training, threshold sweeps & inference
│   ├── models/                        # Serialized model trees & feature configurations
│   ├── manifests/                     # Experiment tracking manifests
│   └── reports/                       # Out-Of-Fold validation & residual error analysis
│
├── P3/                                # Validation Framework & Scoring Specs
│   ├── scripts/                       # 5-fold cross-validation runner
│   └── reports/                       # Validation freeze specs & fold manifests
│
├── audit/                             # End-to-end pipeline audit & integrity checks
│   ├── audit_core.py
│   └── FULL_PIPELINE_AUDIT.md
│
├── utils/                             # Utilities (e.g., submission format validator)
│   └── validate_submission.py
│
├── validation/                        # Scoring metrics & evaluation harnesses
│   ├── metrics.py
│   └── scorer_v1.py
│
├── requirements.txt                   # Python package dependencies
├── .gitignore                         # Comprehensive Git ignore rules
├── .gitattributes                     # Cross-platform line ending normalization
└── README.md                          # Project documentation
```

---

## ⚡ Quick Start & Setup

### 1. Environment Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/Ayushxsingh100/Amazon-ML-Challenge-2026.git
cd Amazon-ML-Challenge-2026

# Create and activate virtual environment (optional)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Dataset Preparation

Place the competition dataset files in a `data/` directory with the following structure:

```text
data/
├── train/
│   ├── train_source1.tsv
│   ├── train_source2.tsv
│   ├── train_source3.tsv
│   └── train_ground_truth.tsv
└── test/
    ├── test_source1.tsv
    ├── test_source2.tsv
    └── test_source3.tsv
```

---

## 🚀 Execution Pipeline

The complete pipeline follows a modular 5-stage architecture:

```mermaid
flowchart LR
    A["Raw Datasets"] --> B["1. Normalization & Preprocessing"]
    B --> C["2. Inverted Index Blocking & Retrieval"]
    C --> D["3. Pairwise Feature Engineering"]
    D --> E["4. LightGBM Classification & Ranking"]
    E --> F["5. F0.5 Threshold Optimization & Submission"]
```

### Stage 1: Data Normalization & Audit
Run the baseline audit to standardize Unicode diacritics, strip company legal suffixes (`Inc`, `LLC`, `Pvt Ltd`, `SARL`), and normalize country codes:
```bash
python audit/audit_core.py
```

### Stage 2: Candidate Retrieval & Blocking
Generate high-recall candidate pairs using hybrid inverted indexing, TF-IDF / BM25 token matching, and country-scoped blocking:
```bash
python P1/scripts/retrieval/generate_v3_candidates.py
```

### Stage 3: Feature Engineering & Model Training
Extract pairwise lexical, token Jaccard, sequence similarity, and coordinate concordance features, then train 5-fold cross-validated LightGBM ranking models:
```bash
python P2/scripts/e02_train_lgb.py
```

### Stage 4: Out-Of-Fold Validation & Threshold Sweep
Evaluate candidate recall and sweep classification margins to maximize Macro F0.5:
```bash
python P2/scripts/run_threshold_sweep.py
```

### Stage 5: Final Submission Generation
Generate the final test predictions and validate submission format:
```bash
python P2/scripts/phase5_generate_submission.py
python utils/validate_submission.py --submission-path output/matching_results.tsv
```

---

## 📊 Cross-Validation Strategy

- **5-Fold Entity-Level Split**: Partitioning is performed deterministically at the Source 1 entity level using `SHA256(seed + entity_id) % 5` to ensure 0% data leakage across folds.
- **Evaluation Metric**: Entity-level Macro F0.5 score:
  $$\text{Macro } F_{0.5} = \frac{1 + 0.5^2 \cdot \text{Precision} \cdot \text{Recall}}{0.5^2 \cdot \text{Precision} + \text{Recall}}$$

---

## 🛠️ Tech Stack & Dependencies

- **Data Processing**: `polars`, `pandas`, `pyarrow`, `duckdb`
- **Machine Learning**: `lightgbm`, `scikit-learn`, `scipy`
- **String Algorithms**: `rapidfuzz`, `sparse_dot_topn`
- **Environment**: Python 3.10+
