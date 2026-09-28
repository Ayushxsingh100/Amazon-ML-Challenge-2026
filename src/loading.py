"""
src/loading.py - Efficient data loading utilities for Amazon ML Challenge 2026.
Handles TSV file loading, chunking, schema validation, and ground truth parsing.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np

# Default directories relative to project root
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"
OUTPUTS_DIR = BASE_DIR / "outputs"

SOURCE_COLUMNS = ["entity_id", "business_name", "business_address", "country"]
SOURCE_DTYPES = {
    "entity_id": "string",
    "business_name": "string",
    "business_address": "string",
    "country": "string",
}

GROUND_TRUTH_COLUMNS = ["source1_entity_id", "matched_entity_ids"]
GROUND_TRUTH_DTYPES = {
    "source1_entity_id": "string",
    "matched_entity_ids": "string",
}


def _ensure_file_exists(target_path: Path) -> Path:
    """If target file doesn't exist, check for .part_* split files and auto-merge them."""
    if target_path.exists():
        return target_path
    
    part_files = sorted(target_path.parent.glob(f"{target_path.name}.part_*"))
    if part_files:
        print(f"[loading.py] Auto-combining {len(part_files)} chunks for {target_path.name}...")
        with open(target_path, "wb") as outfile:
            for p in part_files:
                with open(p, "rb") as infile:
                    outfile.write(infile.read())
        print(f"[loading.py] Successfully created {target_path.name} ({target_path.stat().st_size / (1024*1024):.1f} MB)")
        return target_path

    raise FileNotFoundError(f"Source file not found and no .part_* chunks found: {target_path}")


def get_data_path(split: str, source_num: Union[int, str]) -> Path:
    """Resolve file path for a given split ('train' or 'test') and source (1, 2, or 3)."""
    split_dir = TRAIN_DIR if split == "train" else TEST_DIR
    filename = f"{split}_source{source_num}.tsv"
    path = split_dir / filename
    return _ensure_file_exists(path)


def get_ground_truth_path() -> Path:
    """Resolve path to train_ground_truth.tsv."""
    path = TRAIN_DIR / "train_ground_truth.tsv"
    return _ensure_file_exists(path)


def load_source(
    split: str = "train",
    source_num: Union[int, str] = 1,
    nrows: Optional[int] = None,
    usecols: Optional[List[str]] = None,
    chunksize: Optional[int] = None,
) -> Union[pd.DataFrame, pd.io.parsers.readers.TextFileReader]:
    """
    Load a specific source file (source 1, 2, or 3) for train or test split.
    
    Args:
        split: 'train' or 'test'
        source_num: 1, 2, or 3
        nrows: Optional number of rows to load (useful for fast development/testing)
        usecols: Optional list of columns to load
        chunksize: Optional chunksize for iterator-based stream loading
    
    Returns:
        pd.DataFrame or TextFileReader iterator
    """
    file_path = get_data_path(split, source_num)
    cols = usecols or SOURCE_COLUMNS
    dtypes = {col: SOURCE_DTYPES[col] for col in cols if col in SOURCE_DTYPES}
    
    return pd.read_csv(
        file_path,
        sep="\t",
        nrows=nrows,
        usecols=cols,
        dtype=dtypes,
        na_values=["", "NaN", "nan", "NULL", "null"],
        keep_default_na=True,
        chunksize=chunksize,
    )


def load_ground_truth(
    nrows: Optional[int] = None,
    chunksize: Optional[int] = None,
) -> Union[pd.DataFrame, pd.io.parsers.readers.TextFileReader]:
    """
    Load train_ground_truth.tsv.
    
    Columns:
        source1_entity_id: Query ID from Source 1
        matched_entity_ids: Comma-separated matching entity IDs in Source 2 and/or Source 3
    """
    path = get_ground_truth_path()
    return pd.read_csv(
        path,
        sep="\t",
        nrows=nrows,
        dtype=GROUND_TRUTH_DTYPES,
        chunksize=chunksize,
    )


def parse_ground_truth_map(gt_df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Parse ground truth DataFrame into dictionary mapping S1 entity_id -> list of matched IDs.
    Handles empty or NaN matches gracefully.
    """
    mapping: Dict[str, List[str]] = {}
    for _, row in gt_df.iterrows():
        s1_id = str(row["source1_entity_id"]).strip()
        matched = row["matched_entity_ids"]
        if pd.isna(matched) or not str(matched).strip():
            mapping[s1_id] = []
        else:
            ids = [m.strip() for m in str(matched).split(",") if m.strip()]
            mapping[s1_id] = ids
    return mapping


def parse_ground_truth_pairs(gt_df: pd.DataFrame) -> pd.DataFrame:
    """
    Explode ground truth DataFrame into pairwise rows (source1_entity_id, matched_entity_id, source_type).
    """
    rows = []
    for _, row in gt_df.iterrows():
        s1_id = str(row["source1_entity_id"]).strip()
        matched = row["matched_entity_ids"]
        if pd.notna(matched) and str(matched).strip():
            for m_id in str(matched).split(","):
                m_clean = m_id.strip()
                if m_clean:
                    src_type = "S2" if m_clean.startswith("S2-") else ("S3" if m_clean.startswith("S3-") else "Other")
                    rows.append({
                        "source1_entity_id": s1_id,
                        "matched_entity_id": m_clean,
                        "match_source": src_type,
                    })
    return pd.DataFrame(rows)


def get_dataset_summary(split: str = "train", nrows: Optional[int] = None) -> pd.DataFrame:
    """
    Generate high-level metadata summary across all sources for the specified split:
    - Row count
    - Memory usage
    - Missing value counts per column
    - Unique entity_id count
    """
    summaries = []
    for s in [1, 2, 3]:
        df = load_source(split=split, source_num=s, nrows=nrows)
        row_count = len(df)
        unique_entities = df["entity_id"].nunique()
        missing_names = df["business_name"].isna().sum()
        missing_addresses = df["business_address"].isna().sum()
        missing_countries = df["country"].isna().sum()
        countries_list = df["country"].dropna().unique().tolist()
        
        summaries.append({
            "split": split,
            "source": f"Source {s}",
            "row_count": row_count,
            "unique_entity_ids": unique_entities,
            "duplicate_entity_ids": row_count - unique_entities,
            "missing_name_count": missing_names,
            "missing_address_count": missing_addresses,
            "missing_country_count": missing_countries,
            "unique_countries": len(countries_list),
            "top_countries": ", ".join([str(c) for c in countries_list[:5]]),
        })
    return pd.DataFrame(summaries)
