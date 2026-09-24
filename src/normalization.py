"""
src/normalization.py - High performance text cleaning and entity normalization.
Includes Unicode handling, legal entity suffix stripping/standardization,
street address abbreviation normalization, and canonical country code mapping.
"""

import re
import unicodedata
from typing import Dict, Optional, Set, Union
import pandas as pd


# Canonical Country mapping
COUNTRY_MAP = {
    "us": "US",
    "usa": "US",
    "united states": "US",
    "united states of america": "US",
    "india": "IN",
    "in": "IN",
    "bharat": "IN",
    "france": "FR",
    "fr": "FR",
}

# Legal entity abbreviations and forms
LEGAL_ENTITY_PATTERNS = [
    (r"\b(private\s+limited|pvt\.?\s*ltd\.?)\b", " pvt ltd "),
    (r"\b(public\s+limited|pub\.?\s*ltd\.?)\b", " pub ltd "),
    (r"\b(limited|ltd\.?)\b", " ltd "),
    (r"\b(incorporated|inc\.?)\b", " inc "),
    (r"\b(corporation|corp\.?)\b", " corp "),
    (r"\b(limited\s+liability\s+company|l\.?l\.?c\.?)\b", " llc "),
    (r"\b(limited\s+liability\s+partnership|l\.?l\.?p\.?)\b", " llp "),
    (r"\b(soci[eé]t[eé]\s+[aà]\s+responsabilit[eé]\s+limit[eé]e|s\.?a\.?r\.?l\.?)\b", " sarl "),
    (r"\b(soci[eé]t[eé]\s+par\s+actions\s+simplifi[eé]e|s\.?a\.?s\.?)\b", " sas "),
    (r"\b(soci[eé]t[eé]\s+civile\s+immobili[eè]re|s\.?c\.?i\.?)\b", " sci "),
    (r"\b(company|co\.?)\b", " co "),
]

# Common address token standardizations
ADDRESS_PATTERNS = [
    (r"\b(post\s+office\s+box|p\.?o\.?\s*box)\b", " pobox "),
    (r"\b(streets?|st\.?)\b", " st "),
    (r"\b(roads?|rd\.?)\b", " rd "),
    (r"\b(avenues?|ave\.?)\b", " ave "),
    (r"\b(drives?|dr\.?)\b", " dr "),
    (r"\b(boulevards?|blvd\.?|boul\.?)\b", " blvd "),
    (r"\b(lanes?|ln\.?)\b", " ln "),
    (r"\b(courts?|ct\.?)\b", " ct "),
    (r"\b(highways?|hwy\.?)\b", " hwy "),
    (r"\b(parkways?|pkwy\.?)\b", " pkwy "),
    (r"\b(apartments?|apt\.?)\b", " apt "),
    (r"\b(suites?|ste\.?)\b", " ste "),
    (r"\b(units?|unit\.?)\b", " unit "),
    (r"\b(buildings?|bldg\.?)\b", " bldg "),
    (r"\b(floors?|fl\.?)\b", " fl "),
    (r"\b(rues?|r\.)\b", " rue "),
]

COMPILED_LEGAL = [(re.compile(p, re.IGNORECASE), repl) for p, repl in LEGAL_ENTITY_PATTERNS]
COMPILED_ADDRESS = [(re.compile(p, re.IGNORECASE), repl) for p, repl in ADDRESS_PATTERNS]
PUNCT_REGEX = re.compile(r"[^\w\s]", re.UNICODE)
MULTI_SPACE_REGEX = re.compile(r"\s+")


def remove_accents(text: str) -> str:
    """Normalize accented characters while preserving multilingual scripts like Devanagari."""
    if not text:
        return ""
    # NFKD decomposes accented latin characters into base char + combining mark
    nfkd_form = unicodedata.normalize("NFKD", text)
    # Filter out nonspacing mark characters (Mn)
    return "".join(c for c in nfkd_form if unicodedata.category(c) != "Mn")


def normalize_country(country: Optional[str]) -> str:
    """Normalize country string to uppercase canonical code ('US', 'IN', 'FR', etc.)."""
    if country is None or pd.isna(country):
        return "UNKNOWN"
    c = str(country).strip().lower()
    return COUNTRY_MAP.get(c, c.upper())


def normalize_business_name(
    name: Optional[str],
    strip_legal_suffixes: bool = False,
    preserve_non_latin: bool = True,
) -> str:
    """
    Normalize business name:
    1. Handle None / empty
    2. Lowercase and accent removal
    3. Clean leading/trailing symbols (e.g. '<<', '--', '+')
    4. Normalize legal entity notations (Inc, LLC, Pvt Ltd, SARL)
    5. Clean punctuation and excess whitespace
    """
    if name is None or pd.isna(name):
        return ""
    text = str(name).strip()
    if not text:
        return ""
    
    # Accent removal for latin characters
    text = remove_accents(text)
    text = text.lower()
    
    # Standardize or strip legal entity terms
    for pattern, repl in COMPILED_LEGAL:
        text = pattern.sub("" if strip_legal_suffixes else repl, text)
    
    # Remove punctuation
    text = PUNCT_REGEX.sub(" ", text)
    # Collapse whitespace
    text = MULTI_SPACE_REGEX.sub(" ", text).strip()
    return text


def normalize_address(address: Optional[str]) -> str:
    """
    Normalize business address:
    1. Handle None / empty
    2. Accent removal and lowercasing
    3. Standardize address terms (st, rd, ave, dr, pobox, etc.)
    4. Remove punctuation and collapse spaces
    """
    if address is None or pd.isna(address):
        return ""
    text = str(address).strip()
    if not text:
        return ""
    
    text = remove_accents(text)
    text = text.lower()
    
    # Standardize street and unit terms
    for pattern, repl in COMPILED_ADDRESS:
        text = pattern.sub(repl, text)
        
    text = PUNCT_REGEX.sub(" ", text)
    text = MULTI_SPACE_REGEX.sub(" ", text).strip()
    return text


def extract_blocking_tokens(normalized_name: str, min_len: int = 2) -> Set[str]:
    """Extract informative non-stopword tokens from normalized name for indexing."""
    stopwords = {"and", "the", "of", "in", "for", "a", "an", "co", "inc", "llc", "ltd", "pvt"}
    tokens = normalized_name.split()
    return {t for t in tokens if len(t) >= min_len and t not in stopwords}


def normalize_dataframe(
    df: pd.DataFrame,
    name_col: str = "business_name",
    address_col: str = "business_address",
    country_col: str = "country",
) -> pd.DataFrame:
    """
    Apply full normalization pipeline across dataframe columns in vector/batch mode.
    Adds `norm_name`, `norm_address`, and `norm_country` columns.
    """
    out = df.copy()
    if name_col in out.columns:
        out["norm_name"] = out[name_col].fillna("").apply(normalize_business_name)
    if address_col in out.columns:
        out["norm_address"] = out[address_col].fillna("").apply(normalize_address)
    if country_col in out.columns:
        out["norm_country"] = out[country_col].fillna("").apply(normalize_country)
    return out
