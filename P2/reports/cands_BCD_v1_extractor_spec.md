# Extractor Spec: `cands_BCD_v1`
- exact regex: `[0-9]+[A-Za-z]?`
- first-match semantics: `0` index in DuckDB's `regexp_extract()`
- input column: `business_address`
- normalized/raw: normalized
- case handling: `A-Za-z` implicitly matching lowercased string
- no-match behavior: `''`
- engine: DuckDB
- quirks: fuzzy matching captures embedded alphanumeric clusters (e.g., `3rd` -> `3r`)
