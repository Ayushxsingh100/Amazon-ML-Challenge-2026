# Extractor Spec: `cands_B_v1`

## Definition
The canonical house extractor used in the `cands_B_v1` generation logic perfectly replicates the original P1 pipeline behavior.

- **Exact regex**: `[0-9]+[A-Za-z]?`
- **First-match semantics**: Uses the `0` index in DuckDB's `regexp_extract()` to pull the *first* occurrence of digits (optionally followed by a single letter) found anywhere within the string.
- **Input column**: Applied to `business_address`.
- **Normalization state**: Applied to *normalized* addresses (i.e. lowercase, standardized). The regex itself targets `TRIM(business_address)`.
- **Case handling**: The regex targets `A-Za-z`, but since the input `business_address` has already been lowercased during normalization, it practically matches `a-z`.
- **No-match behavior**: Returns an empty string `''`. (Note: The SQL `JOIN` clauses strictly enforce `house <> ''`, preventing entities with missing house numbers from matching each other).
- **Engine**: DuckDB's built-in RE2 regex engine.
- **Known Quirks**: Because it lacks word boundaries (`\b`), it acts as a fuzzy substring match. For example, it extracts `70w` from `s70w22100` and `3r` from `3rd`. This behavior is crucial for achieving the 92-93% P1 recall on this highly messy dataset. Enforcing strict word boundaries causes the loss of roughly ~15,000 true candidate pairs.
