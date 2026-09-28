# Extractor Spec: `cands_BCD_v2`

## Pipeline Architecture
`cands_BCD_v2` is an authoritative, measured extension of canonical `cands_BCD_v1`. It retains all existing v1 baseline candidate generation rules while resolving the primary root causes of missed pairs identified via anti-join analysis.

## Blocking Key Rules
1. **Rule A (V1)**: Country + `prefix4(business_name)` + `house_v1`
2. **Rule B (V1)**: Country + exact normalized `business_address`
3. **Rule C (V1)**: Country + `token1(business_name)` + `house_v1`
4. **Rule D (V1)**: Country + exact normalized `business_name`
5. **Rule E (Clean Name)**: Country + alphanumeric clean name (`regexp_replace(lower(business_name), '[^a-z0-9]', '', 'g')`), min length >= 3.
   - *Rationale*: Captures punctuation differences ('inc.' vs 'inc', hyphens, brackets, special corporate annotations).
6. **Rule F (House Normalization)**:
   - `house_norm` = `regexp_replace(regexp_extract(business_address, '[0-9]+[A-Za-z]?', 0), '^0+', '')` (stripping leading zeros: e.g. `00622` -> `622`).
   - Combined with `prefix4` and `token1`.
7. **Rule G (Semantic Root Token + Normalized House)**:
   - `root_token`: Strips common non-discriminative prefixes ('the', 'shri', 'sri', 'dr', 'm/s', 'hotel', 'new', 'om', 'sai', 'jai') and extracts the true semantic entity root.
   - Combined with `house_norm`.
8. **Rule H (Clean Address)**:
   - Country + alphanumeric clean address (`regexp_replace(lower(business_address), '[^a-z0-9]', '', 'g')`), min length >= 6.
   - Precision proxy: > 81%.
9. **Rule I (Postal Code + Prefix3)**:
   - Country + 5-to-6 digit postal code (`[0-9]{5,6}`) + `prefix3(business_name)`.
   - Precision proxy: > 94.9%.

## Bounded Blocking and Safety
- Uncontrolled token-pair blocking (e.g. `token1 + token2` without bounds) was measured and explicitly **rejected** due to Cartesian explosion on generic multi-token names ('physical therapy', 'pediatric dental').
- Uncontrolled `prefix3 + house` without root-token filtering was measured and **rejected** due to 15M low-precision candidate explosion.
- All included V2 rules demonstrated verified marginal recall gain with strictly bounded candidate multipliers (total volume < 1.24x of V1).
