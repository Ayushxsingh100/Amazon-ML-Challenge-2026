# Experiment EXP-RET-04: Locality + Name Retrieval

## Overview
- **Goal**: Target true entity pairs where house numbers are absent, divergent, or unparsed, but locality / address prefixes and business name prefixes match.
- **Hypothesis**: Many commercial and industrial entities share an identical street/locality address prefix (first 10 normalized alphanumeric characters of `address_clean`) and a common name prefix (first 3 alphanumeric characters of `name_clean`), which V2 missed because V2 rules strictly required matching house numbers or exact names.
- **Rule Formulation**:
  1. `country + address_prefix_10 + name_prefix_3`:
     ```sql
     s1.country = tgt.country
     AND LEFT(s1.address_clean, 10) = LEFT(tgt.address_clean, 10)
     AND s1.name_prefix_3 = tgt.name_prefix_3
     AND LENGTH(s1.address_clean) >= 10
     AND LENGTH(tgt.address_clean) >= 10
     ```
  2. `country + postal_code (len >= 5) + root_token (len >= 4)`:
     ```sql
     s1.country = tgt.country
     AND s1.postal_code = tgt.postal_code
     AND s1.root_token = tgt.root_token
     AND LENGTH(s1.postal_code) >= 5
     AND LENGTH(s1.root_token) >= 4
     ```

## Quantitative Results (on top of V2)
- **Combined Recall**: **66.9695%** (**+1.2072%** over V2 baseline 65.7623%)
- **New Captured Pairs**: **+92,208** (Total captured: 5,115,376 vs V2 5,023,168)
- **S2 Recall**: **67.0768%** (+38,986 new captured true pairs)
- **S3 Recall**: **66.8691%** (+53,222 new captured true pairs)
- **Total Candidates**: **68,649,982** (Delta over V2: +1,317,458, an increase of only 1.95%)
- **Zero-Candidate S1**: **81,195** (**-12,848** reduction from V2's 94,043)
- **Fanout Distribution**:
  - $p50$: 6.0
  - $p90$: 83.0
  - $p95$: 158.0 ($\le 300$ **PASS**)
  - $p99$: 353.0
  - $p99.9$: 865.0
  - $\max$: 3,306

## Decision
**ACCEPTED**. EXP-RET-04 recovers over 92,000 true pairs missed by V2 with minimal fanout growth (+1.32M pairs, $p95=158$).
