# Phase 3 — Candidate Retrieval Experiment Plan

**Module**: Person 1 (P1) Retrieval Architecture  
**Status**: Specification Only (No Implementation in Phase 1)  
**Basis**: Measured Evidence from 2,615,197 V2 Missed Ground Truth Pairs  

---

## 1. Principles of Controlled Retrieval

Per competition engineering protocol:
1. **Hypothesis-Driven**: Every retrieval experiment must target a specific, measured error bucket identified during Phase 1 forensic analysis.
2. **Incremental Measurement**: Recall gains must be measured incrementally over the V2 candidate baseline on identical evaluation splits.
3. **Candidate Budget Discipline**: Candidate volume must be strictly monitored to prevent quadratic combinatorial explosion into Person 2 feature extraction and LightGBM scoring.
4. **No Premature Implementation**: Implementation occurs exclusively in Phase 3 following Phase 2 baseline validation.

---

## 2. Proposed Retrieval Experiments

### EXP-RET-01: Soft-Name Matching via Sparse Token / Character N-Gram Overlap
- **Target Error Bucket**: `high_name_similarity_house_divergence` (619,003 missed pairs; 23.66% of total missed)
- **Observed Problem**: True entity pairs have highly similar business names ($JW \ge 0.85$), but differing, unparsed, or missing house numbers prevent Rules A, C, F, and G from firing.
- **Evidence**: 294,881 S2 pairs (23.50%) and 324,122 S3 pairs (23.83%) exhibit $JW \ge 0.85$ with house number divergence.
- **Proposed Method**: 
  - Block on identical country + high-frequency token inverted index (BM25 or character 3-gram TF-IDF).
  - Top-$K$ retrieval ($K \le 5$ per S1) filtered by cosine similarity threshold (e.g., $\ge 0.80$).
- **Input Representation**: `business_name_normalized` and `name_clean`.
- **Candidate Limit**: Maximum 5 candidates per S1 entity.
- **Recall Measurement**: Incremental true positives captured in training ground truth relative to V2 union.
- **Volume Measurement**: Total candidate count expansion (budget limit: $\le 10\text{M}$ additional pairs).
- **Acceptance Criteria**: Incremental recall gain $> 2.0\%$ with precision $> 5\%$ in top-5 candidates.
- **Risks**: High candidate volume if common commercial stop words ("store", "shop", "enterprise") are unweighted.

---

### EXP-RET-02: Relaxed Address Matching for Moderate Name Variations
- **Target Error Bucket**: `house_match_moderate_name_variation` (420,135 missed pairs; 16.07% of total missed)
- **Observed Problem**: Entities share the exact normalized house number within the same country, but business names vary due to word order, legal abbreviations, or secondary descriptor additions ($0.60 \le JW < 0.85$), failing Rules A, C, and G.
- **Evidence**: 168,503 S2 pairs (13.43%) and 251,632 S3 pairs (18.50%) have exact house match but fail prefix4, token1, and root_token blocking.
- **Proposed Method**:
  - Block on `(country, house_number_norm)`.
  - Apply secondary candidate filter: token Jaccard similarity $\ge 0.30$ or character 3-gram similarity $\ge 0.40$ on `business_name_normalized`.
- **Input Representation**: `house_number_norm`, `country_normalized`, `business_name_normalized`.
- **Candidate Limit**: Maximum 5 candidates per S1 entity.
- **Recall Measurement**: Count of previously missed GT pairs captured by this block.
- **Volume Measurement**: Track total pairs generated where house numbers have high cardinality.
- **Acceptance Criteria**: Incremental capture of $\ge 150,000$ true pairs with pair volume $< 8\text{M}$.
- **Risks**: Generic building numbers (e.g. house number "1", "2") can explode candidate volume without strict name thresholding.

---

### EXP-RET-03: Cross-Script Transliteration & Phonetic Normalization
- **Target Error Bucket**: `indic_devanagari_script` (279,753 missed pairs; 10.70% of total missed)
- **Observed Problem**: Business names are recorded in Devanagari / Indic script on one side and Latin script on the other (e.g. "श्री राम" vs "Shri Ram"), causing 100% character mismatch across Latin blocking rules.
- **Evidence**: 386,517 S2 pairs (30.80%) and 306,899 S3 pairs (22.56%) feature cross-script representation. 54.78% of all missed pairs are located in India.
- **Proposed Method**:
  - Offline deterministic transliteration layer using Indic transliteration (e.g., standard ITRANS or Soundex/DoubleMetaphone adapted for Indic phonology).
  - Derived Latin transliteration key `name_transliterated_latin`.
  - Block on `LEFT(name_transliterated_latin, 4)` and `house_number_norm`.
- **Input Representation**: `business_name_raw`, `country_normalized`, `house_number_norm`.
- **Candidate Limit**: Standard 1-to-many candidate union.
- **Recall Measurement**: Absolute pair recall on Indian entity subset in training GT.
- **Volume Measurement**: Pair count expansion specifically in `country = 'India'`.
- **Acceptance Criteria**: Capture $\ge 100,000$ previously unreachable cross-script pairs.
- **Risks**: Library dependency and transliteration ambiguity for polyphonic consonants.

---

### EXP-RET-04: Address Token Inverted Index for House-Number-Deficient Entities
- **Target Error Bucket**: `missing_house_number_one_side` & `missing_house_number_both_sides` (683,603 missed pairs; 26.14% of total missed)
- **Observed Problem**: Rural or unnumbered commercial locations (e.g. "Near Bus Stand, Main Road") lack digits, completely disabling all house-number-dependent blocking channels (Rules A, C, F, G).
- **Evidence**: 329,303 S2 missed pairs (26.24%) and 354,300 S3 missed pairs (26.05%) lack house numbers on one or both sides.
- **Proposed Method**:
  - Extract street/locality n-grams (e.g., 2-token address shingles) combined with 3-character name prefix.
  - Require `(country, name_prefix_3, address_shingle)`.
- **Input Representation**: `address_tokens`, `name_prefix_3`, `country_normalized`.
- **Candidate Limit**: Maximum 5 candidates per S1.
- **Recall Measurement**: Ground truth recall on subset where `house_number = ''`.
- **Volume Measurement**: Pair count generation.
- **Acceptance Criteria**: Incremental recall $> 1.5\%$ on unnumbered entity subset.
- **Risks**: Common address tokens ("main", "road", "street", "near") require inverse document frequency (IDF) filtering to avoid massive blocks.

---

## 3. Summary of Experiment Portfolio

| Exp ID | Targeted Error Bucket | Bucket Size | Proposed Channel | Key Risk |
|---|---|---|---|---|
| `EXP-RET-01` | High Name Sim / House Divergence | 619,003 | TF-IDF / 3-Gram Name Retrieval | Stop-word explosion |
| `EXP-RET-02` | House Match / Moderate Name Diff | 420,135 | Relaxed Name on House Key | Common house numbers |
| `EXP-RET-03` | Cross-Script Indic Text | 279,753 | Deterministic Transliteration Key | Ambiguous phonology |
| `EXP-RET-04` | Missing House Numbers | 683,603 | Address Shingle + Name Prefix | Locality word frequency |
