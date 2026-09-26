# Canonical Entity Dataset Schema Specification

**Date**: 2026-09-26  
**Module**: Person 1 (P1) Canonical Entity Layer  
**Storage Format**: Apache Parquet with ZSTD Compression  
**Sorting Order**: Deterministic `entity_id ASC`  

---

## 1. Design Principles

1. **Lossless Preservation**: Raw source columns (`business_name_raw`, `business_address_raw`, `country_raw`) are preserved exactly as provided in the raw competition TSVs without truncation, stripping, or mutation.
2. **Dual Representation**: Standardized text from normalized TSVs is kept side-by-side with raw strings to enable feature extraction and candidate retrieval across multiple representations.
3. **Deterministic Derived Keys**: High-precision blocking keys (prefixes, house numbers, postal codes, root tokens) are computed deterministically using standard SQL expressions.
4. **100% Entity Retention**: Zero rows are dropped. If an entity has empty or missing attributes, empty strings (`""`) are retained and preserved.

---

## 2. Field Definitions and Transformations

| Field Name | Data Type | Source | Derivation / Logic | Null Behavior | Empty-String Behavior | Real Example |
|---|---|---|---|---|---|---|
| `entity_id` | `VARCHAR` | Raw / Norm | Unique primary key string | Never null | Never empty | `"S1-773889195"` |
| `business_name_raw` | `VARCHAR` | Raw TSV | Unmodified source business name | Coalesced to `""` | Preserved as `""` | `"Prime Money"` |
| `business_address_raw` | `VARCHAR` | Raw TSV | Unmodified source business address | Coalesced to `""` | Preserved as `""` | `"17560 Ellis Road, Tahlequah, OK"` |
| `country_raw` | `VARCHAR` | Raw TSV | Unmodified source country code | Coalesced to `""` | Preserved as `""` | `"US"` |
| `business_name_normalized` | `VARCHAR` | Norm TSV | Standardized business name | Coalesced to `""` | Preserved as `""` | `"prime money"` |
| `business_address_normalized`| `VARCHAR` | Norm TSV | Standardized business address | Coalesced to `""` | Preserved as `""` | `"17560 ellis road, tahlequah, ok"` |
| `country_normalized` | `VARCHAR` | Norm TSV | Standardized country code | Coalesced to `""` | Preserved as `""` | `"US"` |
| `name_clean` | `VARCHAR` | Derived | `regexp_replace(lower(trim(name_norm)), '[^a-z0-9]', '', 'g')` | Empty if input null | Empty if no alphanumeric chars | `"primemoney"` |
| `address_clean` | `VARCHAR` | Derived | `regexp_replace(lower(trim(addr_norm)), '[^a-z0-9]', '', 'g')` | Empty if input null | Empty if no alphanumeric chars | `"17560ellisroadtahlequahok"` |
| `name_tokens` | `VARCHAR[]` | Derived | `string_split(trim(regexp_replace(name_norm, '\s+', ' ', 'g')), ' ')` | Empty array | Single-element empty array | `["prime", "money"]` |
| `address_tokens` | `VARCHAR[]` | Derived | `string_split(trim(regexp_replace(addr_norm, '\s+', ' ', 'g')), ' ')` | Empty array | Single-element empty array | `["17560", "ellis", "road,", "tahlequah,", "ok"]` |
| `name_prefix_1` | `VARCHAR` | Derived | `LEFT(TRIM(name_norm), 1)` | Empty string | Empty string | `"p"` |
| `name_prefix_2` | `VARCHAR` | Derived | `LEFT(TRIM(name_norm), 2)` | Empty string | Empty string | `"pr"` |
| `name_prefix_3` | `VARCHAR` | Derived | `LEFT(TRIM(name_norm), 3)` | Empty string | Empty string | `"pri"` |
| `name_prefix_4` | `VARCHAR` | Derived | `LEFT(TRIM(name_norm), 4)` | Empty string | Empty string | `"prim"` |
| `first_token` | `VARCHAR` | Derived | `split_part(TRIM(name_norm), ' ', 1)` | Empty string | Empty string | `"prime"` |
| `root_token` | `VARCHAR` | Derived | Common prefix stripped (Rule G) | Empty string | Empty string | `"prime"` |
| `house_number` | `VARCHAR` | Derived | `regexp_extract(TRIM(addr_norm), '[0-9]+[A-Za-z]?', 0)` | Empty string | Empty string if no house num | `"17560"` |
| `house_number_norm` | `VARCHAR` | Derived | `regexp_replace(house_number, '^0+', '')` | Empty string | Empty string if no house num | `"17560"` |
| `postal_code` | `VARCHAR` | Derived | `regexp_extract(TRIM(addr_norm), '(?:^\|[^0-9])([0-9]{5,6})(?:[^0-9]\|$)', 1)` | Empty string | Empty string if no 5-6 digit zip | `"17560"` |
| `name_alnum` | `VARCHAR` | Derived | Alias to `name_clean` | Empty string | Empty string | `"primemoney"` |
| `address_alnum` | `VARCHAR` | Derived | Alias to `address_clean` | Empty string | Empty string | `"17560ellisroadtahlequahok"` |

---

## 3. Storage Hierarchy

```
P1/data/entities/
├── train/
│   ├── source1/train_s1_entities.parquet  (2,206,821 rows)
│   ├── source2/train_s2_entities.parquet  (5,034,616 rows)
│   └── source3/train_s3_entities.parquet  (5,285,603 rows)
└── test/
    ├── source1/test_s1_entities.parquet   (1,732,544 rows)
    ├── source2/test_s2_entities.parquet   (4,887,273 rows)
    └── source3/test_s3_entities.parquet   (5,082,316 rows)
```

Total entity count across all 6 canonical files: **24,228,873 rows**.
