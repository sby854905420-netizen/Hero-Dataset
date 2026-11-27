# Negative collections

The default [complete data repository](../data_repository/README.md) already includes Negative T1, T2, and T3. These separate Release assets let users identify and combine complete records into other difficulty variants.

| Collection | KG | Tables | Infoboxes | Total complete records | Download |
| --- | ---: | ---: | ---: | ---: | --- |
| T1 | 135 | 40 | 34 | 209 | [negative_t1.tar.gz](https://github.com/sby854905420-netizen/Hero-Dataset/releases/download/v2.0.0/negative_t1.tar.gz) |
| T2 | 9,499 | 5,431 | 5,652 | 20,582 | [negative_t2.tar.gz](https://github.com/sby854905420-netizen/Hero-Dataset/releases/download/v2.0.0/negative_t2.tar.gz) |
| T3 | 134,765 | 11,770 | 15,828 | 162,363 | [negative_t3.tar.gz](https://github.com/sby854905420-netizen/Hero-Dataset/releases/download/v2.0.0/negative_t3.tar.gz) |

## Download and extract

For example, from the repository root:

```bash
curl --fail --location --output negative_t2.tar.gz \
  https://github.com/sby854905420-netizen/Hero-Dataset/releases/download/v2.0.0/negative_t2.tar.gz
tar -xzf negative_t2.tar.gz
```

This produces:

```text
Data/negative_samples/negative_t2/kg/kb.jsonl
Data/negative_samples/negative_t2/tables/tables.jsonl
Data/negative_samples/negative_t2/infoboxes/infoboxes.jsonl
```

T1 and T3 use the same layout under `negative_t1/` and `negative_t3/`. The raw records have the same IDs and content as their copies in the complete repository. Their upstream provenance is in `Data/source_mapping.jsonl`; it is not duplicated in each negative archive.

## Construction

- **T1:** candidates proposed during automatic annotation but outside final human-confirmed Gold. Any complete parent used as Gold is excluded globally.
- **T2:** use BGE-M3 dense embeddings and the rewritten query to retrieve the top 100 chunks per question. Restore complete parents, deduplicate, and merge the candidates across all 500 questions. For each question, GPT-6.1 Sol with Ultra reasoning searches this shared repository for direct, joint, or useful partial support. Remove every selected supporting parent globally.
- **T3:** start with the remaining repository after excluding Gold, T1, T2, and previously discovered support. For each question, uniformly sample 500 chunks without replacement whose cosine similarity is strictly below 0.3, using seed `20261126`. Restore and merge complete parents, then apply the same shared-repository model screening and global parent removal as T2.

The screening receives the final rewritten question and human-confirmed answer. Evidence that only repeats background information already provided by the question is not removed on that basis. Screening is model-based and has not received an additional exhaustive human review; low embedding similarity or an unsuccessful search does not prove that all support is absent.

## Compose a variant

1. Collect the distinct `source_record_id` values in `Data/gold_annotations.jsonl` and select those complete parents from the full repository. This gives the 1,702 Gold parents.
2. Add the complete records from the chosen T1, T2, and/or T3 archives. These collections are distinct from the Gold parents and from one another.
3. Preserve record IDs, source content, QA, and Gold selectors. Record which collections were included when reporting results.

The union of Gold parents and all three collections is the default 184,856-record benchmark. A selected difficulty variant changes the retrieval environment; report it explicitly rather than presenting its scores as full-corpus results. The full-corpus source mapping remains a provenance superset when only some records are selected.

No removed supporting records, candidate pools, internal annotation logs, or extra question-specific Gold labels are included in these packages.
