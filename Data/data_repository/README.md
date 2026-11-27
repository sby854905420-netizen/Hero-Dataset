# Complete data repository

The default benchmark corpus combines Golden Evidence parents and Negative Sample 1, Negative Sample 2, and Negative Sample 3. It contains **184,856 complete records**: 145,115 KG records, 17,821 tables, and 21,920 infoboxes.

The large JSONL files are hosted in the [v2.0.0 Release](https://github.com/sby854905420-netizen/Hero-Dataset/releases/tag/v2.0.0). A Git clone or automatically generated source archive contains the code and metadata, but not these corpus files.

From the repository root:

```bash
curl --fail --location --output data_repository.tar.gz \
  https://github.com/sby854905420-netizen/Hero-Dataset/releases/download/v2.0.0/data_repository.tar.gz
tar -xzf data_repository.tar.gz
```

The archive extracts to:

```text
Data/source_mapping.jsonl
Data/data_repository/kg/kb.jsonl
Data/data_repository/tables/tables.jsonl
Data/data_repository/infoboxes/infoboxes.jsonl
```

`source_mapping.jsonl` describes the same full corpus. Each JSONL corpus line preserves one complete parent source. KG `record_uid` and table/infobox `evidence_id` join to the mapping and Gold annotations.

The corpus contains complete context, including source content not labeled Gold. Only units listed in `Data/gold_annotations.jsonl` are confirmed Gold evidence. QA and Gold metadata are evaluation inputs; index the three corpus files rather than the metadata files.

The full corpus already includes all three Negative Samples. Download their [separate archives](../negative_samples/README.md) only when selecting or composing a corpus variant. Their records should not be appended again to the full corpus.

See [Data/README.md](../README.md) for schemas, counts, evidence selectors, and provenance.
