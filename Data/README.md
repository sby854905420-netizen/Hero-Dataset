# Hero Dataset data

Version 2.0.0 contains 500 human-confirmed questions, 665 answer items, and a retrieval corpus of 184,856 complete records. Questions comprise 245 `movies` and 255 `tvseries` items. There is one evaluation set and no additional train/development split.

JSONL files use UTF-8 with one JSON object per line. Complete source content and original identifiers are preserved. Large corpus files are downloaded from [GitHub Releases](https://github.com/sby854905420-netizen/Hero-Dataset/releases/tag/v2.0.0); cloning the repository alone does not download them.

## Files and responsibilities

After downloading and extracting the selected Release assets from the repository root:

```text
Data/
├── README.md
├── LICENSE.md
├── qa_pairs.jsonl
├── gold_annotations.jsonl
├── source_mapping.jsonl
├── data_repository/
│   ├── kg/kb.jsonl
│   ├── tables/tables.jsonl
│   └── infoboxes/infoboxes.jsonl
└── negative_samples/
    ├── negative_t1/
    ├── negative_t2/
    └── negative_t3/
```

The three negative directories are optional: the full corpus already includes their contents. Each separately downloaded negative directory has the same `kg/`, `tables/`, and `infoboxes/` file structure as the full corpus.

| Metadata file | One row represents | Information stored |
| --- | --- | --- |
| `qa_pairs.jsonl` | One question | ID, question, confirmed answers, domain |
| `gold_annotations.jsonl` | One question's Gold evidence | Question ID, exact evidence references, parent record IDs, selectors |
| `source_mapping.jsonl` | One complete corpus record | Record ID, source type, corpus file, upstream provenance |

`question_id` joins QA and Gold annotations. `source_record_id` joins Gold evidence, source mapping, and complete corpus records. These shared keys are intentional; question/answer text, evidence positions, and provenance are maintained in their respective files. There is no separate `evidence_sources.jsonl`, duplicated answer list in Gold annotations, or published human proof chain.

Methods use the question and domain as inputs and index only the corpus. Gold answers and question-specific evidence membership must not be included in retrieval documents or generation prompts.

## Questions and answers: `qa_pairs.jsonl`

| Field | Meaning |
| --- | --- |
| `question_id` | Unique string; IDs need not be consecutive |
| `question` | Final rewritten question |
| `answers` | Human-confirmed answer array, each item containing `id` and `label` |
| `domain` | `movies` or `tvseries` |

```json
{"question_id":"9","question":"What is the Original Language of the film Encanto?","answers":[{"id":"","label":"English"}],"domain":"movies"}
```

The current export leaves answer `id` empty and preserves the confirmed answer in `label`; it does not infer or inherit entity IDs from the original dataset. Answer matching, aliases, precision, and tolerances are defined under `evaluation/`.

## Gold evidence: `gold_annotations.jsonl`

This file has 500 rows containing 2,291 evidence uses and 2,222 distinct fine-grained references.

| Field | Meaning |
| --- | --- |
| `question_id` | Question whose answer is supported |
| `evidence` | List of confirmed Gold evidence units |
| `evidence[].ref` | Exact fine-grained evidence reference |
| `evidence[].source_record_id` | Complete parent record containing the unit |
| `evidence[].selector` | Position of that unit inside the parent |

An illustrative row showing the structure is:

```json
{"question_id":"example-question","evidence":[{"ref":"example-table#row=2","source_record_id":"example-table","selector":{"unit":"table_row","row_index":2}}]}
```

The illustration is not an additional benchmark item. Real references must be copied exactly, including URL encoding and suffixes such as `@occ`. A manually assigned reference may not embed its parent ID; use `source_record_id` rather than deriving a parent by splitting `ref`.

All positions and occurrence counts are zero-based:

| Evidence unit | Selector |
| --- | --- |
| KG statement target | `{"unit":"statement_target","target_index":0,"statement_guid":"Q…$…"}` |
| Table data row | `{"unit":"table_row","row_index":2,"whole_table":false}` |
| Complete table | `{"unit":"whole_table","whole_table":true}` |
| Infobox field | `{"unit":"infobox_field","field":"Language","field_occurrence":0,"row_index":15}` |
| Infobox image-caption row | `{"unit":"infobox_row","row_index":0}` |

Table `row_index` addresses the stored `rows` array, excluding separately stored headers. Infobox `field_occurrence` counts repeated equal field names in original order; an optional field `row_index` must agree with its stored row identity. For `infobox_row`, `row_index` addresses the `rows` array and the selected row must have `row_type="image_caption"`.

Gold labels apply only to these selected evidence units. A complete table, infobox, or KG parent is retained as context, and its unselected contents do not become Gold. The evidence list does not encode which units are jointly necessary, interchangeable, or sufficient on their own.

## Complete corpus: `data_repository/`

| Collection | KG | Tables | Infoboxes | Total |
| --- | ---: | ---: | ---: | ---: |
| Golden Evidence parents | 716 | 580 | 406 | 1,702 |
| Negative T1 | 135 | 40 | 34 | 209 |
| Negative T2 | 9,499 | 5,431 | 5,652 | 20,582 |
| Negative T3 | 134,765 | 11,770 | 15,828 | 162,363 |
| **Full corpus** | **145,115** | **17,821** | **21,920** | **184,856** |

The four collections contain distinct complete parent records. The released corpus combines them, preserving full records rather than selected evidence fragments. KG `record_uid` and table/infobox `evidence_id` provide complete record IDs. Preserve complete IDs, including any suffixes.

- **KG, `kg/kb.jsonl`:** a record can contain a single statement target or multiple `targets`; iterate with `parent.get("targets", [parent])`. Retain all targets and typed values, direction, qualifiers, rank, and available statement/revision identities. Inspect raw `wikidata_statement.qualifiers` when interpreting qualifiers.
- **Tables, `tables/tables.jsonl`:** retain page/revision and section context, layered `header`, all `rows`, cell lists/links, and processed spans. Do not expand spans a second time or merge equal-text rows.
- **Infoboxes, `infoboxes/infoboxes.jsonl`:** retain ordered `rows`, repeated fields, internal headings, image captions, list boundaries, and links. Reducing fields to a dictionary loses repeated occurrences and context.

Existing recovery metadata and entity bindings in raw sources are not Gold labels. See [corpus download instructions](data_repository/README.md) and [negative collection instructions](negative_samples/README.md).

## Complete-record provenance: `source_mapping.jsonl`

The mapping has one row for each of the 184,856 complete records, including negative records. It does not contain question membership, Gold selectors, answers, or proof chains.

| Field | Meaning |
| --- | --- |
| `source_record_id` | Complete source ID |
| `source_type` | `kb`, `table`, or `info` |
| `repository_file` | Corpus file path relative to `Data/` |
| `upstream` | Original record identities and revision provenance |

Wikipedia records retain available page title/ID, revision, revision URL, section, and table/infobox index. KG `upstream.targets` retains the available subject, property, statement GUID, and revision identity for every target, including targets not cited as Gold.

Use `repository_file` and `source_record_id` to read the complete released record. Upstream identities support tracing it to Wikipedia or Wikidata. They do not by themselves reproduce byte-identical JSON: the original extraction, span/header processing, list/link parsing, type conversion, and recovery rules also affect the result. Use the released JSONL records to reproduce benchmark experiments.

## Publication boundary

QA and Gold labels reflect final human review. Negative T1 contains model-proposed material left outside confirmed Gold, after excluding complete Gold parents. T2 and T3 undergo shared-repository model screening and global removal of selected supporting parents; this does not guarantee that retrieval-based screening discovered every possible support path.

Original dataset answer/entity hints, human review notes, model execution logs, candidate repositories, and supporting records removed during screening are excluded from public metadata and negative packages. See [LICENSE.md](LICENSE.md) for the data licensing status.
