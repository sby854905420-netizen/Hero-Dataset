# Hero Dataset data documentation

This directory contains the final revised QA, complete retrieval corpus, and evaluation annotations. JSONL files use UTF-8, with one JSON object per line, and do not depend on files outside this repository.

| Item | Count |
| --- | ---: |
| Questions | 500 |
| Gold answer items | 645 |
| Question-level evidence uses | 2,099 |
| Distinct fine-grained evidence references | 2,055 |
| Complete source records | 1,633 |
| Complete KG records | 747 |
| Complete tables | 528 |
| Complete infoboxes | 358 |
| Statement targets inside KG records | 1,058 |

The `domain` metadata contains 245 `movies` and 255 `tvseries` questions. There is one 500-question evaluation set and no additional train/dev/test split. Corpus and annotation text is preserved in its original language; documentation translation does not rewrite the data.

## Files and joins

```text
Data/
├── README.md
├── LICENSE.md
├── source_mapping.jsonl
├── qa/qa_pairs.jsonl
├── data_repository/
│   ├── kg/kb.jsonl
│   ├── tables/tables.jsonl
│   └── infoboxes/infoboxes.jsonl
└── annotations/
    ├── evaluation_annotations.jsonl
    └── evidence_sources.jsonl
```

Methods read `question_id` and `question` from QA and index only `data_repository/`. Gold answers, proofs, and question-specific evidence annotations are evaluation inputs, not retrieval documents or generation prompt inputs. A separate questions file is unnecessary.

- `question_id` joins QA, per-question evaluation annotations, and per-evidence annotations.
- `source_record_id` joins evidence, source mapping, and complete corpus records.
- `evidence_ref` / `ref` locates a statement target, table row, whole table, or infobox field within a source.
- `repository_file` and `repository_locator.file` are relative to `Data/`. Record IDs and paths are sufficient; old archive offsets and line numbers are not required.

## Questions and answers: `qa/qa_pairs.jsonl`

| Field | Meaning |
| --- | --- |
| `question_id` | Original question ID as a unique string; IDs need not be consecutive |
| `question` | Final question text |
| `answers` | Revised gold answer array; each item has `id` and `label` |
| `domain` | `movies` or `tvseries` |

```json
{"question_id":"9","question":"What is the Original Language of the film Encanto?","answers":[{"id":"Q1860","label":"English"}],"domain":"movies"}
```

An answer `id` is not necessarily a Wikidata QID: it may be a date, number, text, or empty string. Do not force all answers into entity IDs, or expand year precision into a full date. The zero-based position in `answers` is used by proof-branch `answer_indices`. Matching, aliases, precision, and tolerances belong to `evaluation/`.

## Complete retrieval corpus: `data_repository/`

Each line is one complete stored source, retaining content and context rather than only the evidence selected by a question. These parent records are the units of retrieval evaluation.

### KG: `kg/kb.jsonl`

- The source ID is `record_uid`, for example `kb-record:1256660`.
- Single-target records store fields such as `subject_qid`, `subject_label`, `pid`, `property_label`, `value`, and `statement_guid` directly.
- Some records wrap multiple targets in `targets`. Read them with `parent.get("targets", [parent])`, not only the first displayed top-level statement.
- `value` preserves its stored type, including entities, strings, times, and quantities. `wikidata_statement` preserves raw qualifiers, value types, rank, and available references. Inspect `wikidata_statement.qualifiers` when reading qualifiers.
- Existing `rev_id`, `rev_timestamp`, `kb_snapshot`, and recovery metadata are retained. They do not imply one revision time for every statement.

### Tables: `tables/tables.jsonl`

- The source ID is `evidence_id`, for example `table:64725222:1208159420:1`.
- `page_title`, `page_id`, `rev_id`, `section_path`, and `table_idx` identify the page revision and table.
- `header` preserves multiple header layers. `rows` contains all data rows. A selector's `row_index` is a position in `rows`, not an absolute HTML row number, and excludes headers stored separately in `header`.
- Cells may contain `text`, `links`, `text_items`, and `item_boundaries`. Spans have already been processed; do not expand `rowspan`/`colspan` again or merge rows with identical text.
- Existing `row_legacy_ids`, entity bindings, and correction metadata are retained. They are not gold support labels.

### Infoboxes: `infoboxes/infoboxes.jsonl`

- The source ID is `evidence_id`, for example `info:62975109:1280037512:0`.
- `page_title`, `page_id`, `rev_id`, `section_path`, and `box_idx` identify the page revision and infobox.
- `rows` includes normal fields, internal headings, and other context. Keeping only `row_type=normal` loses context and may change field-occurrence identities.
- Field rows retain `field`, `value`, `value_items`, list boundaries, links, and `row_idx`. Repeated names must not be overwritten in a simple dictionary.

Complete IDs may have suffixes such as `@occ`. Preserve the full ID throughout retrieval, prediction, citation, and evaluation.

## Per-question gold: `annotations/evaluation_annotations.jsonl`

Each line corresponds to the same question ID and answers in QA.

| Field | Meaning |
| --- | --- |
| `question_id`, `answers` | Question identity and answers, consistent with QA |
| `answer_items` | Stable `gold_item_id`, canonical identity, and label for each answer |
| `proof_branches` | Confirmed proof branches supporting answer items |
| `relevant_record_ids` | Confirmed relevant complete source IDs |
| `evidence` | Fine-grained evidence: `ref`, parent ID, selector, and relevance |

Interpret `proof_branches` as follows:

1. A branch's `answer_indices` identifies the answer items it supports.
2. All `steps` in a branch must hold.
3. Each step's `or_options` lists alternative support combinations. One complete alternative is sufficient; all references inside that alternative are jointly required.
4. Multiple branches may support one answer. Valid branches together must cover every gold answer item.
5. Auxiliary sources may occur in joint support. This does not make each source independently sufficient.

Retrieval hits use complete parents. Grounded EM uses fine-grained selectors and confirmed proof combinations. Retrieving a table does not make every row a valid supporting citation.

## Per-evidence annotations: `annotations/evidence_sources.jsonl`

A row is one evidence use by one question. Sources or fine-grained references used by different questions appear in multiple rows.

| Field | Meaning |
| --- | --- |
| `annotation_id` | Stable use identifier, `question_id::evidence_index` |
| `question_id`, `evidence_index` | Question ID and position in its evidence array |
| `original_evidence_index` | Historical annotation metadata, not a current array position |
| `evidence_ref` | Fine-grained reference matching gold `evidence.ref` |
| `source_type` | `kb`, `table`, or `info` |
| `source_record_id` | Complete parent ID |
| `selector` | Locator for the actual evidence within the parent |
| `repository_locator` | Corpus `file` and `record_uid` in this release |
| `upstream` | Actual selected evidence identity; the selected target for multi-target KG records |
| `support_scope` | Facts the evidence supports |
| `proof_usage` | Existing proof-use relationships and joint citations; final sufficiency is defined by `evaluation_annotations.jsonl` |
| `source_identity`, `same_origin_refs` | Existing origin identities; representations of one statement are not automatically independent origins |
| `retrieval_relevant` | Confirmed relevance for this question |

Twenty-three auxiliary evidence uses have no explicit `proof_usage` entry. Their content, relevance, and `support_scope` remain available. An empty array neither establishes independent sufficiency nor authorizes a new proof. Every gold answer already has confirmed complete proof branches.

All indexes and occurrence counts are zero-based:

| Evidence unit | Example selector | Reference suffix |
| --- | --- | --- |
| KG statement target | `{"unit":"statement_target","target_index":0,"statement_guid":"Q…$…"}` | `#target=0` |
| Table data row | `{"unit":"table_row","row_index":0,"whole_table":false}` | `#row=0` |
| Complete table | `{"unit":"whole_table","whole_table":true}` | `#table=all` |
| Infobox field | `{"unit":"infobox_field","field":"Language","field_occurrence":0,"row_index":15}` | `#field=Language&occurrence=0` |

Infobox `field_occurrence` counts equal names in original row order. Its optional `row_index` is the selected row's original `row_idx`. Field names in reference strings are URL-encoded; selector `field` retains the original text. Whole-table and row references are not interchangeable.

## Source mapping: `source_mapping.jsonl`

This file has one row per complete parent: 1,633 rows. It does not record question membership or historical release migrations.

| Field | Meaning |
| --- | --- |
| `source_record_id` | Complete source ID |
| `source_type` | `kb`, `table`, or `info` |
| `repository_file` | Corpus path relative to `Data/` |
| `upstream` | Existing complete upstream identity metadata |

Wikipedia table and infobox metadata retains the site, page title/ID, revision, fixed `oldid_url`, section path, and `table_idx` or `box_idx`, plus available anchors and page metadata. Extractor indexes are zero-based; they are not the numbers a reader sees in a browser.

For KG records, `upstream.targets` covers **every target** with `target_index`, `subject_qid`, `statement_guid`, `property_id`, `rev_id`, `revision_url`, and available metadata: 1,058 targets in total. Even targets not selected as question evidence retain their identities. Top-level summaries do not replace target-specific provenance.

To reconstruct a source, fix the recorded revision before locating its table, infobox, or statement under the original extraction semantics. Snapshot dates describe acquisition batches; precise identity comes from per-record and per-target revisions.

Revisions and table indexes alone cannot guarantee identical reconstructed JSON. HTML acquisition, enumeration/filtering, nesting, span expansion, header projection, text/list boundaries, link parsing, KG type conversion, and existing recovery corrections affect extraction. The mapping supports source tracing; no automatic refetching program is included. Use the released complete JSONL corpus to reproduce benchmark experiments.

## Release boundary

Complete source text is preserved. Source membership matches the parents referenced by the final annotations. Earlier candidate pools, review files, migration maps, separate questions files, data manifests, checksum lists, and local validation scripts are excluded.

See [LICENSE.md](LICENSE.md) for the data licensing status.
