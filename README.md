# Hero Dataset

Hero Dataset is a question-answering benchmark over heterogeneous structured evidence: knowledge graphs (KGs), tables, and infoboxes. Systems retrieve information, return answers, and cite the specific evidence that supports them.

This release contains the final revised dataset, the **Naive RAG** baseline in `baseline/NavReg/`, and shared quality and efficiency evaluators. Local Qwen inference and the OpenAI Responses API are implemented. Full-dataset baseline results and A100 validation are pending; no formal baseline scores are included.

## Dataset

| Item | Count |
| --- | ---: |
| Questions | 500 |
| Gold answer items | 645 |
| Question-level evidence uses | 2,099 |
| Complete KG records | 747 |
| Complete tables | 528 |
| Complete infoboxes | 358 |
| Complete source records | 1,633 |

The corpus preserves complete sources, including rows, fields, and statement targets not cited by individual questions. A KG record may contain multiple targets; these are not separate source records.

Corpus membership is derived from the final revised annotations. This is a fixed candidate corpus, so results should not be interpreted as retrieval performance over all of Wikipedia or Wikidata. There is one 500-question evaluation set and no additional training or development split.

Retrieval metrics use complete sources. Grounded EM checks statement targets, table rows, whole tables, or infobox fields against confirmed proofs. Original corpus and annotation text is preserved; documentation translation does not rewrite the data.

## Repository layout

| Path | Purpose |
| --- | --- |
| `Data/qa/qa_pairs.jsonl` | The single QA entry point: IDs, questions, and revised gold answers |
| `Data/data_repository/` | Complete KG, table, and infobox corpora |
| `Data/annotations/` | Final evaluation annotations and evidence selectors |
| `Data/source_mapping.jsonl` | Source identities, upstream revisions, and locations |
| `baseline/NavReg/` | Naive RAG implementation, prompts, configuration, and environment files |
| `evaluation/` | Quality scoring, answer equivalence, and per-run efficiency summaries |
| `configs/` | Execution profiles and individual experiment configurations |
| `scripts/` | Entry points for serialization, indexing, experiments, and evaluation |

API keys, model weights, generated indexes, run outputs, Chinese documentation backups, local verification material, and machine-specific deployment notes are excluded from publication. There is no separate questions file, data manifest, checksum list, or experiment-comparison tool.

## Naive RAG baseline

The baseline serializes all three source formats with fixed rules, chunks them with the selected embedding model's tokenizer, creates one unified index, retrieves evidence, and generates answers with citations. It has one retrieval pass and no reranker.

| Component | Configuration |
| --- | --- |
| Embedding option 1 | `BAAI/bge-large-en-v1.5`, 1024 dimensions, 512-token input |
| Embedding option 2 | `BAAI/bge-m3`, dense-only 1024 dimensions, 8192-token input |
| Index | GPU Faiss exact inner-product search on normalized vectors, equivalent to cosine similarity |
| Chunk K | 5, 10, 20, 30, 40, 50, 60, 70, 80, 100 |
| Qwen generation | Official `Qwen/Qwen3.8-27B-FP8`; a separate NVIDIA quantized checkpoint is configured for local debugging |
| API generation | `gpt-6-luna`, provider-default generation parameters |
| Prompt | Shared system/user templates and three fixed examples outside the evaluation set |

K counts text chunks. Their complete parent sources are restored and deduplicated in first-hit order without filling back up to K sources. A separate top-100 unique-source ranking supports retrieval evaluation; its extra candidates do not enter generation. Gold answers and proof annotations are used only during evaluation.

Body chunks do not overlap. For retrieval only, table cell and infobox value text is capped at its first 100 tokens; the original corpus and full generation sources are preserved. Citations have no count cap. The method does not silently reduce K or remove evidence when a prompt exceeds its budget.

Implementation dependencies and exact model revisions are declared under `baseline/NavReg/`. One command runs one experiment, not an automatic K sweep or five repeats. `--seed` controls question order and local Qwen sampling, but does not control API sampling. API keys are supplied through `OPENAI_API_KEY`, not stored in configurations or outputs.

## Evaluation

The six quality metric families are Sufficiency, Hit, MRR, Answer Set F1, Answer Set EM, and Grounded EM. Sufficiency and Hit each report K=5 and K=10, yielding eight quality columns. Formal quality scores use all 500 questions, including failures and missing predictions.

The three efficiency families are mean input/output/total tokens, mean LLM calls, and mean per-question elapsed time. Actual retry waiting is excluded from elapsed time; request queueing and execution remain included. Reasoning tokens belong to output and are counted once. Reasoning text is not persisted. Judge consumption is tracked separately from method efficiency.

## Documentation and interfaces

- [Data documentation](Data/README.md): file schemas, identifiers, selectors, proof interpretation, and source mapping.
- [Baseline documentation](baseline/NavReg/README.md): serialization, chunking, retrieval, prompts, model settings, generic commands, and outputs.
- [Evaluation documentation](evaluation/README.md): prediction format, metric definitions, answer adjudication, and efficiency accounting.

The reusable shell interfaces are `scripts/prepare_data.sh`, `scripts/build_index.sh`, `scripts/run_experiment.sh`, and `scripts/evaluate.sh`. Each exposes `--help`. Environment recipes are provided in `baseline/NavReg/environment.yml` and `environment.lock.yml`; they describe the tested local stack and do not establish compatibility with every machine.

## Licensing and citation

Final licenses for the code and newly contributed data have not yet been selected. Public hosting does not imply a license. See [LICENSE](LICENSE) and [Data/LICENSE.md](Data/LICENSE.md). Formal author and publication metadata is pending; no authors, paper, or DOI are invented in `CITATION.cff`.
