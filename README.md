# Hero Dataset

Hero Dataset is a question-answering benchmark over knowledge graphs (KGs), tables, and infoboxes. Systems retrieve complete sources, return answers, and cite the specific evidence that supports them.

Version 2.0.0 contains the latest 500 human-confirmed questions and answers, Golden Evidence, and three negative collections. It includes the **Naive RAG** baseline in `baseline/NavReg/` and shared quality and efficiency evaluators. Full-dataset baseline results and A100 validation remain pending; no formal baseline scores are included.

## Dataset

| Item | Count |
| --- | ---: |
| Questions | 500 |
| Gold answer items | 665 |
| Question-level Gold evidence uses | 2,291 |
| Unique fine-grained Gold evidence references | 2,222 |
| Complete KG records | 145,115 |
| Complete tables | 17,821 |
| Complete infoboxes | 21,920 |
| Complete source records | 184,856 |

The questions comprise 245 `movies` and 255 `tvseries` items. There is one evaluation set and no additional training or development split. Complete sources preserve rows, fields, and statement targets beyond the selected Gold evidence. A source containing a Gold unit is not entirely labeled Gold.

| Collection | Complete records |
| --- | ---: |
| Golden Evidence parents | 1,702 |
| Negative T1 | 209 |
| Negative T2 | 20,582 |
| Negative T3 | 162,363 |
| **Full repository** | **184,856** |

The four collections are combined in the default retrieval corpus. The three negative collections are also available separately for composing difficulty variants. This is a fixed benchmark corpus; performance does not represent retrieval over all of Wikipedia or Wikidata.

## Download the corpus

The large corpus files are hosted as [v2.0.0 Release assets](https://github.com/sby854905420-netizen/Hero-Dataset/releases/tag/v2.0.0), separately from Git history. A Git clone or GitHub's automatically generated source archive does not include these assets.

From the repository root, download and extract the complete corpus:

```bash
curl --fail --location --output data_repository.tar.gz \
  https://github.com/sby854905420-netizen/Hero-Dataset/releases/download/v2.0.0/data_repository.tar.gz
tar -xzf data_repository.tar.gz
```

This supplies `Data/data_repository/` and `Data/source_mapping.jsonl`. The complete corpus already contains all three negative collections; their separate archives are needed only to select or combine subsets.

| Release asset | Contents after extraction from the repository root |
| --- | --- |
| [`data_repository.tar.gz`](https://github.com/sby854905420-netizen/Hero-Dataset/releases/download/v2.0.0/data_repository.tar.gz) | `Data/data_repository/` and `Data/source_mapping.jsonl` |
| [`negative_t1.tar.gz`](https://github.com/sby854905420-netizen/Hero-Dataset/releases/download/v2.0.0/negative_t1.tar.gz) | `Data/negative_samples/negative_t1/` |
| [`negative_t2.tar.gz`](https://github.com/sby854905420-netizen/Hero-Dataset/releases/download/v2.0.0/negative_t2.tar.gz) | `Data/negative_samples/negative_t2/` |
| [`negative_t3.tar.gz`](https://github.com/sby854905420-netizen/Hero-Dataset/releases/download/v2.0.0/negative_t3.tar.gz) | `Data/negative_samples/negative_t3/` |

## Repository layout

| Path | Purpose |
| --- | --- |
| `Data/qa_pairs.jsonl` | Question IDs, final questions, confirmed answers, and domains |
| `Data/gold_annotations.jsonl` | Per-question Gold references and their precise positions within complete records |
| `Data/source_mapping.jsonl` | Every complete record's corpus location and upstream provenance |
| `Data/data_repository/` | Complete KG, table, and infobox corpora, supplied by the full Release asset |
| `Data/negative_samples/` | Separate negative collections and download/composition instructions |
| `baseline/NavReg/` | Naive RAG implementation, prompts, configuration, and environment files |
| `evaluation/` | Quality scoring, answer equivalence, and per-run efficiency summaries |
| `configs/` | Execution profiles and individual experiment configurations |
| `scripts/` | Entry points for serialization, indexing, experiments, and evaluation |

The three metadata files have separate roles: QA stores what is asked and answered; Gold annotations store evidence membership and precise positions; source mapping stores complete-record locations and provenance. Shared IDs join these files. Answers are not duplicated in Gold annotations, and human proof chains are not published.

Original source answer/entity hints, annotation logs, review notes, candidate pools, and supporting material removed during negative construction are excluded. API keys, model weights, generated indexes, and run outputs are also excluded.

## Naive RAG baseline

The baseline serializes all three source formats, chunks them with the selected embedding model's tokenizer, creates one unified index, retrieves evidence, and generates answers with citations. It has one retrieval pass and no reranker.

| Component | Configuration |
| --- | --- |
| Embedding option 1 | `BAAI/bge-large-en-v1.5`, 1024 dimensions, 512-token input |
| Embedding option 2 | `BAAI/bge-m3`, dense-only 1024 dimensions, 8192-token input |
| Index | GPU Faiss exact inner-product search on normalized vectors, equivalent to cosine similarity |
| Chunk K | 5, 10, 20, 30, 40, 50, 60, 70, 80, 100 |
| Qwen generation | Official `Qwen/Qwen3.8-27B-FP8`; a separate NVIDIA quantized checkpoint is configured for local debugging |
| API generation | `gpt-6-luna`, provider-default generation parameters |
| Prompt | Shared system/user templates and three fixed reserve-question examples |

K counts text chunks. Their complete parent sources are restored and deduplicated in first-hit order without filling back up to K sources. A separate top-100 unique-source ranking supports retrieval evaluation; its extra candidates do not enter generation. Gold answers and question-specific Gold evidence are evaluation inputs only.

Body chunks do not overlap. For retrieval only, table cell and infobox value text is capped at its first 100 tokens; the original corpus and full generation sources are preserved. Citations have no count cap. The method does not silently reduce K or remove evidence when a prompt exceeds its budget.

Implementation dependencies and exact model revisions are declared under `baseline/NavReg/`. One command runs one experiment, not an automatic K sweep or five repeats. `--seed` controls question order and local Qwen sampling, but does not control API sampling. API keys are supplied through `OPENAI_API_KEY`.

## Evaluation

The published Gold annotations are evidence lists without proof-chain or alternative-support relationships. Accordingly, this release reports **Hit@5, Hit@10, MRR@100, Answer Set F1, and Answer Set EM**. Proof-dependent Sufficiency and Grounded EM are not computed. Formal quality scores use all 500 questions, including failures and missing predictions.

The efficiency metrics are mean input/output/total tokens, mean LLM calls, and mean per-question elapsed time. Actual retry waiting is excluded from elapsed time; request queueing and execution remain included. Reasoning tokens belong to output and are counted once. Reasoning text is not persisted. Judge consumption is tracked separately from method efficiency.

## Documentation and interfaces

- [Data documentation](Data/README.md): metadata schemas, identifiers, evidence selectors, and provenance.
- [Corpus documentation](Data/data_repository/README.md): full-corpus contents and download instructions.
- [Negative collections](Data/negative_samples/README.md): separate downloads, construction, and composition.
- [Baseline documentation](baseline/NavReg/README.md): serialization, chunking, retrieval, prompts, models, commands, and outputs.
- [Evaluation documentation](evaluation/README.md): prediction format, metrics, answer adjudication, and efficiency accounting.

The shell interfaces are `scripts/prepare_data.sh`, `scripts/build_index.sh`, `scripts/run_experiment.sh`, and `scripts/evaluate.sh`; each exposes `--help`. Environment recipes in `baseline/NavReg/environment.yml` and `environment.lock.yml` describe the tested local stack rather than compatibility with every machine. Download and extract the corpus before running these commands.

## Licensing and citation

Final licenses for the code and newly contributed data have not yet been selected. Public hosting does not imply a license. See [LICENSE](LICENSE) and [Data/LICENSE.md](Data/LICENSE.md). Formal author and publication metadata is pending; no authors, paper, or DOI are invented in `CITATION.cff`.
