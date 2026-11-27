# NavReg: Naive RAG baseline

NavReg implements the Hero Dataset Naive RAG baseline: fixed serialization of KG, table, and infobox sources; native-tokenizer chunking; BGE-M3 dense embeddings; one unified dense index; one retrieval pass; and LLM answers with citations. There is no reranker.

The implementation includes serialization, GPU embedding/Faiss, full-source restoration, Qwen/Transformers and OpenAI Responses generation, failure recording, resume, and evaluation. It uses one NF4 Qwen checkpoint and BGE-M3 on every machine. Full 500-question results and large-K context feasibility remain unverified. Verification artifacts are not published.

Method rules are in `configs/method.yml` relative to this directory. Runtime paths and token budgets are in the repository's `configs/runtime.yml`; relative paths resolve from that configuration file's directory.

## Environment and serialization

Use Conda `Hero` with Python 3.11. `environment.yml` specifies PyTorch 2.14.0/CUDA 13.0, Transformers 5.20.2, bitsandbytes 0.50.2, Accelerate, NumPy 2.3.5, GPU Faiss, and GCC/G++ 14. The CUDA/PyTorch/Faiss builds must match the target GPU; token budgets must be measured on that GPU. The API adapter uses HTTPX directly and does not require the OpenAI SDK.

```bash
conda env create -f baseline/NavReg/environment.yml
conda activate Hero
python -m pip install -e baseline/NavReg
```

First download and extract the [v2.0.0 full corpus](../../Data/data_repository/README.md). A Git clone does not include the large corpus JSONL files. Run from the repository root:

```bash
bash scripts/prepare_data.sh --output-dir outputs/serialized/navreg-fixed-serialization-v2
```

`--workers` controls serialization threads (default 4); it does not authorize concurrent access to a shared GPU Faiss object. The output directory must not exist. The final corpus contains 184,856 sources. Only these corpus files are read:

- `Data/data_repository/kg/kb.jsonl`
- `Data/data_repository/tables/tables.jsonl`
- `Data/data_repository/infoboxes/infoboxes.jsonl`

Outputs are `records.jsonl` (one complete source with `context_text`, `units`, and original coordinates), `labels.json` (names available in the corpus), and `summary.json` (serialization counts/status). These are neither chunks nor embeddings.

## Fixed serialization

- **KG:** each statement target retains Subject, Relation, Value, Qualifiers, and Rank. Entity/property IDs, qualifier associations, date precision, quantity units, and statement GUIDs are preserved. Multi-target parents retain every target. Raw statement qualifiers determine their values; normalized copies supply names only.
- **Tables:** Markdown rows retain page, section, caption, layered headers, original row/column coordinates, list boundaries, and cell content.
- **Infoboxes:** retain page and section, then fields/values in original order, including repeated fields, internal headings, and image captions. Headings do not become invented citable fields. Caption rows use `infobox_row` selectors addressed by their position in the source `rows` array.

Complete source IDs, including `@occ` suffixes, are unchanged. Units carry original statement-target, table-row, or infobox-field selectors. Serialization units are not embedding chunks.

## Chunking and embedding

BGE-M3 runs on GPU with a fixed model/tokenizer revision and one index combining all three formats:

| Model | Revision | Dimensions / maximum input |
| --- | --- | --- |
| `BAAI/bge-m3` | `5617a9f61b028005a4858fdac845db406aefb181` | dense-only 1024 / 8192 tokens |

Official model modules provide CLS pooling. Computation and stored vectors use float32, with L2-normalized query/document vectors. Queries and passages have no prefix; sparse/ColBERT outputs are disabled. Batch size initially defaults to 1 and may be explicitly adjusted after hardware trials.

BGE-M3's native tokenizer counts the entire chunk: body, repeated context, headers, coordinates, and special tokens. No body overlap is used, and different parent sources are never packed together.

For retrieval only, each rendered table data-cell value and infobox field value keeps its first **100 tokens**, excluding special tokens. The tail is discarded, not emitted as another fragment. Field names, column names, entity titles, and internal headings are retained. KG statements are not subject to this cell-value cap. Full source serialization and `Data/` are unchanged; chunk coverage records truncation and original coordinates. Changing this rule requires rebuilding the index.

- **Table rows:** pack original rows with all columns when possible. If one row exceeds the budget, greedily group consecutive columns from left to right. If columns 1–5 fit but column 6 does not, column 6 starts the next block. Each block repeats its own column headers, page context, and source row index, retaining original column indexes. No additional identity column is selected or repeated.
- **Infobox fields:** pack field rows, repeating the entity title and active internal section heading. Preserve field names, original order, and repeated-name occurrence. Internal headings provide context without changing field identity.
- **KG:** keep each complete statement and its qualifiers together. Qualifiers use `; ` and statements use blank lines; these separators are ordinary text, not newly registered special tokens.

The 100-token value cap does not replace the whole-chunk check. If essential context, a cell/header, or an indivisible KG statement still exceeds the limit, fail with source coordinates. Do not silently truncate context or expand the model limit. Generation restores full sources rather than presenting retrieval-truncated values as complete evidence.

## Index and retrieval

Faiss `IndexFlatIP` provides exact search. The inner product of normalized vectors is cosine similarity. The bundle is saved in CPU format and loaded into RAM/GPU for each run. The runtime partitions GPU storage into segments of at most 1024 vectors to avoid GPU top-K limits, scores every segment, and merges their results. These remain one logical index, not separate indexes by source type. All search/merge work counts toward retrieval time. GPU scratch memory is limited to 64 MiB.

Experiment **K means text chunks**, with allowed values:

```text
5, 10, 20, 30, 40, 50, 60, 70, 80, 100
```

Chunks rank by descending cosine score. The current implementation leaves equal-score ordering to the retrieval result order; a portable secondary ordering is not yet fixed. Identical source text can produce exact ties, so do not claim portable tie ordering.

The top K chunks select their complete parent KG records, tables, or infoboxes. Restore parents, deduplicate in first-hit order, and pass them to the prompt. If fewer than K parents remain, do not retrieve more to fill the count.

Retrieval evaluation independently retains up to 100 unique parents. A parent's score is its highest chunk score; exact all-chunk ranking is aggregated in first-hit order. This ranking supports Hit@5/10 and MRR@100. Extra evaluation candidates are not passed to generation.

Qwen's actual tokenizer checks the complete prompt plus output reservation. A context overflow fails with the observed length and configuration. The code does not reduce K, remove sources, or crop full evidence automatically. API context limits are enforced by the provider rather than a Qwen-tokenizer estimate. Large-K context feasibility still requires trials.

## Generation prompt and examples

Both backends use `prompts/system.txt`, `prompts/answer.txt`, and `prompts/examples.json`. `build_messages()` builds eight messages: one system message, three user/assistant demonstrations, and the current user message. Substitution occurs once, so placeholder-like source text remains data.

The model returns only:

```json
{"answers":["answer item"],"citations":["E1"]}
```

Context construction assigns labels only to visible citable units. The runner maps labels to official source IDs/selectors and adds question ID, status, and retrieval ranking to the five-field prediction. The model does not generate telemetry.

Prompt rules allow evidence-based joining, filtering, comparison, ordering, and arithmetic. Exact counts, ranks, earliest/latest conclusions, and exhaustive sets require a direct statement or evidence covering the relevant scope. Missing retrieval is not proof of No, zero, or none. A decisive example or counterexample can support a Boolean answer. Independent supported items may be returned for a multi-answer question without claiming completeness. Preserve identities, roles, time qualifiers, precision, and units; do not guess through unresolved conflicts.

Nonempty answers require distinct current evidence labels that jointly support all answers and necessary intermediate facts. There is **no citation-count cap**. With no supported answer, both arrays are empty. Source instructions are treated as data. Whole-table labels require actual full-table visibility; whole-table and row citations do not automatically substitute for each other during evaluation.

Three fixed demonstrations come from human-reviewed reserve questions outside the final 500:

| Original ID | Source / operation | Question | Answer |
| --- | --- | --- | --- |
| 80 | KG / day-precision date | What was Kelly Overton's date of birth? | 28 August 1978 |
| 759 | Infobox / multiple items | What countries is Mel Gibson a citizen of? | United States; Ireland |
| 7461 | Table / layered-header Boolean | Has the movie The Godfather won any Academy Award? | Yes |

The first two were approved; the third was corrected and then approved. Their question IDs are outside the final 500. Evidence was recovered from the historical source material and serialized with the same rules, retaining date precision, the complete citizenship list, and the relevant Academy Awards row/column context. Demonstrations are fixed prompt assets; the indexer does not add their snippets as extra corpus records.

Example `provenance` records review status, original IDs, selectors, and upstream revisions. It is not sent to the model, and original review files are not needed at runtime. Demonstrations use `DEMO1_E1`, `DEMO2_E1`, and `DEMO3_E1`; current questions may cite only their own evidence labels. The example assets themselves are not indexed or appended to the retrieval corpus. All demonstration messages count as input tokens in one request, not three extra calls. Prompt assets are saved and compared directly on resume, without file digests.

## Generation model and runtime budgets

| Backend | Checkpoint / interface |
| --- | --- |
| Qwen | `unsloth/Qwen3.8-27B-unsloth-bnb-4bit`, Transformers 5.20.2 |
| API | `gpt-6-luna`, OpenAI Responses |

The `huggingface` backend uses the same Unsloth checkpoint and official chat template on every machine, pinned to revision `8aa5f05d26b7205477066e1449e0af13f762a299`. Prequantized NF4 weights use bitsandbytes 0.50.2 with BF16 arithmetic, unquantized components, and KV cache. The loader verifies the checkpoint's NF4 configuration and the actual 4-bit modules. No CPU offload or speculative decoding is enabled. Effective model identity, quantization, engine, and sampling parameters are recorded.

The conservative startup settings are:

| Total window | Complete input budget | Non-thinking output | Thinking output |
| ---: | ---: | ---: | ---: |
| 16384 | 8192 | 8192 | 8192 |

All budgets are tokens. The complete input includes instructions, examples, question, and evidence; thinking and final JSON share the output budget. Both modes share the input budget, and unused output reservation does not enlarge it automatically. These are startup budgets, not a verified maximum context length.

Set `generation.context_window_tokens`, `generation.input_token_budget`, and `generation.max_new_tokens_by_mode` in `configs/runtime.yml` after measuring GPU memory. Pass another file with `--config path/to/runtime.yml` when needed. Keep input plus each mode's output budget within the total window and the checkpoint's native 262144-token limit. Changing machines does not select another model or method. A context overflow reports an error without reducing K or cropping evidence.

### Qwen sampling modes

The default is `non_thinking`. `--mode thinking` explicitly selects the other profile. Both use the same checkpoint.

| Setting | Non-thinking | Thinking |
| --- | ---: | ---: |
| `enable_thinking` | false | true |
| `preserve_thinking` | true | true |
| `reasoning_effort` | not supplied | xhigh |
| `temperature` | 0.7 | 1.0 |
| `top_p` | 0.80 | 0.95 |
| `top_k` | 20 | 20 |
| `min_p` | 0.0 | 0.0 |
| `presence_penalty` | 1.5 | 0.0 |
| `repetition_penalty` | 1.0 | 1.0 |

These are the profiles recorded from the [fixed official model documentation](https://huggingface.co/Qwen/Qwen3.8-27B/blob/37561816e66b5b9153b1b64c10968c18c1e6f4a0/README.md). Transformers is pinned to 5.20.2. A logits processor applies the configured presence penalty; unsupported settings must not be silently ignored or replaced with different penalties.

Qwen reasoning is generated output. The adapter counts actual generated token IDs up to the native closing think boundary, decodes only the final portion, and discards reasoning text and IDs. A think marker prefilled by the chat template belongs to input. If the boundary is missing, preserve observable usage and fail without saving unseparated thought text. Output exhaustion or invalid final JSON is a failure.

### API defaults

GPT generation and answer-equivalence judging use `gpt-6-luna`. Requests do not override provider sampling defaults or send an unsupported API seed. No separate immutable dated snapshot is assumed: requested/returned model identities and observable settings are logged; unavailable values remain `null`.

Both API paths explicitly send `tools=[]`, `tool_choice="none"`, and `store=false`, with no external conversation or previous response. No web/file search, MCP, or custom tool is enabled. The fixed prompt constrains evidence use; disabling tools does not erase model pretraining knowledge. Only final content, safe metadata, and usage are saved, not full responses or reasoning summaries. See the [Responses API reference](https://developers.openai.com/api/reference/python/resources/responses/methods/create).

Keys are read from the environment variable named by `api_key_env` (default `OPENAI_API_KEY`). An ignored local file may supply that environment, but is not loaded automatically or copied into outputs.

## Experiments and seeds

Each command runs one complete 500-question experiment unless `--debug --limit N` explicitly requests a subset. There is no automatic five-repeat or K loop. Separate tasks/runs handle different configurations; there is no cross-run comparison tool.

Seed precedence is CLI `--seed`, experiment `execution.run_seed`, then 0. Sort all question IDs as strings, shuffle with an independent `random.Random(run_seed)`, and save the exact order. Qwen additionally derives each request seed from run seed, question ID, and attempt number using a fixed integer formula. This stream is independent of question shuffling. API seeds affect question order only, not provider sampling. Different seeds do not guarantee different answers or bitwise reproducibility across hardware.

The experiment unit is one complete dataset run. Each run owns its GPU index instance and result directory. The published workflow runs on one machine; runtime configuration changes paths and budgets without changing the method.

## Requests, failures, and recovery

Each generation attempt has a 60-second total deadline, including connection, provider queueing, prefill, and generation. Model loading and retrieval are outside that request deadline. Retryable transport errors, rate limits, temporary server failures, and timeouts allow up to three additional attempts (four total), with a fixed 5-second wait. Actual wait is measured and excluded from question latency; request time remains included. Transport automatic retries are disabled.

Invalid JSON, output truncation, or invisible/invalid citations do not trigger answer repair or regeneration. A failed prediction has a legal ranking and empty answer/citation arrays. Structurally valid citations are checked against the published Gold set as a diagnostic; the runner never reads gold to decide whether to retry. The current Gold lists do not define complete proof combinations, so Grounded EM and Sufficiency are not computed.

The journal records requests before sending and final content/usage after return. Each completed question is committed durably before the recoverable prediction/trace views are appended. `--resume` requires identical effective configuration, prompt assets, index metadata, seed, and question order. Runs from the previous model or configuration format require a new run directory; do not resume them under the unified configuration. All committed terminal questions, including failures, are skipped. Each run directory has an exclusive writer lock.

A crash after remote execution but before persistence cannot guarantee exactly-once billing or execution. Preserve interrupted attempts and unknown usage. Unobserved elapsed time across an interruption makes `active_seconds=null`, with known subtotals retained; downtime is not treated as query time or guessed away.

## Commands and output files

From the repository root:

```bash
bash scripts/build_index.sh
bash scripts/run_experiment.sh \
  --experiment configs/experiments/navreg_m3_k5.yml \
  --run-dir outputs/runs/first --debug --limit 3
```

Omit `--debug --limit 3` to run the complete 500 questions. `--config` selects runtime paths and budgets; `--backend openai_api` selects GPT; `--mode thinking` selects Qwen thinking. `--index-dir` selects an index bundle and `--serialized-dir` its complete source records. `--resume` continues an existing run under the same effective configuration.

An index bundle contains `index.faiss`, `chunk_map.jsonl`, and `manifest.json` with settings/counts, not file digests. Batch size may change without changing document vectors; model, chunking, representation, or precision changes require a matching rebuilt bundle.

Each run contains:

- `run_manifest.json`: effective configuration, index metadata, prompt assets, and question order.
- `runtime.json`: actual generator identity, embedding settings, hardware, and startup time.
- `logs/attempts.jsonl`: durable request/question journal.
- `predictions.jsonl`: the evaluator's five fields.
- `trace.jsonl`: visible evidence, selectors, attempts, actual token usage, and timing.
- `efficiency.json`: the three agreed efficiency averages.
- `metrics/`: quality outputs for a full run; unresolved matches may require the fixed judge.

The runner reads final questions from `Data/qa_pairs.jsonl`; answers and `Data/gold_annotations.jsonl` are used only by evaluation. Full runs automatically score. With unresolved pairs, enabled adjudication, and an API key, the fixed judge is called; `--skip-judge` defers that. Debug subsets summarize efficiency without claiming full-dataset quality. For definitions and standalone commands, see [evaluation](../../evaluation/README.md).

Only mean input/output/total tokens, mean LLM calls, and mean question time are efficiency metrics. Failures and retries count. Reasoning is part of output, cached input is part of input, and neither is added twice. Judge usage is separate. No P50/P95 or comparison tables are produced.

## Release scope

This release includes final data and reusable code. Model weights, indexes, run outputs, functional checks, trial reports, and Chinese/local archives are excluded by `.gitignore`. Full-dataset results and maximum-context validation remain pending. The environment file does not establish compatibility with every GPU.
