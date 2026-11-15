# Hero Dataset evaluation

The evaluator scores six quality metric families against the final `Data/` release: 500 questions, 645 gold answer items, and 1,633 complete sources (747 KG records, 528 tables, and 358 infoboxes). It reads QA, complete corpus records, and evidence annotations directly, without old archives or byte-offset indexes.

## Commands

Run from the repository root:

```bash
conda run --no-capture-output -n Hero python evaluation/score.py \
  --predictions /path/to/predictions.jsonl \
  --output-dir outputs/runs/my-evaluation/metrics
```

`score.py` is an offline standard-library scorer and never calls model services. Its output directory must be absent or empty. `--data-dir`, `--answer-policies`, `--judge-config`, and `--judgments` override the corresponding assets. The default data directory is this repository's `Data/`.

The shell wrapper also supports evaluating a run directory:

```bash
bash scripts/evaluate.sh --run-dir /path/to/run
bash scripts/evaluate.sh --run-dir /path/to/run --adjudicate
```

The second command explicitly permits the fixed API judge for unresolved answer pairs. Without that flag, standalone evaluation does not call the judge. Full baseline runs automatically attempt adjudication when enabled and a key is available; `--skip-judge` defers it. Debug subsets report their own efficiency, not a fabricated 500-question quality score.

## Prediction format

JSONL contains one question per line and only the following five fields. `question_id` must be a string. Omitted `status` defaults to `completed`; omitted arrays default to empty.

```json
{"question_id":"9","status":"completed","retrieved_record_ids":["kb-record:1256660"],"answers":[{"canonical_id":"Q1860","label":"English"}],"citations":[{"source_record_id":"kb-record:1256660","selector":{"unit":"statement_target","target_index":0}}]}
```

- `retrieved_record_ids`: up to 100 unique complete source IDs, in ranked order. Use KG `record_uid` and table/infobox `evidence_id`, retaining suffixes such as `@occ`. Invalid IDs or duplicate ranks reject the submission; ranks are not repaired or shifted.
- `answers`: independent answer items, as strings or objects using `canonical_id`, `id`, `label`, and/or `text`. An entity `canonical_id` must be a Wikidata QID. Final response text and telemetry belong in separate logs, without reasoning text.
- `citations`: fine-grained references, with no count limit. Repeated references represent the same unit and do not add support. Each citation must have a valid source and selector. Citations need not occur in the retrieval ranking; retrieval and citation quality are scored independently.
- `status`: `completed`, `failed`, `timeout`, or `refused`. Failed, timed-out, refused, and missing questions receive zero on every quality metric. The full denominator remains 500.

Failures must still be structurally valid:

```json
{"question_id":"9","status":"failed","retrieved_record_ids":[],"answers":[],"citations":[]}
```

A legal ranking may be retained, but invalid answer/citation content must not be written to the prediction. The runner records empty answers/citations on failure. Safe final content may remain in trace; status does not bypass submission validation.

## Citation selectors

Positions are zero-based. Structure is validated against the public corpus without reading question-specific gold:

| Source | Example selector | Unit |
| --- | --- | --- |
| KG | `{"unit":"statement_target","target_index":0}` | One statement target; optional `statement_guid` must agree |
| Table | `{"unit":"table_row","row_index":2}` | One data row |
| Table | `{"unit":"whole_table"}` | The complete table |
| Infobox | `{"unit":"infobox_field","field":"Language","field_occurrence":0}` | One occurrence of a field; optional `row_index` must agree |

Every corpus table admits a structurally valid whole-table citation; the method does not need gold to create it. However, a method may return it only when the full table was actually visible. The prediction file cannot prove visibility, so methods must retain visible context in their traces.

Legal references not registered for that question appear in `unconfirmed_citation_keys`. They do not support the current proof and incur no separate extra-citation penalty. Row, whole-table, and infobox-field references do not automatically substitute for each other. Newly discovered valid proofs require consistent gold/evaluator version review.

## Six quality metric families

A proof branch supports its listed answer items when every step has at least one complete alternative support combination. All references inside a combination are jointly required. Satisfied branches together must cover every gold answer item.

| Family | Output columns | Per-question definition |
| --- | --- | --- |
| Sufficiency | `Sufficiency@5`, `Sufficiency@10` | 1 if the top K complete sources contain confirmed evidence sufficient to prove every answer; otherwise 0 |
| Hit | `Hit@5`, `Hit@10` | 1 if the top K sources contain at least one confirmed relevant source; otherwise 0 |
| MRR | `MRR@100` | Reciprocal rank of the first relevant source within 100; 0 if absent |
| Answer Set F1 | `Answer Set F1` | After normalization and confirmed deduplication, maximum one-to-one matching gives `2 * matches / (prediction_count + gold_count)` |
| Answer Set EM | `Answer Set EM` | 1 if predicted and gold answer sets match completely; otherwise 0 |
| Grounded EM | `Grounded EM` | 1 if Answer Set EM is 1 and cited fine-grained evidence proves every answer; otherwise 0 |

All eight columns are macro-averaged over 500 questions and multiplied by 100. Retrieval uses complete sources; Grounded EM uses actual statements, rows, whole tables, or fields. Retrieving the right table is not the same as citing the row that supports the answer.

## Answer matching and pending adjudication

`answer_policies.json` records answer types, scope, confirmed aliases, date precision, units, and numeric tolerances. `answer_matcher.py` does not invent aliases or make API requests. An unfamiliar expression is not automatically wrong and does not expand gold.

The configured judge is `gpt-6-luna`, with provider-default generation parameters and the English `judge_prompt.txt`. The judge configuration/version and literal prompt are bound to cached decisions. The request identifier is an alias, not an assumed immutable dated checkpoint. Requested/returned identity and observable settings are recorded; unavailable metadata remains `null`.

A pending pair means equivalence is not yet confirmed, not that answers or evidence are missing. For example, an unregistered alternative name for the same country may need adjudication. While any answer pair affecting matching remains pending, aggregate Answer Set F1, Answer Set EM, and Grounded EM are `null`; retrieval metrics remain computable. A failed or `uncertain` judge result stays pending rather than becoming a baseline error.

To resolve pending pairs:

1. Use the frozen judge configuration. Changing its model, parameters, or prompt requires a new version/context.
2. Score with that configuration to obtain its bound pending queue.
3. Use `scripts/evaluate.sh --adjudicate` for the fixed judge, or an official review. Offline `score.py` never sends requests.
4. For imported review, fill `decision` (`equivalent`, `different`, or `uncertain`) and a short `reason` in `judgments.template.json`, preserving `context` and `cache_key`.
5. Rescore with the same configuration and `--judgments /path/to/judgments.json`. Mismatched context is rejected. Remaining unresolved pairs keep the three answer aggregates unknown.

Confirmed decisions for the same version, question, and normalized answer pair are reused, including across repeated runs. The shared `judge_cache.json` uses a file lock; it does not affect baseline retrieval time. Keys are direct canonical JSON, without hashes. The judge compares the supplied pair only; it does not answer the question anew, add gold answers, or use external knowledge. Judge usage/time is recorded separately from method efficiency.

Generation and judge requests explicitly disable tools (`tools=[]`, `tool_choice="none"`) and response storage (`store=false`), with no external conversation or previous response. Only prescribed messages are sent. A short final judgment explanation is not an intermediate reasoning trace. The English judge prompt has its own version; old-language cache contexts cannot be silently reused.

## Quality outputs

- `summary.json`: eight quality columns, missing/failed question counts, and pending-pair counts.
- `per_question.jsonl`: all 500 question results, matches, citation sufficiency, and unconfirmed references.
- `run_manifest.json`: binding between evaluation input, data/evaluator versions, matching policy, and judge configuration; this is a runtime artifact, not a published data manifest.
- `pending_answer_pairs.jsonl`: unresolved pairs, empty when none remain.
- `judgments.template.json`: generated when needed; imported/applied decisions are retained in `judgments.used.json`.
- `validation_errors.json`: locations/reasons for rejected submissions; no official score is produced.

Exit codes are 0 for completed scoring, 2 for invalid input/data/configuration, and 3 for pending adjudication.

## Three efficiency metric families

Only the following are efficiency metrics. Startup, index construction, whole-run wall time, failure statuses, and other raw metadata do not create extra metric families. No P50/P95 or cross-experiment comparisons are produced.

| Family | Per-question accounting and aggregate |
| --- | --- |
| LLM Tokens per Question | Sum actual input/output tokens across all attempts; report mean input, mean output, and mean total |
| LLM Calls per Question | Count independent submitted generation attempts; report mean calls |
| End-to-end Latency | Actual elapsed time from question processing to result readiness/termination, excluding active retry waits; report mean seconds |

Input includes the system prompt, native chat wrapping, all three example user/assistant pairs, the current question, actual evidence, and repeated input on retries. Examples are input messages in one request, not three extra calls. Output includes generated reasoning and final content, not just the visible answer JSON. Use provider usage or actual generated token counts, not model self-reports or retokenization of final text.

Cached input is a subset of input; reasoning is a subset of total output when already included. Neither is added twice. If reported separately, normalize output to include reasoning exactly once. Save reasoning counts only, not text, summaries, raw streams, full responses, or token IDs that reconstruct it. Unobservable subcounts are `null`, not invented zeros; a known output total can still be used when its reasoning subcount is unknown.

Calls count independent generations rather than HTTP requests or stream events. Five candidates for one prompt count as five generations; input usage reported once is counted once. Batch prompts must be attributed to their actual questions, not divided arbitrarily. Submitted failures, timeouts, and retries count. Pure embedding computation is not an LLM generation call.

The current vector retrieval stage has zero LLM calls/tokens, but query embedding and vector search still take time. End-to-end QA includes retrieval through answer/citation readiness. Offline indexing, unrelated startup/loading, and warmup are separate. Judge consumption is excluded.

## Trace protocol: `navreg-trace-v1`

`predictions.jsonl` contains only the five scoring fields. Each question's separate `trace.jsonl` row records status, ranking, visible evidence, original selectors, label mappings, error type, timing, and attempts.

| Field group | Meaning |
| --- | --- |
| `question_id`, `status`, `trace_version` | Question identity and trace state |
| `retrieval`, `visible_evidence`, `citation_labels` | Full evaluation ranking, selected chunks/parents, actual text, and official selectors |
| `timing.question_start_perf_s` | Monotonic start before query embedding/retrieval or method inference |
| `timing.retrieval_end_perf_s` | Ranking-ready time |
| `timing.question_end_perf_s` | Final answer/citation parsing, validation, and serialization ready, or actual failure time |
| `timing.retry_wait_seconds` | Actual accumulated active retry waits, measured rather than estimated from retry count |
| `attempts_complete`, `llm_attempts` | Whether submitted attempts are fully accounted for, and their per-attempt records |

Each attempt records request/attempt/retry identity; requested/returned model, revision, tokenizer and engine; provider/backend; run/question seed and whether applied; task/purpose; candidate count/status; request start/end; actual input/output/total tokens; cached/reasoning subcounts; usage source/completeness and numeric raw usage; final content; and whitelisted safe response metadata. Missing API revisions are `null`. API keys and Authorization headers never enter these records.

The API maps `usage.input_tokens`, `usage.output_tokens`, and their detail subcounts directly. `reasoning_tokens` comes from `output_tokens_details` and is already included in output. Local Qwen counts actual generated IDs, separating native thinking boundaries in memory before discarding thought text/IDs. Prefilled thinking markers belong to input; generated thinking/final content share the output budget. Missing boundaries or exhausted output without a complete final answer fail safely.

Unknown main usage stays `null`, not zero. With no generation attempts and `attempts_complete=true`, zero calls/tokens can be confirmed. Failed and timed-out resource consumption remains included. HTTP transport automatic retries are disabled so attempts stay visible.

For uninterrupted questions:

```text
retrieval_latency = retrieval_end_perf_s - question_start_perf_s
qa_latency = question_end_perf_s - question_start_perf_s - retry_wait_seconds
```

Only active waits between retry attempts are excluded. Network time, provider queueing, generation, and failed request execution remain included. GPU/asynchronous work must actually complete before endpoints are measured. Overlapping work must not be summed as elapsed time, and a batch wall time divided by count is not a substitute for per-question timing.

Formal efficiency uses all 500 questions, including failures/timeouts. Missing traces, attempts, usage, or unobservable interrupted time make the affected average `null`; known subtotals and coverage remain available. Never fill unknown consumption with zero or average only successful questions. Server precision, context, concurrency, hardware, cache settings, and actual sampling are recorded with the run.

## Retries, seeds, and recovery

A command runs one full experiment by default, using Qwen non-thinking generation. Repeats require separate commands and new directories, not an automatic five-run loop. Repeating the same 500 questions does not create more independent questions.

CLI `--seed` overrides experiment configuration, then defaults to 0. IDs are string-sorted before shuffling; exact order is persisted. Resume skips committed items without reshuffling the remainder. Qwen derives per-question/per-attempt sampling seeds from run seed, question ID, and attempt number. API seeds control local order only. Sampling/hardware changes can affect reproducibility; different seeds do not guarantee different answers.

Every request has a 60-second total deadline. Retryable transport errors, limits, temporary server errors, and timeouts permit three additional attempts, each preceded by a fixed 5-second wait. Requests are not extended merely because stream tokens arrive. The same policy applies to the judge, with separate consumption/time records. Local timeouts abort engine requests; remote interruption cannot guarantee that provider execution or billing stopped, so missing usage stays unknown.

Illegal JSON, truncation, or invalid/invisible citations fail without repair calls or gold-guided retries. Valid but insufficient citations are scored by Grounded EM. Failures produce legal prediction records with empty answers/citations.

A durable journal records an attempt before submission and its safe final content/usage after return, then commits each question. Prediction/trace files can be rebuilt from commits. Terminal committed questions, including failures, are skipped on identical-config resume. An interrupted unconfirmed remote attempt remains recorded, but cannot offer a strict exactly-once guarantee. Unobserved time across a hard interruption makes `timing.active_seconds=null`, preserving `known_active_seconds` without counting downtime or guessing missing intervals.

The runner and `summarize.py` implement these per-run efficiency rules. Local functional checks and synthetic results are excluded from the release and are not baseline scores.
