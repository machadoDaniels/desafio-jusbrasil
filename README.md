# Jusbrasil BRACIS 2026 — pipeline v2

V2 extracts legal citations from TXT documents, extracts their structured fields with parallel NER calls, and verifies them against a canonical SQLite database. It produces a final classification (`real`, `inventada`, or `incompleta`), a canonical ID for real citations, and an audit of the model calls and database queries.

The implemented order is **Extraction → Completeness → NER → Veracity**. The proposed **Extraction → NER → Sufficiency and recovery → Veracity** design is documented in [v3_suggestion.md](v3_suggestion.md); expansion and recovery are not implemented in v2.

## How v2 works

```mermaid
flowchart LR
    A[Original TXT] --> B[1. Extraction: LLM + local span matching]
    B --> C[2. Completeness: LLM]
    C --> D[3. NER: parallel field extractors]
    D --> E[4. Veracity: deterministic SQLite queries]
    E --> F[Predictions and evaluation reports]
```

### 1. Extraction

The LLM returns citation text, type, and confidence. Python calculates offsets; the model does not generate them.

The extractor first searches for the returned text literally. If that fails, it tries whitespace-normalized matching and maps the result back to the original document. A located citation stores the original slice `texto[inicio:fim]`, preserving its formatting. Offsets are Unicode character positions with an exclusive end.

Oversized documents are split into overlapping chunks with token budgets and capacity-error handling. Chunk offsets are translated to document offsets, and duplicate spans are merged. See [extraction limits and chunking](docs/extraction-limits.md).

With `extractor.debug: true`, unlocated candidates remain in checkpoints with null offsets. Otherwise, they are discarded. Unlocated candidates never enter the final submission. Normalizing whitespace does not repair every change to a number: removing a line break entirely can still prevent matching.

### 2. Completeness

An LLM returns only `completa: true` or `false`, using the citation and nearby context. It does not query the database or decide whether a citation is invented.

The current prompts require a searchable numbered reference for jurisprudence and a numbered provision plus an identifiable legal instrument for legislation. Context may join parts of the same reference but must not supply identifiers from a different citation.

This is a limitation of v2: references containing only court, year, and judge are marked incomplete before attempting a database search. The broader task definition also allows searchable but ambiguous references; v3 proposes moving this decision after NER and adding the corresponding search strategies.

### 3. Parallel NER

Every citation reaches NER, including those marked incomplete. Each extractor receives the original citation text and returns only its assigned fields. All calls use the configured entity model; they run independently within a shared concurrency limit.

| Extractor | Fields |
| --- | --- |
| Nature | `natureza`, `numero_sumula`, `sumula_vinculante` |
| Identifiers | `numero_processo_cnj`, `numero_classe_tribunal`, `numero_registro_tribunal` |
| Process class | `classe_processual`, `cadeia_recursal` |
| Court | `tribunal` |
| State | `uf` |
| Year | `ano` |
| Judge | `relator` |
| Legal instrument and article | `diploma`, `numero_artigo` |
| Legal instrument number | `numero_diploma` |

Jurisprudence uses the first seven extractors; legislation uses the last two. The coordinator merges their fields and applies deterministic normalization, including CNJ handling and judge-name normalization into `relator_norm`.

Each system prompt contains shared evidence rules, field-specific instructions, synthetic examples, and the response JSON schema. Allowed values, including all 27 UF codes, come from the Python contracts and are explicitly included in that schema. Additional synthetic evaluation cases are kept outside the prompts.

`entities.max_concurrency` limits active LLM requests across the coordinator. Each response is capped at 512 tokens. Failed requests are retried and audited; they are not silently converted into empty fields. See [parallel entity extraction](docs/parallel-entity-extraction.md).

### 4. Veracity

The standard verifier uses parameterized, read-only SQLite queries against structured columns. It does not call an LLM and has no automatic FTS fallback. A separate `4_veracity_fts` module exists as an alternative implementation.

| Condition | Current result |
| --- | --- |
| `completa: false`, or entity output unavailable | Skip lookup and return `incompleta`. |
| Extracted fields cannot form a supported query | Return `incompleta`. |
| Supported query finds no matching record | Return `inventada`. |
| One canonical record matches | Return `real` with its canonical ID. |
| Multiple records remain after supported disambiguation | Return `incompleta`. |

A complete citation can still be invented. Conversely, a citation may be incomplete because it lacks query information or because its query remains ambiguous. In v2, NER calls are still spent on citations whose completeness decision will prevent a lookup.

## Setup and configuration

Use Python 3.12 or newer and install the project dependencies:

```bash
uv sync
cp .env.example .env
```

Set `OPENAI_API_KEY` in the environment or `.env`. For a compatible endpoint that does not require authentication, the SDK still needs a nonempty placeholder such as `EMPTY`. Do not commit real credentials.

Copy and edit [configs/pipeline.yaml](configs/pipeline.yaml) for a run. Set:

- `input_dir`: original TXT documents.
- `workdir`: a fresh output directory.
- `database`: the enriched reference database.
- `model` and `base_url` for extraction, completeness, and entities: values supported by your inference endpoint.
- Sampling, concurrency, and retry settings for each model stage.

Paths are resolved from the working directory, so run the commands below from the project root. Pass `--config` explicitly; the stage CLIs otherwise default to a root-level `pipeline.yaml`.

| Setting | Behavior |
| --- | --- |
| `temperature`, `top_p`, `top_k`, `reasoning_effort` | Null omits the corresponding setting and uses the server default. Support depends on the endpoint. |
| `async_requests` | Enables concurrent extraction/completeness when using their standalone stage CLIs. |
| `max_concurrency` | Limits concurrent work; for NER, the shared runner limits individual LLM requests. |
| `entities.max_retries` | Maximum application attempts per extractor call. |
| `entities.request_timeout_seconds` | Per-attempt timeout, excluding time waiting for a concurrency slot. |
| `extractor.debug` | Keeps unlocated candidates in intermediate checkpoints when true. |
| `extractor.context_window_tokens`, `max_output_tokens`, `token_margin` | Control the extraction input/output budget. |
| `extractor.chunk_overlap_chars` | Adds shared context between extraction chunks. |
| `extractor.tokenizer_path` | Optional tokenizer endpoint; otherwise extraction uses a conservative byte-based estimate. |

The entity coordinator loads `relatores_padronizacao.json` from the database's parent directory. Keep the matching dictionary beside the selected database.

## Run the pipeline

The full entry point executes all four stages and exports predictions:

```bash
uv run desafio-jusbrasil --config configs/pipeline.yaml
```

The current orchestrator calls extraction and completeness synchronously and NER asynchronously. To use the configured asynchronous modes for the first two stages, run the standalone commands in order:

```bash
uv run python -m desafio_jusbrasil.1_extractor --config configs/pipeline.yaml
uv run python -m desafio_jusbrasil.2_completeness --config configs/pipeline.yaml
uv run python -m desafio_jusbrasil.3_entities --config configs/pipeline.yaml
uv run python -m desafio_jusbrasil.4_veracity --config configs/pipeline.yaml
```

Each stage consumes the preceding stage's saved checkpoints. Use a fresh `workdir` for each experiment. The last command also exports final prediction JSON files.

Database enrichment is a separate preparation step, configured in [configs/database_preprocessing.yaml](configs/database_preprocessing.yaml):

```bash
uv run python -m desafio_jusbrasil.database_preprocessing --config configs/database_preprocessing.yaml
```

## Evaluate and generate readable reports

Generate local stage references from the published development annotations:

```bash
uv run scripts/generate_stage_golds.py --output outputs/gold
```

Entity references also use local normalization and reviewed overrides; they are development annotations, not an independent official entity benchmark. See [gold rules](docs/gold-rules.md).

Evaluate a saved run, replacing `outputs/my-run` with its work directory:

```bash
uv run scripts/evaluate_extraction.py outputs/my-run --gold outputs/gold
uv run scripts/evaluate_completeness.py outputs/my-run --gold outputs/gold
uv run scripts/evaluate_entities.py outputs/my-run --gold outputs/gold
uv run scripts/evaluate_veracity.py outputs/my-run --gold outputs/gold
uv run scripts/clear_outputs.py outputs/my-run --gold outputs/gold
```

The report generator reads saved results without model calls. It creates:

```text
outputs/clear_outputs/<date-and-time>/
├── README.md
├── metrics_summary.md
├── citation_errors.md
├── manifest.json
└── <document_id>/
    ├── <document_id>.txt
    └── evaluation.md
```

`metrics_summary.md` starts with overall and per-stage processing times, followed by All/N1/N2 precision, recall, and F1 tables, strict overall correctness, and one NER field table with hits such as `100% (91/91)`. `citation_errors.md` lists each affected citation with the observed stage errors and expected/extracted values.

Timing is read from an existing `experiment.json`; ordinary stage commands do not automatically create this timing file. Missing timings are N/A. Concurrent request durations are never summed as elapsed time.

For experiments supplied with gold upstream checkpoints, declare that provenance explicitly, for example `--gold-supplied-stage 02-completeness`. Such stages are excluded from model scores, and strict overall is unavailable. See [readable report details](docs/clear-outputs.md).

Create a local submission CSV and calculate the separate official competition metric:

```bash
uv run python scripts/json_to_submission.py outputs/my-run/predictions outputs/my-run/submission.csv
uv run python scripts/evaluate.py outputs/my-run/submission.csv
```

These commands create and score local files; they do not submit anything to Kaggle.

## V2 development results

The full run on 2026-09-28 processed all 26 original documents through all four stages. It used `google/gemma-4-12B-it-qat-w4a16-ct`, concurrency 8, entity temperature 0, and `data/desafio1_bracis_enriched_gold.db`. Extraction and completeness retained server sampling defaults. This database and entity temperature differ from the editable defaults in `configs/pipeline.yaml`.

| Step | Precision | Recall | F1 | Wall time |
| --- | ---: | ---: | ---: | ---: |
| Extraction | 98.96% | 99.48% | 99.22% | 24.73 s |
| Completeness | 96.89% | 97.40% | 97.14% | 11.22 s |
| Entities | 94.30% | 94.79% | 94.55% | 56.12 s |
| Veracity | 94.82% | 95.31% | 95.06% | 1.33 s |
| Overall, strict | 91.19% | 91.67% | 91.43% | 93.41 s |

Strict overall requires the same citation to pass every stage, including citation type. Its F1 was 98.99% for N1 and 83.42% for N2. The run produced 194 candidates, of which 193 had located spans; matching found 191 of 192 reference citations. There were 19 citation error entries.

The official competition score was 1.023337 under its separate class/level weighting and calibration bonus, which can produce scores above 1. It is not strict overall F1.

The prompts use synthetic demonstrations, but both the corpus and synthetic evaluations informed prompt development. These numbers are development results, not an unseen-data benchmark. See [the full v2 evaluation](reports/full-pipeline-v2-20260928.md) and [NER prompt development](reports/ner-prompt-improvement-20260928.md). Detailed run artifacts under `outputs/` are local and excluded from Git.

## Checkpoints and audits

Each stage writes `<stage>/<document_id>/resultado.json` and numbered audit files. Intermediate stage directories are `01-extraction`, `02-completeness`, `03-entities`, and `04-veracity`; final JSON files are in `predictions/`.

NER audits identify the field extractor, citation, request, response, attempt, start time, duration, and queue time. Verification audits record the structured fields, SQL, parameters, matched records, and classification. Stage manifests record configuration. Evaluators ignore manifests and numbered audit files.

## Tests and known limitations

Run the local unit tests without inference:

```bash
uv run python -m unittest discover -s tests -v
```

The v2 check ran 68 tests: 63 passed, with five pre-existing failures/errors in database-preprocessing configuration/schema expectations and legacy FTS verifier tests. The added extraction, parallel NER, synthetic-example, and reporting regressions passed. The real full-run stage scores and all 14 NER field counts were also checked against the saved evaluators.

V2 does not repair fragmented citations, expand spans, or search jurisprudence using court/year/judge alone. The next design addresses these limitations through bounded re-extraction and NER recovery, with containment and neighboring-reference checks: [v3 proposal](v3_suggestion.md).
