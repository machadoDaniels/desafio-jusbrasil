# Parallel entity extraction

Stage 3 now uses focused LLM calls coordinated by
`src/desafio_jusbrasil/entity_extraction/entity_extraction.py`. Both the full
pipeline and the standalone `3_entities` command use this coordinator.

| Module | Output fields |
| --- | --- |
| `natureza_extraction.py` | `natureza`, `numero_sumula`, `sumula_vinculante` |
| `numero_processo_cnj_extraction.py` | `numero_processo_cnj`, `numero_classe_tribunal`, `numero_registro_tribunal` |
| `classe_processual_extraction.py` | `classe_processual`, `cadeia_recursal` |
| `tribunal_extraction.py` | `tribunal` |
| `uf_extraction.py` | `uf` |
| `ano_extraction.py` | `ano` |
| `relator_extraction.py` | `relator` |
| `diploma_extraction.py` | `diploma`, `numero_artigo` |
| `numero_diploma_extraction.py` | `numero_diploma` |

Each module owns its prompt, strict response schema, and async `extract` function.
Jurisprudence citations run the first seven modules; legislation runs the last two.
Every module receives the original citation text, independently of other outputs.
Its response schema is included in the system prompt as well as the structured
response parameter, so the model sees field names and allowed values even when
the serving backend applies the schema only as a decoding constraint.
Prompts contain general extraction rules without worked examples from evaluation
citations. Court and law-number prompts explicitly prohibit filling absent values
from legal knowledge.

`shared.py` handles requests. `entities.max_concurrency` limits simultaneous LLM
requests across the entire coordinator, not separately for each field or document.
All modules use the configured entity model and sampling settings. Each response
is limited to 512 tokens; the appeal chain is bounded by the number of allowed
classes to prevent runaway repeated output. Only failed
calls are retried; the request timeout excludes time waiting for the concurrency
slot. Production clients disable SDK retries so each application attempt is audited.

The coordinator merges disjoint output fields and applies the existing CNJ,
súmula, class-label and judge-name normalization. A failed extractor raises an
error rather than returning null fields. Successful sibling calls and failed
attempts remain in the audit saved by stage 3. Completed citations remain in the
document checkpoint when a later citation fails.

The `03-entities/<document>/resultado.json` format is unchanged. Numbered audit
files now represent individual extractor attempts and include `extractor`,
`candidate`, `started_at`, `duration_seconds`, and `queue_seconds`, along with
the request, response, and extracted group fields. Audit order is deterministic
by citation, extractor, then attempt, rather than completion time.

Run using a configuration whose work directory already contains stage-2 inputs:

```bash
uv run python -m desafio_jusbrasil.3_entities --config path/to/config.yaml
uv run python -m desafio_jusbrasil.4_veracity --config path/to/config.yaml
uv run scripts/evaluate_entities.py outputs/my-run --gold outputs/my-gold/03-entities
uv run scripts/clear_outputs.py outputs/my-run --gold outputs/my-gold
```

Use a new work directory for comparison experiments. Reusing gold stage-2 inputs
isolates entity extraction; it does not measure the full pipeline or generalization
to unseen documents. Removing dataset examples also changes the prompt, so a
comparison with the old run does not isolate the effect of parallel grouping alone.

## Synthetic examples and prompt development

Each focused module now stores its invented demonstrations in `EXAMPLES`. The
shared formatter places them in the system prompt, clearly separated from the
actual user citation. Additional synthetic checks live in
`tests/fixtures/entity_prompts/`; these checks are not included in model prompts.
The corpus-overlap test validates the example outputs against their schemas and
rejects normalized example texts that occur in input documents.

To evaluate just one extractor, preserving prompts, source snapshots, audits and
per-field precision/recall in a new directory:

```bash
.venv/bin/python scripts/evaluate_entity_extractor.py tribunal \
  --output-dir outputs/prompt-trials/tribunal/trial-01
```

Use `--config PATH` and `--gold PATH` to select another model configuration or gold
entity directory. `--synthetic-only` runs just the additional synthetic cases.
The development evaluator uses temperature 0 and two concurrent requests so
multiple prompt workers can share the endpoint. Gold cases use existing final
normalization; synthetic checks compare the extractor's raw fields. The later
combined entity/veracity run verifies all merged, normalized fields together.

Synthetic cases and corpus labels used for prompt revision are development data.
High scores on them are not an independent measure of performance on unseen data.
