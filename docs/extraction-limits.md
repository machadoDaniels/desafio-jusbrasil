# Extraction limits and chunking

The extractor splits oversized documents into overlapping ranges without changing
the source text. Both synchronous and asynchronous execution use the same budget,
splitting, offset translation, and merge rules.

The `extractor` section accepts these settings in addition to the existing stage
configuration:

```yaml
extractor:
  model: google/gemma-4-12B-it-qat-w4a16-ct
  context_window_tokens: 16384
  max_output_tokens: 4096
  token_margin: 256
  chunk_overlap_chars: 400
  tokenizer_path: null
```

These are the defaults. Existing configurations remain valid. Set the context
window to the deployed server's limit, rather than the model family's advertised
maximum. The output allowance and margin must leave room for the prompt.

## Token counting

For vLLM, set `tokenizer_path: /tokenize`. The extractor calls that path on the
same origin as `base_url`, sends the complete chat messages, and uses the returned
token count and `max_model_len`. It respects the smaller of the configured and
reported context limits. The tokenizer endpoint must be reachable and compatible
with vLLM's request/response format; a configured endpoint failure is surfaced.

When no tokenizer path is configured, the extractor conservatively estimates
input tokens from the UTF-8 byte length of messages and the response schema. This
does not require a tokenizer dependency or another endpoint. It is an estimate,
not an exact tokenizer guarantee, and usually creates smaller chunks. Audit files
identify which counting method was used.

Before inference, the extractor reserves `max_output_tokens` plus `token_margin`.
Each model request also sets `max_completion_tokens`. Context-capacity rejections
and output-length truncation cause further splitting; unrelated API errors are
not treated as capacity errors. A request that cannot fit even a one-character
range fails explicitly instead of looping or silently dropping the document.

## Boundaries and results

Splitting prefers a nearby paragraph boundary and includes up to
`chunk_overlap_chars` characters of shared context. Overlap shrinks for very small
ranges so every split makes progress. A citation crossing a split can be extracted
from the overlapping range. A citation longer than the overlap is not guaranteed
to appear intact in either range; increase overlap when the corpus contains such
citations and validate recall on representative inputs.

Each extracted span is located inside its own chunk and translated back to the
original document's Unicode character offsets. Identical spans are merged while
separate occurrences of the same text remain separate citations. A partial span
touching an artificial boundary is removed when a longer same-type span from an
overlap contains it. Unlocated candidates remain available when `debug` is true.

`extrair_auditada` and `extrair_auditada_async` now return
`(candidates, list_of_audits)`, matching the audit-list pattern in the other model
stages. The stage writes each inference attempt as its own numbered audit file,
including failed capacity attempts, chunk offsets, and successful request budgets.
Callers that previously wrapped a single audit in a list should pass the returned
list directly. The saved `DocumentoExtraido` format is unchanged.

## Entity normalization and class extraction

Both the full pipeline and standalone entity stage construct their agent through
`AgenteExtratorEntidades.from_config`. It loads `relatores_padronizacao.json` from
the configured database's parent directory. A missing file fails at initialization
instead of silently using an empty dictionary in these production entry points.

The jurisprudence prompt now explicitly separates numeric identifier fields from
class fields, requires a one-item chain when only one class is stated, and includes
the allowed class names from the contract. These are model instructions, not a
guarantee of semantic correctness; evaluate entity fields independently of the
final citation classification.

Run the focused regressions with:

```bash
.venv/bin/python -m unittest discover -s tests -p test_issue_fixes.py -v
```
