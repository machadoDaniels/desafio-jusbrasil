# Extraction and entity fixes — 2026-09-28

The reported context-limit failure, missing normalization dependency, and first
document's `APL`/`RSE` omissions have been addressed. Entity evaluation now covers
all 26 local development documents. This does not establish perfect extraction
on arbitrary inputs; the remaining measured errors are listed below.

## Changes

- Stage 1 splits input into overlapping ranges within its configured token budget,
  reserves output capacity, translates spans to original Unicode offsets, and
  merges duplicates. It can count tokens using the server's `/tokenize` endpoint
  or use a conservative UTF-8 byte estimate without that endpoint. Capacity errors
  and output truncation trigger smaller ranges. Other API errors still propagate.
- Extraction now returns a list of audit records, consistent with the other model
  stages. Each chunk request is preserved in a separate numbered JSON audit file.
- The full pipeline and standalone stage 3 share `AgenteExtratorEntidades.from_config`,
  which loads the judge-name dictionary beside the configured database.
- The jurisprudence prompt distinguishes class fields from numeric identifier
  fields, explicitly requires a one-item chain for a single class, and provides
  the allowed class names directly from the contract.

Configuration and behavior are described in [extraction-limits.md](../docs/extraction-limits.md).
Original input files, existing experiment outputs, and user pipeline settings were
preserved. No database was modified.

## Entity experiment

Model: `google/gemma-4-12B-it-qat-w4a16-ct`, temperature 0, concurrency 8. Other
sampling settings remained at the same server defaults. Both conditions used the
same normalization dictionary and all 192 gold stage-2 candidates, isolating
stage-3 changes from upstream extraction/completeness errors. The original prompt
was loaded before modification; the baseline source commit is
`8dc0dc32f37a40b1dd111e9195f97209562381d5`.

Reference: the local annotations, manual overrides, and normalization rules used
by `scripts/generate_stage_golds.py`. These are derived entity labels, not an
independent official entity leaderboard. Each condition was run once on development
data. The server did not expose an immutable weights revision.

| Metric | Original prompt, dictionary supplied | Revised prompt, dictionary supplied |
| --- | ---: | ---: |
| Exact entity records, all citation types | 136/192 (70.83%) | 155/192 (80.73%) |
| Exact entity records, N1 | 70/99 (70.71%) | 82/99 (82.83%) |
| Exact entity records, N2 | 66/93 (70.97%) | 73/93 (78.49%) |
| Process class accuracy, including nulls | 136/164 (82.93%) | 159/164 (96.95%) |
| Process class recall on populated gold fields | 103/131 (78.63%) | 126/131 (96.18%) |
| Appeal chain accuracy, including nulls | 131/164 (79.88%) | 157/164 (95.73%) |
| Appeal chain recall on populated gold fields | 98/131 (74.81%) | 124/131 (94.66%) |
| Normalized judge accuracy, including nulls | 163/164 (99.39%) | 163/164 (99.39%) |
| First document, exact entity records | 6/8 | 8/8 |

The normalization dictionary was supplied in both controlled conditions. Therefore
the 70.83% baseline is not the previously reported 37.5% first-document smoke score,
which also contained the orchestrator's missing-dictionary defect.

A separate fresh run through the real full-pipeline entry point produced **8/8
exact entity records** for `gen_n1_001`, confirming both the class omissions and
the three normalized names are corrected through that entry point.

### Remaining errors and side effects

The revised prompt still has five main-class and seven chain mismatches, involving
noisy/ambiguous abbreviations such as `AgR-REspe`, `AgREsp`, `AGR-RESPE`, `R-Rp`,
`TST-AgARR`, and `A.REsp`, plus an omitted `E` in a multi-class chain. These are
not the original single-class omissions. No blanket correctness claim is made.

UF accuracy decreased from 155/164 to 152/164; the aggregate entity score improved.
One normalized judge remains unmatched and legal-diploma-number accuracy remains
14/28 in both conditions. The latter compares model output with the project's
annotation policy, which is distinct from merely recognizing a legal code.

With gold upstream inputs and the same saved `data/desafio1_bracis_enriched_gold.db`,
the downstream verifier produced **191/192 correct classifications and links in
both conditions**. This checks downstream regression with fixed inputs; it is not
a full-pipeline score for all 26 documents and is not the competition's composite
score. The remaining mismatch was in `gen_n2_010`.

## Long-document experiment

Input: the 26 original documents, sorted by filename and joined with two newlines,
86,762 characters and 192 gold citations. Gold offsets were shifted to the combined
document. The server context limit was 16,384 tokens. Extraction used temperature
0, a 4,096-token output allowance, 256-token margin, and 400-character overlap.

| Condition | Inference calls | Matched / located predictions / gold | Precision | Recall | F1 |
| --- | ---: | --- | ---: | ---: | ---: |
| Original single request | 1 rejected | No extraction | — | — | — |
| Combined input, server tokenizer | 4 | 185 / 185 / 192 | 100% | 96.35% | 98.14% |
| Combined input, default byte estimate | 16 | 188 / 188 / 192 | 100% | 97.92% | 98.95% |
| Original documents separately, new default code | 26 | 192 / 193 / 192 | 99.48% | 100% | 99.74% |

Scoring uses the existing extraction evaluator's one-to-one IoU >= 0.5 matching.
All matched citation types were correct. The combined runs also retained six and
four unlocated model responses respectively under `debug: true`; these do not
count as located predictions. The default combined run's four misses correspond
to unlocated OCR/newline variants. The tokenizer run had one additional omitted
citation and six unlocated responses.

All recorded request budgets were within the configured/server limit, including
the output reserve and margin. Maximum actual prompt-plus-output usage was 10,374
tokens with server counting and 4,083 with the default estimate. The two combined
runs took about 117 and 116 seconds respectively; processing the original documents
separately with concurrency 8 took about 22 seconds. These are single-run timings,
not controlled throughput benchmarks; the server can be shared.

Chunking fixes capacity handling, not every model copying or recall error. The
default mode is conservative and creates more, smaller requests. Citations longer
than the overlap may still cross boundaries without appearing intact in a chunk.

## Code verification

All **10 focused regression tests pass**, covering sync/async parity, Unicode
offsets, repeated citations, overlap deduplication, server limits, fallback
estimation, context rejection, output truncation, unrelated API errors, impossible
budgets, multi-audit checkpoints, and both production normalization entry points.

The full suite ran **48 tests: 43 passed, with the same five failures present before
these changes**. No additional failures were introduced. Existing failures:

- `test_pipeline_fake_sem_rede_checkpoint_e_auditoria`: preprocessing fixture lacks `prompts`.
- `test_schema_expoe_descricoes_semanticas_ao_modelo`: missing preprocessing schema description.
- `test_numeros_e_cadeia_respeitam_contrato`: expected preprocessing validation is absent.
- `test_veracidade_deriva_termos_dos_campos_juridicos`: references an absent FTS method.
- `test_veracidade_monta_uma_consulta_fts_parametrizada`: expects FTS in the structured verifier.

Changed Python files pass formatting checks. Targeted lint passes excluding existing
numeric-module-name and unused-import findings; `git diff --check` passes.

## Artifacts and reproduction

Full local evidence is under `outputs/issue-resolution-20260928/`: original source
snapshots, baseline/candidate numbered model audits, manifests, per-document
evaluations, combined input/gold, extraction audits, full-pipeline smoke output,
and test log. `manifest.json` records SHA-256 hashes of source, data, dictionary,
and database. Those local artifacts are ignored by Git; this report preserves the
main results.

Useful commands from the repository root:

```bash
.venv/bin/python -m unittest discover -s tests -p test_issue_fixes.py -v
.venv/bin/python -m unittest discover -s tests -q
.venv/bin/python scripts/evaluate_entities.py outputs/issue-resolution-20260928/candidate --gold outputs/entities-score-gemma4-12b-20260928/generated-gold
.venv/bin/python scripts/evaluate_extraction.py outputs/issue-resolution-20260928/combined-default --gold outputs/issue-resolution-20260928/combined-gold
```

For a new isolated entity run, generate gold into a new directory with
`scripts/generate_stage_golds.py --output <gold-dir>`, copy its `02-completeness/`
into a new configured workdir, then run
`.venv/bin/python -m desafio_jusbrasil.3_entities --config <experiment.yaml>`.
Use the matching generated `03-entities/` as the evaluator's reference and keep
model settings fixed when comparing prompts. Do not overwrite historical gold
or experiment directories.
