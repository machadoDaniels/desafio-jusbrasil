# Parallel entity extraction: 2026-09-28

The grouped extractors processed all 26 documents / 192 gold upstream citations. The final run is functionally complete but regresses against the previous monolithic entity extractor.

- [Readable report index](../outputs/clear_outputs/2026-09-28_17-56-52_751542-0300/README.md)
- [Saved run](../outputs/parallel-entities-20260928T205541Z/)
- [Implementation and commands](../docs/parallel-entity-extraction.md)

## Results

| Metric | Previous candidate run | Parallel groups |
| --- | ---: | ---: |
| Exact entity records | 155/192 (80.73%) | 115/192 (59.90%) |
| Correct veracity classification and link | 191/192 (99.48%) | 131/192 (68.23%) |
| Entity elapsed time | ~46.1 seconds (reconstructed) | 49.32 seconds (measured) |
| LLM requests | 192 | 1,204 |
| Maximum concurrent LLM requests | Configured 8 | Measured 8 |

Verification and prediction export took 0.38 seconds. The final run had no failed requests or retries. Saved timing excludes evaluation and report rendering.

## Field comparison

| Field | Previous correct | Parallel correct | Total |
| --- | ---: | ---: | ---: |
| `jurisprudencia.ano` | 164 | 162 | 164 |
| `jurisprudencia.cadeia_recursal` | 157 | 137 | 164 |
| `jurisprudencia.classe_processual` | 159 | 123 | 164 |
| `jurisprudencia.natureza` | 164 | 164 | 164 |
| `jurisprudencia.numero_processo_cnj` | 164 | 161 | 164 |
| `jurisprudencia.numero_sumula` | 164 | 164 | 164 |
| `jurisprudencia.relator_norm` | 163 | 163 | 164 |
| `jurisprudencia.sequencias_numericas_identificadoras` | 161 | 109 | 164 |
| `jurisprudencia.sumula_vinculante` | 164 | 164 | 164 |
| `jurisprudencia.tribunal` | 161 | 157 | 164 |
| `jurisprudencia.uf` | 152 | 157 | 164 |
| `lei.diploma` | 28 | 28 | 28 |
| `lei.numero_artigo` | 28 | 28 | 28 |
| `lei.numero_diploma` | 14 | 28 | 28 |

The largest regressions are process identifiers and process classes. Law-number extraction improved from 14/28 to 28/28 after removing the instruction to infer numbers from code names. These are observations on the existing development data, not guarantees for unseen citations.

## Scope and reproducibility

Model: `google/gemma-4-12B-it-qat-w4a16-ct`; temperature 0; shared request limit 8; response limit 512 tokens. The original citation text is the only source evidence provided to each extractor. Final normalization and the judge dictionary remain in use.

Both comparisons use gold stage-2 candidates/completeness and the same saved `data/desafio1_bracis_enriched_gold.db`. Extraction and completeness were not rerun. Prompts were rewritten in English and dataset examples removed while grouping calls, so this comparison does not isolate parallel execution alone. Server load and weights are not controlled between runs.

The first attempt (`parallel-entities-20260928T204923Z`) was interrupted after multiple process-class timeouts. Response length and appeal-chain size were then bounded. The first completed attempt (`parallel-entities-20260928T205313Z`) scored 76/192 exact entity records and 134/192 correct veracity results. The final implementation also includes each response schema in the system prompt so allowed field values are visible to the model. All artifacts from earlier attempts were retained.

The run directory includes configuration, experiment timing, request statistics, source hashes, full request/response audits, entity/veracity evaluations and test logs.

## Validation

Six focused tests pass for shared concurrency, routing, failed-group retries, failure audit persistence, normalization, and making the schema visible in prompts. The full suite ran 59 tests with the same five pre-existing failures (54 passed). All 26 generated reports contain the expected 192 citations; copied input files match the originals byte for byte.
