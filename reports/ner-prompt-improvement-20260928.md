# NER prompt improvement with Luna: 2026-09-28

All nine extractor prompts were revised and tested by Luna workers. The final combined run matches 185/192 complete entity records (96.35%), compared with 115/192 (59.90%) for the earlier parallel prompts. No gold labels or evaluator rules were changed.

- [Readable reports for all 26 inputs](../outputs/clear_outputs/2026-09-28_18-21-05_527886-0300/README.md)
- [Saved combined run](../outputs/ner-prompts-20260928T211436Z/README.md)
- [Extractor implementation](../src/desafio_jusbrasil/entity_extraction/entity_extraction.py)
- [Development evaluation script](../scripts/evaluate_entity_extractor.py)

## Combined results

| Metric | Earlier monolithic run | Earlier parallel prompts | Improved parallel prompts |
| --- | ---: | ---: | ---: |
| Exact entity records | 155/192 (80.73%) | 115/192 (59.90%) | 185/192 (96.35%) |
| Correct veracity classification and link | 191/192 (99.48%) | 131/192 (68.23%) | 186/192 (96.88%) |
| Entity wall time | ~46.1 s (reconstructed) | 49.32 s | 54.73 s |
| Entity LLM calls | 192 | 1,204 | 1,204 |

Eleven of fourteen scored entity fields reached 100%. Veracity improved substantially over the first parallel version but remains below the older monolithic run. Entity-field accuracy and downstream classification accuracy are distinct measurements.

## Field results

| Field | Earlier parallel correct | Improved correct | Non-null precision | Non-null recall |
| --- | ---: | ---: | ---: | ---: |
| `jurisprudencia.ano` | 162/164 | 164/164 | 100.00% | 100.00% |
| `jurisprudencia.cadeia_recursal` | 137/164 | 164/164 | 100.00% | 100.00% |
| `jurisprudencia.classe_processual` | 123/164 | 164/164 | 100.00% | 100.00% |
| `jurisprudencia.natureza` | 164/164 | 164/164 | 100.00% | 100.00% |
| `jurisprudencia.numero_processo_cnj` | 161/164 | 164/164 | 100.00% | 100.00% |
| `jurisprudencia.numero_sumula` | 164/164 | 164/164 | 100.00% | 100.00% |
| `jurisprudencia.relator_norm` | 163/164 | 163/164 | 100.00% | 96.88% |
| `jurisprudencia.sequencias_numericas_identificadoras` | 109/164 | 159/164 | 94.12% | 96.39% |
| `jurisprudencia.sumula_vinculante` | 164/164 | 164/164 | 100.00% | 100.00% |
| `jurisprudencia.tribunal` | 157/164 | 164/164 | 100.00% | 100.00% |
| `jurisprudencia.uf` | 157/164 | 163/164 | 98.98% | 100.00% |
| `lei.diploma` | 28/28 | 28/28 | 100.00% | 100.00% |
| `lei.numero_artigo` | 28/28 | 28/28 | 100.00% | 100.00% |
| `lei.numero_diploma` | 28/28 | 28/28 | 100.00% | 100.00% |

Tribunal class and registration identifiers are scored together as a set. Raw `relator` is not scored; its normalized value is. All metrics use existing deterministic normalization.

## Agent development trials

Each extractor received a separate Luna assignment. Workers were reused for the last assignments when the available thread limit was reached. Agents used the old prompt for general definitions, wrote synthetic demonstrations, and tested at temperature 0 with two concurrent Gemma requests. Every trial retained its source, prompt, cases, raw request/response audits, and scores.

| Extractor | Selected trial | Gold exact | Separate synthetic checks |
| --- | --- | ---: | ---: |
| `ano` | [round-02](../outputs/prompt-improvement-20260928/ano/round-02/report.json) | 164/164 | 8/8 |
| `classe_processual` | [round-05](../outputs/prompt-improvement-20260928/classe_processual/round-05/report.json) | 164/164 | 11/11 |
| `diploma` | [round-04](../outputs/prompt-improvement-20260928/diploma/round-04/report.json) | 28/28 | 7/7 |
| `natureza` | [round-01](../outputs/prompt-improvement-20260928/natureza/round-01/report.json) | 164/164 | 8/8 |
| `numero_diploma` | [round-01](../outputs/prompt-improvement-20260928/numero_diploma/round-01/report.json) | 28/28 | 8/8 |
| `numero_processo_cnj` | [round-02](../outputs/prompt-improvement-20260928/numero_processo_cnj/round-02/report.json) | 159/164 | 8/8 |
| `relator` | [round-02](../outputs/prompt-improvement-20260928/relator/round-02/report.json) | 163/164 | 8/8 |
| `tribunal` | [round-01](../outputs/prompt-improvement-20260928/tribunal/round-01/report.json) | 164/164 | 8/8 |
| `uf` | [round-01](../outputs/prompt-improvement-20260928/uf/round-01/report.json) | 163/164 | 8/8 |

All 74 additional synthetic checks passed. The 57 prompt demonstrations and the additional checks validated against their response schemas, were disjoint, and had no normalized full-text overlap with the input corpus. An additional audit found no complete gold citation embedded inside a longer prompt demonstration. Canonical legal-code identities and standard class abbreviations remain permitted domain definitions.

The process-class file received a small wording/line-wrap cleanup after its selected agent trial. The final combined run used the frozen current source and independently confirmed 164/164 for both class and appeal chain. Source hashes and snapshots are saved with the combined run.

## Remaining entity errors

- Five process-identifier records: a broken number was truncated; OCR letters were dropped in two numbers; two ambiguous identifiers were populated where gold expects null.
- One unsupported UF value (`RS`) was returned for a citation without an explicit state.
- One explicitly copied OCR judge name could not be resolved by the existing name dictionary.

These seven mismatches are preserved in the reports. No per-document overrides, gold edits, or scoring exceptions were introduced.

## Scope and verification

Model: `google/gemma-4-12B-it-qat-w4a16-ct`; temperature 0; global request limit 8; 512 output tokens per call. The final run had 1,204 successful requests, no retries, and a measured peak of eight concurrent calls. Entity extraction took 54.73 seconds; verification and export took 1.37 seconds.

The same 26 documents / 192 gold upstream candidates and completeness decisions were reused. Extraction and completeness were not rerun. The database and downstream verifier were unchanged. Both corpus labels and synthetic checks were used for prompt development, so these results are not an independent unseen-data benchmark. Runtime comparisons are single trials on a shared endpoint, not controlled throughput measurements.

The full suite ran 60 tests: 55 passed, with the same five pre-existing failures in database preprocessing and legacy FTS tests. The new synthetic-example checks and six parallel extraction tests passed. All 26 generated reports were checked for 192 citation sections and byte-identical original TXT copies.
