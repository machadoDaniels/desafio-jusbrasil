# V2 full pipeline evaluation — 2026-09-28

All four stages ran from the original 26 TXT documents, using each preceding stage's predicted output. No gold citations or completeness labels were supplied to inference.

- [Metrics and NER hits](../outputs/clear_outputs/2026-09-28_19-45-26_898393-0300/metrics_summary.md)
- [Citation errors](../outputs/clear_outputs/2026-09-28_19-45-26_898393-0300/citation_errors.md)
- [Per-input reports](../outputs/clear_outputs/2026-09-28_19-45-26_898393-0300/README.md)
- [Configuration](../outputs/full-pipeline-20260928T224314Z/config.yaml)
- [Stage timings](../outputs/full-pipeline-20260928T224314Z/experiment.json)
- [Official competition evaluation](../outputs/full-pipeline-20260928T224314Z/competition-evaluation.json)
- [Verification evidence](../outputs/full-pipeline-20260928T224314Z/validation.json)

Model: `google/gemma-4-12B-it-qat-w4a16-ct`. Eight concurrent requests; improved parallel NER prompts. Entity temperature is 0; extraction/completeness use the configured server defaults. The reference database is `data/desafio1_bracis_enriched_gold.db`, unchanged from the previous experiment. Prompt source snapshots and hashes are saved with this run.

## Results across all inputs

| Step | Precision | Recall | F1 | Time |
| --- | ---: | ---: | ---: | ---: |
| Extraction | 98.96% | 99.48% | 99.22% | 24.73 s |
| Completeness | 96.89% | 97.40% | 97.14% | 11.22 s |
| Entities | 94.30% | 94.79% | 94.55% | 56.12 s |
| Veracity | 94.82% | 95.31% | 95.06% | 1.33 s |
| Overall, strict | 91.19% | 91.67% | 91.43% | 93.41 s |

Strict overall requires all four decisions to be correct for the same matched citation. Its F1 is 98.99% for N1 and 83.42% for N2. The total processing time includes export and small orchestration overhead; report generation is excluded.

The run produced 194 candidate citations: 193 have located spans and one is unlocated. Matching finds 191 of the 192 reference citations and two extra located outputs. The unlocated citation lost an original line break; it is excluded from precision denominators under the existing evaluators. All unmatched outputs and the missed reference appear in the error report. There are 19 error entries covering 34 distinct mismatches.

The official competition score is **1.023337**, using its separate class-level, level-weighted scoring rules and calibration bonus. The bonus allows scores above 1. N1 macro-F1 is 100%; N2 macro-F1 is 90.28%. This score is distinct from strict overall F1.

All stage score counts and all 14 NER field counts agree with the existing evaluators. All 26 input copies are byte-identical to the originals, report links resolve, and the source hashes remained unchanged during inference. The entity stage made 1,218 requests with no errors. These are development-corpus results, not an independent unseen-data benchmark.

## Reproduction

From the project root, use the saved runner in a fresh output folder with its own configuration. The runner records stage durations in `experiment.json`; `evaluate_run.py` evaluates a completed run and generates the reports. The configured inference endpoint accepts unauthenticated requests; the SDK was invoked with the non-secret placeholder `OPENAI_API_KEY=EMPTY`.

A local SDK initialization attempt before inference was preserved in `setup-attempt.json` and `setup-attempt.log`. It made no model calls; reported timings belong to the subsequent successful full run.

Detailed links above point to local run artifacts under `outputs/`, which are excluded from Git. The metrics in this summary are retained in version control.
