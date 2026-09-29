# Readable citation reports

Generate reports from an existing pipeline run:

```bash
uv run scripts/clear_outputs.py outputs/my-run --gold outputs/my-gold
```

The run directory contains `01-extraction`, `02-completeness`, `03-entities`, and
`04-veracity` checkpoints. Missing stages are allowed. Gold must include
`03-entities`. Both nested `<document>/resultado.json` and legacy flat
`<document>.json` checkpoints are supported. Numbered audit files are ignored.

For the saved 26-document entity experiment:

```bash
uv run scripts/clear_outputs.py \
  outputs/ner-prompts-20260928T211436Z \
  --gold outputs/entities-score-gemma4-12b-20260928/generated-gold \
  --gold-supplied-stage 02-completeness
```

This experiment used gold candidates and completeness as upstream inputs, so its
reports describe saved entity/veracity results, not a full pipeline evaluation.

Results go to `outputs/clear_outputs/<local-date-and-time-with-offset>/`. Open the
`README.md` index. Each input has its own folder containing the original TXT,
copied byte for byte, and `evaluation.md`. A manifest records source paths.

Two additional Markdown files appear at the root:

- `metrics_summary.md`: a first table with overall and per-step processing time;
  separate All, N1, and N2 score tables for extraction,
  completeness, entities, veracity, and strict overall correctness; one NER table
  with all scored fields, formatted as `100% (91/91)`; and links to input errors.
- `citation_errors.md`: each affected citation once, with expected/extracted text,
  offsets, links to the full report, and explicit stage/field/value mismatches.
  Missing expected citations and extra outputs are included. Repeated span/type
  errors across stages are combined.

Processing times come from the source run's `experiment.json`: `elapsed_seconds`,
`extraction_seconds`, `completeness_seconds`, `entity_seconds`, and
`veracity_seconds`. The existing `veracity_and_materialization_seconds` is also
supported and explicitly labeled as including prediction export. Missing timings
and steps supplied from gold are N/A. Overall covers the recorded run, which may
include only some pipeline steps. Concurrent request durations are never summed,
and report-generation time is not presented as model processing time.

Scores pool counts across documents. Precision is correct/located predictions,
recall is correct/expected citations, and F1 is their harmonic mean. Extraction
scores matched spans; strict overall also requires the correct citation type and
every downstream decision for the same pair. Overall is not the official
competition score. NER hits include correct nulls and use matched citations with
available entity outputs and matching types. Identifier and appeal-chain sets use
the existing evaluator's comparison rules. Zero denominators are shown as N/A.

Each stage is evaluated from its own checkpoint. Absent stages and missing gold
references are marked unavailable; gold-only documents count as missed outputs
when the stage otherwise has predictions. Both prediction and gold document IDs
are included, so `--input-dir` must contain their original texts.

Use `--gold-supplied-stage STAGE` for stages copied from reference data. Repeat
the option for multiple stages. This explicit provenance is recorded in the
manifest; identical predictions alone do not establish that gold was supplied.
Such stages are excluded from scores, and strict overall is unavailable unless
all four steps were evaluated. No mean-per-input or matched-accuracy tables are
generated.

Each citation shows completeness, the extracted and matched expected citation,
an entity-field comparison, and saved veracity. Completeness means `completa`,
not `inventada`; invention is determined by saved veracity. An `incompleta`
result leaves reality undetermined. Missing outputs remain explicitly unavailable.
Unmatched gold citations appear at the end of the report. Matching and entity
comparisons reuse the existing evaluator's rules.

Use `--input-dir PATH` for other input texts or `--output-root PATH` for another
destination. Each invocation creates a new timestamped directory and reads only
existing artifacts; it does not call a model or rerun verification.
