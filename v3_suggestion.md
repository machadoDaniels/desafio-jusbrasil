# Pipeline v3 suggestion

Status: proposed design. This document does not change the current implementation.

## Motivation

The current pipeline evaluates completeness before extracting entities and generally requires a numbered identifier for jurisprudence. This can reject references that could be searched using other fields, such as court, year, and judge. It also does not recover citations whose boundaries were extracted incorrectly.

The proposed pipeline extracts entities first, checks whether they support a meaningful query, and gives insufficient or suspiciously fragmented citations one recovery attempt before verification.

## Stage order

| Stage | Responsibility | Output |
| --- | --- | --- |
| 1. Extraction | Identify separate citation texts and locate them in the original document. | Citation candidates with type and document offsets. |
| 2. NER | Extract structured fields independently for each citation. | Entity fields attached to each candidate. |
| 3. Sufficiency and recovery | Check query sufficiency and, where needed, rerun extraction on a larger context window. Rerun NER only for an accepted changed citation. | Original or recovered citation, refreshed fields when applicable, and search eligibility. |
| 4. Veracity | Search the reference database and resolve candidates using available identifying fields. | `real`, `inventada`, or `incompleta`, with a canonical ID only for `real`. |

There is no separate LLM completeness gate before NER. Sufficiency means that the available fields support a meaningful query through an implemented search strategy; it does not mean that the citation exists or uniquely identifies a record.

## 1. Extraction

- The LLM identifies citation texts and types within the document or its chunks.
- Python locates those texts in the original document and assigns character offsets. Offsets use an inclusive start and exclusive end.
- Located citation text is taken from the original document slice, preserving formatting.
- Keep citations separate. Do not concatenate adjacent references merely because they occur in the same sentence or paragraph.

## 2. NER

Run the applicable entity extractors for every candidate before deciding sufficiency. Extract only evidence belonging to that citation, with the existing normalization rules.

A case number is one possible search key. Other combinations, including court, year, and judge, may support a search without guaranteeing a unique result. The verifier must implement the corresponding search strategies; changing the stage order alone does not add that capability.

## 3. Sufficiency and recovery

### When to attempt recovery

Give a citation at most one recovery attempt when:

- Its extracted fields do not support a meaningful query; or
- There is evidence of a broken boundary, such as a split number, an unfinished phrase, or an adjacent extracted fragment that may continue the reference.

Every insufficient citation gets this opportunity, even without a strong truncation signal. Missing fields alone cannot establish whether the original extraction was cut short. An intentionally brief citation may correctly remain unchanged.

Checking suspicious fragments also matters for citations that are already searchable: a truncated number can support a query but identify the wrong record or incorrectly produce no match.

### Recovery procedure

1. Preserve the target citation, its document offsets, and its original entity fields.
2. Build a bounded context window around the target using the original document. Expand the **input context**, not the accepted citation itself.
3. Run citation extraction on that window. Allow multiple separate citations; do not ask the model to merge everything in the window into the target.
4. Locate each returned citation in the original context and convert its local offsets to document-level offsets.
5. Select a candidate that contains the entire target span and is strictly longer:

   ```python
   contains_target = (
       candidate.start <= target.start
       and candidate.end >= target.end
   )
   is_longer = (
       candidate.end - candidate.start > target.end - target.start
   )
   ```

   These are conceptual field names; the current code uses `inicio` and `fim`.

6. Check whether the proposed candidate also encompasses another previously extracted citation. Accept a merge only when there is clear evidence that the original outputs were fragments of the same reference. Containment alone does not establish this.
7. Accept only one unambiguous qualifying candidate. Do not select the longest candidate automatically when several plausible candidates remain.
8. Rerun NER on the accepted citation's exact original text, then reassess query sufficiency.

Intersection and greater length alone are too permissive: a longer neighboring citation could overlap the target without being its continuation. Containment is the initial acceptance rule, followed by the separate-reference check. These checks reduce boundary errors but do not guarantee that the extractor is correct.

### Recovery outcomes

| Outcome | Action |
| --- | --- |
| One larger candidate qualifies and its refreshed fields support a query | Replace the target and pass the recovered citation to veracity. |
| One larger candidate qualifies but its refreshed fields remain insufficient | Preserve the recovered citation and classify it as `incompleta` without a database query. |
| No larger candidate qualifies, or the selection remains ambiguous | Keep the original citation and fields. If still insufficient, classify it as `incompleta`. |
| A previously searchable citation has an unsuccessful recovery attempt | Keep its original search eligibility and pass it to veracity. Recovery failure alone does not make it incomplete. |

If recovery combines fragments of one reference, replace those fragment outputs with one citation. Preserve unrelated neighboring citations. Coordinate overlapping recovery proposals so the same recovered citation is not emitted multiple times.

### Example

Original target:

```text
Recurso Especial nº 2.467-
```

Expanded context:

```text
Recurso Especial nº 2.467-
.648-RS foi citado junto ao REsp 12345/SP.
```

The expected recovery extraction returns two separate candidates:

```text
Recurso Especial nº 2.467-
.648-RS
```

```text
REsp 12345/SP
```

Select the first candidate because it contains and extends the target. Rerun NER on it. Keep the second reference separate. If `.648-RS` was previously emitted as its own fragment, replace it together with the original target after validating that they form one reference.

## 4. Veracity

| Condition | Classification |
| --- | --- |
| Available information cannot support a meaningful query after recovery | `incompleta`, without a database lookup. |
| A query with sufficient identifying information finds no matching record in the reference database | `inventada`. |
| Available identifying information resolves to one matching canonical record | `real`, with that record's canonical ID. |
| Multiple distinct canonical records remain after using the available disambiguating fields | `incompleta`. |

When an initial search returns multiple candidates, apply remaining relevant extracted fields before declaring unresolved ambiguity. Count distinct canonical records rather than duplicate rows created by joins or indexing.

A reference may therefore be incomplete either before searching or after an ambiguous search. Query sufficiency is separate from uniqueness. Query-construction failures or insufficiently identifying searches must not be treated as evidence that a citation is invented.

## Audit and evaluation

Preserve the original extraction alongside every recovery attempt:

- Target citation and original offsets.
- Recovery trigger and context-window offsets.
- Extraction request, response, and located candidates.
- Acceptance or rejection reason, including overlap with neighboring citations.
- Accepted span and refreshed NER fields, when changed.
- Fragment replacements and links to their original candidates.
- Search strategy, database results, disambiguation, and final classification.

Measure initial extraction and NER separately from their post-recovery results. Record recovery time and additional model calls, plus how many citations were corrected, remained unchanged, or regressed. Final metrics should use the deduplicated post-recovery citations. Keep initial errors visible so recovery does not hide the behavior of stage 1.

## Implementation decisions still to define

- The supported query strategies and field combinations for each citation type.
- Context-window size and limits for recovery.
- How to establish that an expansion crossing another candidate combines fragments rather than separate references.
- How to handle targets without located offsets. The containment rule requires a located span, so `null` offsets need a separate localization strategy and cannot silently enter this procedure.

Start with one recovery attempt per affected citation. Evaluate this bounded design before introducing repeated recovery or search-driven expansion loops.
