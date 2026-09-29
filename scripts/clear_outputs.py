# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy>=2.0", "pandas>=2.0", "pydantic>=2.0", "pyyaml>=6.0"]
# ///
"""Turn saved pipeline checkpoints into per-document citation reports.

Example:
    uv run scripts/clear_outputs.py outputs/my-run --gold outputs/my-gold

This reads saved results only; it does not run models or database searches.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from evaluate_entities import _CAMPOS, _casar, _comparar_campos
from desafio_jusbrasil.contracts import ConsultaJurisprudencia, ConsultaLegislacao

ROOT = Path(__file__).resolve().parents[1]
STAGES = ("01-extraction", "02-completeness", "03-entities", "04-veracity")
ENTITY_MODELS = {"jurisprudencia": ConsultaJurisprudencia, "lei": ConsultaLegislacao}
STAGE_NAMES = dict(zip(STAGES, ("Extraction", "Completeness", "Entities", "Veracity")))
PAYLOADS = dict(zip(STAGES[1:], ("completude", "campos_extraidos", "veracidade")))


def load_documents(directory: Path) -> dict[str, dict[str, Any]]:
    """Accept nested checkpoints and legacy flat files, excluding audit files."""
    documents = {}
    paths = sorted(directory.glob("*.json")) + sorted(directory.glob("*/resultado.json"))
    for path in paths:
        if path.stem.endswith("_eval"):
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and "documento_id" in data and "candidatos" in data:
            document_id = data["documento_id"]
            if Path(document_id).name != document_id or document_id in {".", ".."}:
                raise ValueError(f"Invalid document ID in {path}")
            if document_id in documents:
                raise ValueError(f"Duplicate checkpoint for {document_id} in {directory}")
            documents[document_id] = {"path": path, "items": data["candidatos"]}
    return documents


def merge_stages(stages: dict[str, dict], document_id: str) -> list[dict]:
    """Join the same citation across stages by span, text, type and occurrence."""
    records: dict[tuple, dict] = {}
    for stage in STAGES:
        occurrences: Counter = Counter()
        for item in stages[stage].get(document_id, {}).get("items", []):
            candidate = item.get("candidato", item)
            identity = tuple(candidate.get(field) for field in ("inicio", "fim", "tipo", "trecho"))
            key = (*identity, occurrences[identity])
            occurrences[identity] += 1
            record = records.setdefault(key, {"candidate": candidate, "sources": {}})
            record["sources"][stage] = item
            for field in ("completude", "campos_extraidos", "veracidade"):
                if field in item:
                    record[field] = item[field]
    return sorted(records.values(), key=lambda record: (
        record["candidate"].get("inicio") is None,
        record["candidate"].get("inicio") or 0,
        record["candidate"].get("fim") or 0,
    ))


def match_records(expected: list[dict], extracted: list[dict]) -> dict[int, int]:
    gold_indices = [i for i, r in enumerate(expected) if located(r)]
    pred_indices = [i for i, r in enumerate(extracted) if located(r)]
    pairs, _, _ = _casar(
        [expected[i]["candidate"] for i in gold_indices],
        [extracted[i]["candidate"] for i in pred_indices],
    )
    return {pred_indices[pi]: gold_indices[gi] for gi, pi in pairs}


def located(record: dict) -> bool:
    candidate = record["candidate"]
    return candidate.get("inicio") is not None and candidate.get("fim") is not None


def cell(value: Any) -> str:
    # HTML code avoids broken Markdown tables for pipes, backticks and newlines.
    return "<code>" + html.escape(json.dumps(value, ensure_ascii=False)).replace("|", "&#124;") + "</code>"


def quote(text: str) -> str:
    fence = "```"
    while fence in text:
        fence += "`"
    return f"{fence}text\n{text}\n{fence}"


def entity_table(expected: dict | None, extracted: dict) -> list[str]:
    candidate_type = extracted["candidate"]["tipo"]
    expected_fields = expected.get("campos_extraidos") if expected else None
    extracted_fields = extracted.get("campos_extraidos")
    model = ENTITY_MODELS[candidate_type]
    expected_type = expected["candidate"]["tipo"] if expected else candidate_type
    expected_schema = ENTITY_MODELS[expected_type]
    expected_model = expected_schema.model_validate(expected_fields) if expected_fields is not None else None
    extracted_model = model.model_validate(extracted_fields) if extracted_fields is not None else None
    mismatches = []
    lines = []
    if expected_type != candidate_type:
        lines.append("Entity comparison: citation type mismatch; fields are shown without scoring.")
    elif expected_model is not None and extracted_model is not None:
        mismatches, _ = _comparar_campos(expected_model, extracted_model, candidate_type)
        lines.append("Entity comparison: " + (
            "mismatch in " + ", ".join(f"`{field}`" for field in mismatches) + "."
            if mismatches else "all evaluated fields match."
        ))
    else:
        lines.append("Entity comparison: unavailable (expected or extracted entities are missing).")
    lines += ["", "| Entity field | Expected value | Extracted value |", "| --- | --- | --- |"]
    for field in dict.fromkeys((*expected_schema.model_fields, *model.model_fields)):
        left = cell(getattr(expected_model, field)) if expected_model and hasattr(expected_model, field) else "Unavailable"
        right = cell(getattr(extracted_model, field)) if extracted_model and hasattr(extracted_model, field) else "Unavailable"
        scored_field = "sequencias_numericas_identificadoras" if field in {
            "numero_classe_tribunal", "numero_registro_tribunal"
        } else field
        if scored_field in mismatches:
            right = f"**{right}**"
        lines.append(f"| `{field}` | {left} | {right} |")
    return lines


def render_report(document_id: str, extracted: list[dict], expected: list[dict], run: Path,
                  gold: Path, timestamp: str, has_extraction: bool) -> str:
    matches = match_records(expected, extracted)
    lines = [f"# Citation evaluation: {document_id}", "",
             f"Generated: {timestamp}. Source run: `{run}`. Gold: `{gold}`.", "",
             f"Input: [{document_id}.txt]({document_id}.txt). "
             f"Citations: {len(extracted)}; matched gold citations: {len(matches)}/{len(expected)}.", "",
             "Completeness reports `completa`; it does not decide whether a citation is invented. "
             "The `inventada`, `real`, and `incompleta` labels below come from saved veracity output. "
             "An unmatched gold span does not by itself mean the citation is invented.", "",
             "Gold matching uses the evaluator's one-to-one span IoU >= 0.5 rule. "
             "Bold entity values mark scored mismatches. `null` is a stored empty field; "
             "Unavailable means the corresponding record is absent. Raw `relator` is not scored; "
             "`relator_norm` is. The two tribunal identifiers are compared together as a set, "
             "and appeal-chain order and repetitions are ignored.", ""]
    if not has_extraction:
        lines += ["No stage-1 checkpoint exists in this run. Citations and completeness values "
                  "are read from downstream checkpoints; their presence does not establish that "
                  "extraction or completeness was run by this experiment.", ""]
    for number, record in enumerate(extracted, 1):
        candidate = record["candidate"]
        reference = expected[matches[number - 1]] if number - 1 in matches else None
        completeness = record.get("completude", {}).get("completa")
        veracity = record.get("veracidade") or {}
        classification = veracity.get("classificacao")
        invented = {"inventada": "Yes", "real": "No", "incompleta": "Undetermined (incomplete citation)"}.get(
            classification, "Unavailable")
        lines += [f"## Citation {number}", "", "### Completeness", "",
                  f"Output: `completa = {json.dumps(completeness)}`." if completeness is not None
                  else "Output: Unavailable.", "",
                  f"Invented (`inventada`): **{invented}** (saved veracity result).", "",
                  "### Extracted citation", "", quote(candidate["trecho"]), "",
                  f"Type: `{candidate['tipo']}`. Character offsets: "
                  f"`{candidate.get('inicio')}:{candidate.get('fim')}` (end exclusive).", "",
                  "### Expected citation", ""]
        if reference:
            gold_candidate = reference["candidate"]
            lines += [quote(gold_candidate["trecho"]), "",
                      f"Type: `{gold_candidate['tipo']}`. Character offsets: "
                      f"`{gold_candidate.get('inicio')}:{gold_candidate.get('fim')}`.", "",
                      "Expected completeness: " + cell(reference.get("completude", {}).get("completa")) + ".", ""]
        else:
            lines += ["No matching gold citation is available.", ""]
        lines += entity_table(reference, record) + ["", "### Veracity", ""]
        if classification == "inventada":
            lines += ["Output: **`inventada`**. Real: **No**."]
        elif classification == "real":
            lines += ["Output: **`real`**. Real: **Yes**."]
        elif classification == "incompleta":
            lines += ["Output: **`incompleta`**. Real: **Undetermined**."]
        else:
            lines += ["Output: Unavailable. Real: **Undetermined**."]
        if veracity.get("id_canonico") is not None:
            lines += ["", f"Canonical ID: `{veracity['id_canonico']}`."]
        if veracity.get("justificativa"):
            lines += ["", "Saved explanation:", "", quote(veracity["justificativa"])]
        expected_veracity = reference.get("veracidade", {}) if reference else {}
        if expected_veracity:
            lines += ["", f"Expected classification: `{expected_veracity.get('classificacao')}`."]
        lines += [""]
    missed = [record for i, record in enumerate(expected) if i not in matches.values()]
    if missed:
        lines += ["## Gold citations without a matched output", ""]
        for record in missed:
            candidate = record["candidate"]
            lines += [f"Offsets `{candidate.get('inicio')}:{candidate.get('fim')}`:", "",
                      quote(candidate["trecho"]), ""]
    return "\n".join(lines)


def stage_record(record: dict, stage: str) -> dict:
    """Read only this stage's checkpoint, never an inherited downstream value."""
    item = record["sources"][stage]
    return {"candidate": item.get("candidato", item),
            **{key: item[key] for key in PAYLOADS.values() if key in item}}


def compare_stage(stage: str, expected: dict, extracted: dict) -> tuple[list[dict], dict]:
    """Return explicit mismatches and scored entity values for a matched pair."""
    errors = []
    fields = {}

    def mismatch(field, left, right, description):
        errors.append({"field": field, "expected": left, "output": right,
                       "description": description})

    expected_type = expected["candidate"]["tipo"]
    extracted_type = extracted["candidate"]["tipo"]
    if stage in (STAGES[0], STAGES[2]) and expected_type != extracted_type:
        mismatch("tipo", expected_type, extracted_type, "Citation type differs from the reference.")
    if stage == STAGES[1]:
        left = (expected.get("completude") or {}).get("completa")
        right = (extracted.get("completude") or {}).get("completa")
        if right is None or left != right:
            mismatch("completa", left, right, "Completeness decision differs or is missing.")
    elif stage == STAGES[2]:
        left = expected.get("campos_extraidos")
        right = extracted.get("campos_extraidos")
        if right is None:
            mismatch("campos_extraidos", left, None, "Entity output is unavailable.")
        elif expected_type == extracted_type:
            model = ENTITY_MODELS[expected_type]
            wrong, fields = _comparar_campos(model.model_validate(left), model.model_validate(right), expected_type)
            for field in wrong:
                mismatch(field, *fields[field], "Extracted value differs after evaluator normalization.")
    elif stage == STAGES[3]:
        left = expected.get("veracidade") or {}
        right = extracted.get("veracidade") or {}
        if right.get("classificacao") is None or left.get("classificacao") != right.get("classificacao"):
            mismatch("classificacao", left.get("classificacao"), right.get("classificacao"),
                     "Veracity classification differs or is missing.")
        if left.get("classificacao") == "real" and left.get("id_canonico") != right.get("id_canonico"):
            mismatch("id_canonico", left.get("id_canonico"), right.get("id_canonico"),
                     "The expected real citation's canonical ID differs or is missing.")
    return errors, fields


def evaluate_document(document_id: str, extracted: list[dict], expected: list[dict],
                      stages: dict, gold_stages: dict, gold_supplied: set[str]) -> dict:
    """Evaluate each saved stage independently and deduplicate errors by citation."""
    matches = match_records(expected, extracted)
    reverse_matches = {gi: pi for pi, gi in matches.items()}
    result = {"document_id": document_id, "scores": {}, "fields": {}, "errors": {}}
    correct_pairs = []

    def add_error(pi, gi, stage, error):
        if pi is None:
            pi = reverse_matches.get(gi)
        if gi is None:
            gi = matches.get(pi)
        key = ("output", pi) if pi is not None else ("missing", gi)
        entry = result["errors"].setdefault(key, {
            "number": pi + 1 if pi is not None else gi + 1,
            "expected": expected[gi] if gi is not None else None,
            "extracted": extracted[pi] if pi is not None else None,
            "issues": [],
        })
        # Missing/extra spans and type errors may recur in several checkpoints.
        previous = next((issue for issue in entry["issues"] if all(
            issue[k] == error[k] for k in error)), None)
        if previous is not None:
            if STAGE_NAMES[stage] not in previous["steps"]:
                previous["steps"].append(STAGE_NAMES[stage])
        else:
            entry["issues"].append({**error, "steps": [STAGE_NAMES[stage]]})

    for stage in STAGES:
        pred_indices = [i for i, record in enumerate(extracted) if stage in record["sources"]]
        gold_indices = [i for i, record in enumerate(expected) if stage in record["sources"]]
        predictions = [stage_record(extracted[i], stage) for i in pred_indices]
        references = [stage_record(expected[i], stage) for i in gold_indices]
        status = "Scored"
        if stage in gold_supplied:
            status = "Gold supplied"
        elif not stages[stage]:
            status = "Unavailable"
        elif document_id not in gold_stages[stage]:
            status = "Reference unavailable"
        elif stage in PAYLOADS and any(record.get(PAYLOADS[stage]) is None for record in references):
            status = "Reference unavailable"
        score = {"status": status, "correct": 0,
                 "predicted": sum(located(r) for r in predictions), "expected": len(references)}
        result["scores"][stage] = score
        if status != "Scored":
            continue
        stage_matches = match_records(references, predictions)
        successful = set()
        for pi, gi in stage_matches.items():
            errors, fields = compare_stage(stage, references[gi], predictions[pi])
            # Extraction precision/recall scores spans; strict overall also requires type.
            if not errors or stage == STAGES[0]:
                score["correct"] += 1
            if not errors:
                successful.add((pred_indices[pi], gold_indices[gi]))
            if stage == STAGES[2]:
                for field, values in fields.items():
                    counts = result["fields"].setdefault(field, Counter())
                    counts["total"] += 1
                    counts["hits"] += not any(error["field"] == field for error in errors)
            for error in errors:
                add_error(pred_indices[pi], gold_indices[gi], stage, error)
        correct_pairs.append(successful)
        for pi, prediction in enumerate(predictions):
            if pi not in stage_matches:
                description = ("Output has no located span; excluded from precision denominators."
                               if not located(prediction) else "No expected span matches this output at IoU >= 0.5.")
                add_error(pred_indices[pi], None, stage, {
                    "field": "citation span", "expected": None,
                    "output": prediction["candidate"]["trecho"], "description": description})
        for gi, reference in enumerate(references):
            if gi not in stage_matches.values():
                add_error(None, gold_indices[gi], stage, {
                    "field": "citation span", "expected": reference["candidate"]["trecho"],
                    "output": None, "description": "Expected citation has no matched output at IoU >= 0.5."})
    all_scored = all(score["status"] == "Scored" for score in result["scores"].values())
    result["scores"]["overall"] = {
        "status": "Scored" if all_scored else "Requires all four steps",
        "correct": len(set.intersection(*correct_pairs)) if all_scored else 0,
        "predicted": result["scores"][STAGES[-1]]["predicted"],
        "expected": result["scores"][STAGES[0]]["expected"],
    }
    return result


def percentage(numerator: int, denominator: int) -> str:
    return f"{100 * numerator / denominator:.2f}".rstrip("0").rstrip(".") + "%" if denominator else "N/A"


def group_documents(documents: list[dict], group: str) -> list[dict]:
    return [doc for doc in documents if group == "All" or f"_{group.lower()}_" in doc["document_id"]]


def render_timing(documents: list[dict], run: Path) -> list[str]:
    """Report recorded wall time, without adding concurrent request durations."""
    source = run / "experiment.json"
    metadata = json.loads(source.read_text(encoding="utf-8")) if source.is_file() else {}

    def seconds(key: str) -> float | None:
        value = metadata.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0:
            return value
        return None

    lines = ["## Processing time", "", "| Step | Elapsed time | Notes |", "| --- | ---: | --- |"]
    overall = seconds("elapsed_seconds")
    total_note = "Recorded run wall time, including overhead; covers only the steps executed."
    if metadata.get("status") and metadata["status"] != "complete":
        total_note += f" Run status: {cell(metadata['status'])}."
    lines.append(f"| Overall | {f'{overall:.2f} s' if overall is not None else 'N/A'} | "
                 f"{total_note if overall is not None else 'Total duration was not recorded.'} |")
    keys = dict(zip(STAGES, ("extraction_seconds", "completeness_seconds", "entity_seconds", "veracity_seconds")))
    for stage, name in STAGE_NAMES.items():
        statuses = {doc["scores"][stage]["status"] for doc in documents}
        duration = seconds(keys[stage])
        note = "Recorded stage wall time."
        if statuses == {"Gold supplied"}:
            duration, note = None, "Gold supplied; this step was not rerun."
        elif statuses == {"Unavailable"}:
            duration, note = None, "No stage checkpoint is available."
        else:
            if duration is None and stage == STAGES[-1]:
                duration = seconds("veracity_and_materialization_seconds")
                note = "Includes prediction export; verification alone was not timed separately."
            if duration is None:
                note = "Stage duration was not recorded."
        lines.append(f"| {name} | {f'{duration:.2f} s' if duration is not None else 'N/A'} | {note} |")
    lines += ["", "Elapsed times describe processing the source run, not generating this report. "
              "Concurrent LLM request durations are not summed.", ""]
    if source.is_file():
        lines += [f"Timing source: `{source.resolve()}`.", ""]
    return lines


def render_metrics(documents: list[dict], run: Path, gold: Path) -> str:
    lines = ["# Metrics summary", "", f"Source run: `{run.resolve()}`. Gold: `{gold.resolve()}`.", "",
             *render_timing(documents, run),
             "## Scores", "",
             "Precision = correct / extracted; recall = correct / expected; F1 = their harmonic mean. "
             "Counts are pooled across inputs. Unlocated predictions are excluded from precision denominators "
             "and listed in the error report. Zero denominators are N/A.", "",
             "- **Extraction:** one-to-one span matches at IoU >= 0.5.",
             "- **Completeness:** matched citations with the correct `completa` value.",
             "- **Entities:** matched citations with all scored entity fields correct after normalization.",
             "- **Veracity:** correct classification and, for `real`, the correct canonical ID.",
             "- **Overall, strict:** the same citation passes all four steps, including citation type. "
             "This is not an average of stage scores or the official competition score.", "",
             "Gold-supplied or absent stages are not scored. Overall requires all four steps. "
             "A missing document within an otherwise available stage contributes missed citations. "
             "If a selected group lacks reference data, its affected score is unavailable.", ""]
    for group in ("All", "N1", "N2"):
        selected = group_documents(documents, group)
        lines += [f"### {group}", "", f"Inputs: {len(selected)}.", "",
                  "| Step | Correct / extracted / expected | Precision | Recall | F1 |",
                  "| --- | ---: | ---: | ---: | ---: |"]
        for stage, name in (*STAGE_NAMES.items(), ("overall", "Overall, strict")):
            scores = [doc["scores"][stage] for doc in selected]
            statuses = sorted({score["status"] for score in scores if score["status"] != "Scored"})
            if statuses or not scores:
                lines.append(f"| {name} | {'; '.join(statuses) if statuses else 'No inputs'} | N/A | N/A | N/A |")
                continue
            correct, predicted, expected = (sum(s[key] for s in scores) for key in ("correct", "predicted", "expected"))
            lines.append(f"| {name} | {correct} / {predicted} / {expected} | "
                         f"{percentage(correct, predicted)} | {percentage(correct, expected)} | "
                         f"{percentage(2 * correct, predicted + expected)} |")
        lines += [""]
    lines += ["## Hits for each NER", "",
              "Hits count matched citations with available entity outputs and the same citation type. "
              "Correct `null` values count as hits. Missing outputs, type errors, and unmatched spans "
              "affect the citation-level scores above; they are not field decisions. "
              "N/A means no evaluated values.", "",
              "| Field | All | N1 | N2 |", "| --- | ---: | ---: | ---: |"]
    for field in dict.fromkeys(field for fields in _CAMPOS.values() for field in fields):
        cells = []
        for group in ("All", "N1", "N2"):
            counts = Counter()
            for doc in group_documents(documents, group):
                counts.update(doc["fields"].get(field, {}))
            cells.append(f"{percentage(counts['hits'], counts['total'])} ({counts['hits']}/{counts['total']})"
                         if counts["total"] else "N/A")
        lines.append(f"| {field} | " + " | ".join(cells) + " |")
    lines += ["", "`numero_classe_tribunal` and `numero_registro_tribunal` are scored together as the set "
              "`sequencias_numericas_identificadoras`. Appeal chains are compared as sets. "
              "Only `relator_norm`, not raw `relator`, is scored.", "", "## Inputs with errors", "",
              "| Input | Citations with errors | Details |", "| --- | ---: | --- |"]
    for doc in documents:
        name = doc["document_id"]
        count = len(doc["errors"])
        link = f"[Errors](citation_errors.md#{name})" if count else "—"
        lines.append(f"| [{name}]({name}/evaluation.md) | {count} | {link} |")
    lines += ["", f"**{sum(len(doc['errors']) for doc in documents)} citation error entries.** "
              "See the [error report](citation_errors.md) for expected and extracted values.", ""]
    return "\n".join(lines)


def render_errors(documents: list[dict], run: Path, gold: Path) -> str:
    lines = ["# Citation errors", "", f"Source run: `{run.resolve()}`. Gold: `{gold.resolve()}`.", "",
             "[Metrics summary](metrics_summary.md)", "",
             "Each affected citation appears once with every observed mismatch from evaluated stages. "
             "Repeated span/type mismatches across stages are combined. `null` is a stored empty value; "
             "a missing span or payload is described explicitly. Unmatched output does not establish "
             "that a legal authority is invented.", ""]
    statuses = sorted({f"{STAGE_NAMES[stage]}: {doc['scores'][stage]['status']}"
                       for doc in documents for stage in STAGES if doc['scores'][stage]['status'] != 'Scored'})
    if statuses:
        lines += ["Unevaluated stages: " + "; ".join(statuses) + ".", ""]
    total = sum(len(doc["errors"]) for doc in documents)
    issues = sum(len(entry["issues"]) for doc in documents for entry in doc["errors"].values())
    lines += [f"**{total} citation entries; {issues} mismatches.**", "",
              "| Step | All mismatches | N1 | N2 |", "| --- | ---: | ---: | ---: |"]
    for name in STAGE_NAMES.values():
        counts = [sum(name in issue["steps"] for doc in group_documents(documents, group)
                      for entry in doc["errors"].values() for issue in entry["issues"])
                  for group in ("All", "N1", "N2")]
        lines.append(f"| {name} | " + " | ".join(map(str, counts)) + " |")
    lines += ["", "Step counts overlap when a mismatch recurs in multiple stages. "
              "Unevaluated stages have no recorded mismatches; this does not imply correctness.", ""]
    for doc in documents:
        if not doc["errors"]:
            continue
        name = doc["document_id"]
        lines += [f"## {name}", "", f"[Full report]({name}/evaluation.md) · [Input TXT]({name}/{name}.txt)", ""]
        for key, entry in sorted(doc["errors"].items(), key=lambda pair: (pair[0][0] == "missing", pair[1]["number"])):
            number = entry["number"]
            title = f"Missing expected citation {number}" if key[0] == "missing" else f"Citation {number}"
            lines += [f"### {title}", ""]
            if key[0] != "missing":
                lines += [f"[Citation details]({name}/evaluation.md#citation-{number})", ""]
            for label, record in (("Expected", entry["expected"]), ("Extracted", entry["extracted"])):
                lines += [f"**{label} citation:**", ""]
                if record is None:
                    lines += ["Unavailable — no matched citation.", ""]
                else:
                    candidate = record["candidate"]
                    lines += [quote(candidate["trecho"]), "", f"Type: `{candidate['tipo']}`. "
                              f"Offsets: `{candidate.get('inicio')}:{candidate.get('fim')}` (end exclusive).", ""]
            lines += ["| Step | Field | Expected | Output | Error |", "| --- | --- | --- | --- | --- |"]
            for issue in entry["issues"]:
                lines.append(f"| {', '.join(issue['steps'])} | `{issue['field']}` | "
                             f"{cell(issue['expected'])} | {cell(issue['output'])} | {issue['description']} |")
            lines += [""]
    if not total:
        lines += ["No citation errors found in evaluated stages.", ""]
    return "\n".join(lines)


def generate(run: Path, gold: Path, input_dir: Path, output_root: Path,
             gold_supplied: set[str] | None = None) -> Path:
    stages = {stage: load_documents(run / stage) for stage in STAGES}
    gold_stages = {stage: load_documents(gold / stage) for stage in STAGES}
    gold_supplied = set(gold_supplied or ())
    if not any(stages.values()):
        raise ValueError(f"No stage checkpoints found in {run}")
    document_ids = sorted({key for collection in (stages, gold_stages)
                           for documents in collection.values() for key in documents})
    if not gold_stages["03-entities"]:
        raise ValueError(f"No expected entity checkpoints found in {gold / '03-entities'}; pass --gold")
    for document_id in document_ids:
        if not (input_dir / f"{document_id}.txt").is_file():
            raise ValueError(f"Missing input TXT: {input_dir / (document_id + '.txt')}")
    now = datetime.now().astimezone()
    destination = output_root / now.strftime("%Y-%m-%d_%H-%M-%S_%f%z")
    destination.mkdir(parents=True, exist_ok=False)
    index = ["# Saved citation reports", "", f"Generated: {now.isoformat(timespec='seconds')}", "",
             f"Source run: `{run.resolve()}`", "", f"Gold: `{gold.resolve()}`", "",
             "[Metrics summary](metrics_summary.md) · [Citation errors](citation_errors.md)", "",
             "| Input | Citations | Report |", "| --- | ---: | --- |"]
    manifest = {"generated_at": now.isoformat(), "run": str(run.resolve()), "gold": str(gold.resolve()),
                "input_dir": str(input_dir.resolve()), "gold_supplied_stages": sorted(gold_supplied),
                "documents": {}}
    evaluations = []
    for document_id in document_ids:
        extracted = merge_stages(stages, document_id)
        expected = merge_stages(gold_stages, document_id)
        evaluations.append(evaluate_document(document_id, extracted, expected, stages, gold_stages, gold_supplied))
        folder = destination / document_id
        folder.mkdir()
        shutil.copyfile(input_dir / f"{document_id}.txt", folder / f"{document_id}.txt")
        report = render_report(document_id, extracted, expected, run.resolve(), gold.resolve(),
                               now.isoformat(timespec="seconds"), document_id in stages[STAGES[0]])
        (folder / "evaluation.md").write_text(report, encoding="utf-8")
        index.append(f"| {document_id} | {len(extracted)} | [Open report]({document_id}/evaluation.md) |")
        manifest["documents"][document_id] = {
            "citations": len(extracted),
            "sources": {stage: str(documents[document_id]["path"].resolve())
                        for stage, documents in stages.items() if document_id in documents},
        }
    (destination / "metrics_summary.md").write_text(render_metrics(evaluations, run, gold), encoding="utf-8")
    (destination / "citation_errors.md").write_text(render_errors(evaluations, run, gold), encoding="utf-8")
    (destination / "README.md").write_text("\n".join(index) + "\n", encoding="utf-8")
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="Saved pipeline run containing stage directories")
    parser.add_argument("--gold", type=Path, default=ROOT / "outputs/gold", help="Gold run with 03-entities")
    parser.add_argument("--input-dir", type=Path, default=ROOT / "desafio-jusbrasil-bracis-2026/txt")
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs/clear_outputs")
    parser.add_argument("--gold-supplied-stage", action="append", choices=STAGES, default=[],
                        help="Stage copied from gold, excluded from model scores (repeatable)")
    args = parser.parse_args()
    try:
        destination = generate(args.run, args.gold, args.input_dir, args.output_root, set(args.gold_supplied_stage))
    except (OSError, ValueError) as error:
        parser.exit(1, f"Error: {error}\n")
    print(f"Reports written to {destination}")
    print(f"Open {destination / 'README.md'}")


if __name__ == "__main__":
    main()
