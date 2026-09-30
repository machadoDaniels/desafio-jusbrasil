"""Measure one focused extractor against fixed gold and separate synthetic cases.

Each invocation saves its prompt, source, audits and scores in a new directory.
This is a development tool: cases used for prompt revision are not held-out tests.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
from collections import Counter
from datetime import UTC, datetime
from importlib import import_module
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv
from openai import AsyncOpenAI

from evaluate_entities import _valor_campo, _valor_comparavel
from desafio_jusbrasil.contracts import CandidatoCitacao, DocumentoEntidades, PipelineConfig
from desafio_jusbrasil.entity_extraction import EntityExtractor, EntityExtractionError
from desafio_jusbrasil.utils import listar_resultados

GROUPS = {
    "natureza": ("jurisprudencia", ("natureza", "numero_sumula", "sumula_vinculante")),
    "numero_processo_cnj": ("jurisprudencia", ("numero_processo_cnj", "sequencias_numericas_identificadoras")),
    "classe_processual": ("jurisprudencia", ("classe_processual", "cadeia_recursal")),
    "tribunal": ("jurisprudencia", ("tribunal",)),
    "uf": ("jurisprudencia", ("uf",)),
    "ano": ("jurisprudencia", ("ano",)),
    "relator": ("jurisprudencia", ("relator_norm",)),
    "diploma": ("lei", ("diploma", "numero_artigo")),
    "numero_diploma": ("lei", ("numero_diploma",)),
}
DEFAULT_GOLD = ROOT / "outputs/entities-score-gemma4-12b-20260928/generated-gold/03-entities"
DEFAULT_CONFIG = ROOT / "outputs/parallel-entities-20260928T205541Z/config.yaml"


def summary(results: list[dict]) -> dict:
    counters: dict[str, Counter] = {}
    for result in results:
        for field, expected in result["expected"].items():
            values = counters.setdefault(field, Counter())
            predicted = result["predicted"].get(field)
            equal = not result.get("error") and _valor_comparavel(field, expected) == _valor_comparavel(field, predicted)
            values["total"] += 1
            values["correct"] += equal
            values["expected_values"] += expected is not None
            values["predicted_values"] += predicted is not None
            values["correct_values"] += equal and expected is not None
    per_field = {}
    for field, values in counters.items():
        per_field[field] = dict(values)
        per_field[field].update(
            accuracy=values["correct"] / values["total"],
            precision=values["correct_values"] / values["predicted_values"] if values["predicted_values"] else None,
            recall=values["correct_values"] / values["expected_values"] if values["expected_values"] else None,
        )
    correct = sum(result["correct"] for result in results)
    return {"total": len(results), "correct": correct,
            "accuracy": correct / len(results) if results else None,
            "errors": sum(bool(result.get("error")) for result in results),
            "fields": per_field,
            "mismatches": [{key: value for key, value in result.items() if key != "audits"}
                           for result in results if not result["correct"]]}


async def evaluate(args) -> dict:
    module = import_module(f"desafio_jusbrasil.entity_extraction.{args.extractor}_extraction")
    kind, scoring_fields = GROUPS[args.extractor]
    config = PipelineConfig.from_yaml(args.config)
    config.entities.max_concurrency = 2
    config.entities.temperature = 0.0
    config.entities.max_retries = 2
    cases = []
    for path in listar_resultados(args.gold):
        doc = DocumentoEntidades.model_validate_json(path.read_text())
        for position, item in enumerate(doc.candidatos):
            if item.candidato.tipo.value == kind and item.campos_extraidos is not None:
                cases.append({"set": "gold", "document": doc.documento_id, "position": position,
                              "candidate": item.candidato,
                              "expected": {field: _valor_campo(item.campos_extraidos, field) for field in scoring_fields}})
    synthetic_path = ROOT / f"tests/fixtures/entity_prompts/{args.extractor}.json"
    if synthetic_path.exists():
        for position, example in enumerate(json.loads(synthetic_path.read_text())):
            cases.append({"set": "synthetic", "document": "synthetic", "position": position,
                          "candidate": CandidatoCitacao(trecho=example["text"], tipo=kind),
                          "expected": example["expected"]})
    if args.synthetic_only:
        cases = [case for case in cases if case["set"] == "synthetic"]
    if not cases:
        raise ValueError("No evaluation cases found")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "cases").mkdir()
    source = Path(module.__file__).read_text()
    (args.output_dir / "extractor.py").write_text(source)
    (args.output_dir / "prompt.txt").write_text(module.PROMPT)
    metadata = {"extractor": args.extractor, "started_at": datetime.now(UTC).isoformat(),
                "model": config.entities.model, "temperature": 0, "concurrency": 2,
                "source_sha256": hashlib.sha256(source.encode()).hexdigest(), "status": "running"}
    (args.output_dir / "manifest.json").write_text(json.dumps(metadata, indent=2))
    started = perf_counter()
    async with AsyncOpenAI(base_url=config.entities.base_url, max_retries=0) as client:
        coordinator = EntityExtractor.from_config(client, config)

        async def run_case(index: int, case: dict) -> dict:
            candidate = case["candidate"]
            result = {**case, "candidate": candidate.model_dump(mode="json")}
            try:
                fields, audits = await module.extract(candidate, coordinator.runner)
                if case["set"] == "gold":
                    normalized = coordinator.normalize(fields, candidate)
                    predicted = {field: _valor_campo(normalized, field) for field in scoring_fields}
                else:
                    predicted = fields
                result.update(raw_fields=fields, predicted=predicted, audits=audits)
                result["correct"] = all(
                    _valor_comparavel(field, value) == _valor_comparavel(field, predicted.get(field))
                    for field, value in case["expected"].items()
                )
            except EntityExtractionError as error:
                result.update(predicted={}, correct=False, error=str(error), audits=error.auditorias)
            (args.output_dir / "cases" / f"{index:04d}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
            return result

        results = await asyncio.gather(*(run_case(index, case) for index, case in enumerate(cases)))
    report = {name: summary([result for result in results if result["set"] == name])
              for name in ("gold", "synthetic")}
    metadata.update(status="complete", seconds=perf_counter() - started, finished_at=datetime.now(UTC).isoformat())
    (args.output_dir / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (args.output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return {name: {key: value for key, value in score.items() if key != "mismatches"} for name, score in report.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("extractor", choices=GROUPS)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--synthetic-only", action="store_true")
    args = parser.parse_args()
    load_dotenv()
    print(json.dumps(asyncio.run(evaluate(args)), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
