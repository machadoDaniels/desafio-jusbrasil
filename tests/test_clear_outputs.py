"""Regression checks for reporting saved citations without inventing stage results."""

import json
from copy import deepcopy
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from clear_outputs import (
    STAGES, entity_table, evaluate_document, generate, match_records, merge_stages,
    render_errors, render_metrics, render_report,
)


def citation(start=0, text="art. 1", kind="lei"):
    return {"inicio": start, "fim": start + len(text), "tipo": kind, "trecho": text}


class ClearOutputsTests(unittest.TestCase):
    def test_join_uses_spans_instead_of_stage_order(self):
        first, second = citation(), citation(20)
        stages = {stage: {} for stage in STAGES}
        stages[STAGES[0]]["doc"] = {"items": [second, first]}
        stages[STAGES[1]]["doc"] = {"items": [
            {"candidato": first, "completude": {"completa": True}},
            {"candidato": second, "completude": {"completa": False}},
        ]}
        records = merge_stages(stages, "doc")
        self.assertEqual(len(records), 2)
        self.assertTrue(records[0]["completude"]["completa"])
        self.assertFalse(records[1]["completude"]["completa"])

    def test_unlocated_and_duplicate_spans_do_not_reuse_gold(self):
        expected = [{"candidate": citation()}]
        extracted = expected * 2 + [{"candidate": {**citation(), "inicio": None, "fim": None}}]
        matches = match_records(expected, extracted)
        self.assertEqual(matches, {0: 0})
        report = render_report("doc", extracted, expected, Path("run"), Path("gold"), "now", True)
        self.assertEqual(report.count("## Citation "), 3)
        self.assertIn("Real: **Undetermined**", report)
        self.assertNotIn("**`inventada`**", report)

    def test_incomplete_is_not_presented_as_real_or_invented(self):
        record = {"candidate": citation(), "veracidade": {"classificacao": "incompleta"}}
        report = render_report("doc", [record], [], Path("run"), Path("gold"), "now", False)
        self.assertIn("Real: **Undetermined**", report)
        self.assertIn("Invented (`inventada`): **Undetermined", report)
        self.assertNotIn("Real: **No**", report)

    def test_different_citation_types_are_reported(self):
        expected = {"candidate": citation(kind="jurisprudencia"), "campos_extraidos": {"tribunal": "STF"}}
        extracted = {"candidate": citation(), "campos_extraidos": {"numero_artigo": "1"}}
        table = "\n".join(entity_table(expected, extracted))
        self.assertIn("citation type mismatch", table)
        self.assertIn("`tribunal`", table)
        self.assertIn("`numero_artigo`", table)

    def test_generate_copies_original_input_and_supports_both_layouts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run, gold, inputs = root / "run", root / "gold", root / "inputs"
            inputs.mkdir()
            original = b"art. 1\r\nOriginal input\r\n"
            (inputs / "doc.txt").write_bytes(original)
            record = {"candidato": citation(), "completude": {"completa": True},
                      "campos_extraidos": {"numero_artigo": "1"}}
            for path in (run / "03-entities/doc/resultado.json", gold / "03-entities/doc.json"):
                path.parent.mkdir(parents=True)
                path.write_text(json.dumps({"documento_id": "doc", "candidatos": [record]}))
            destination = generate(run, gold, inputs, root / "clear_outputs")
            self.assertEqual((destination / "doc/doc.txt").read_bytes(), original)
            report = (destination / "doc/evaluation.md").read_text()
            self.assertIn("all evaluated fields match", report)
            self.assertIn("Output: Unavailable. Real: **Undetermined**", report)
            self.assertTrue((destination / "README.md").exists())
            self.assertTrue((destination / "manifest.json").exists())
            self.assertTrue((destination / "metrics_summary.md").exists())
            self.assertTrue((destination / "citation_errors.md").exists())

    def stages(self, items, document_id="gen_n2_001"):
        return {stage: {document_id: {"items": [
            item["candidato"] if stage == STAGES[0] else deepcopy(item) for item in items
        ]}} for stage in STAGES}

    def evaluate(self, stages, gold, supplied=()):
        return evaluate_document("gen_n2_001", merge_stages(stages, "gen_n2_001"),
                                 merge_stages(gold, "gen_n2_001"), stages, gold, set(supplied))

    def record(self, start=0):
        return {"candidato": citation(start), "completude": {"completa": True},
                "campos_extraidos": {"numero_artigo": "1"},
                "veracidade": {"classificacao": "real", "id_canonico": 123}}

    def test_metrics_and_errors_include_missing_extra_and_multiple_stage_errors(self):
        first, missed = self.record(), self.record(20)
        wrong = deepcopy(first)
        wrong["completude"]["completa"] = False
        wrong["campos_extraidos"]["numero_artigo"] = "2"
        wrong["veracidade"]["id_canonico"] = 456
        doc = self.evaluate(self.stages([wrong, self.record(40)]), self.stages([first, missed]))
        self.assertEqual(doc["scores"][STAGES[0]]["correct"], 1)
        for stage in (*STAGES[1:], "overall"):
            self.assertEqual(doc["scores"][stage]["correct"], 0)
            self.assertEqual(doc["scores"][stage]["predicted"], 2)
            self.assertEqual(doc["scores"][stage]["expected"], 2)
        self.assertEqual(len(doc["errors"]), 3)
        self.assertEqual(sum(len(e["issues"]) for e in doc["errors"].values()), 5)
        summary = render_metrics([doc], Path("run"), Path("gold"))
        self.assertIn("| Extraction | 1 / 2 / 2 | 50% | 50% | 50% |", summary)
        self.assertIn("| numero_artigo | 0% (0/1) | N/A | 0% (0/1) |", summary)
        self.assertNotIn("Mean metrics", summary)
        self.assertNotIn("## Extraction", summary)
        errors = render_errors([doc], Path("run"), Path("gold"))
        self.assertIn("**3 citation entries; 5 mismatches.**", errors)
        self.assertIn("Missing expected citation 2", errors)
        self.assertIn("`id_canonico`", errors)

    def test_unavailable_and_gold_supplied_stages_are_not_scored(self):
        gold = self.stages([self.record()])
        stages = deepcopy(gold)
        stages[STAGES[0]] = {}
        doc = self.evaluate(stages, gold, [STAGES[1]])
        self.assertEqual(doc["scores"][STAGES[0]]["status"], "Unavailable")
        self.assertEqual(doc["scores"][STAGES[1]]["status"], "Gold supplied")
        self.assertEqual(doc["scores"]["overall"]["status"], "Requires all four steps")
        self.assertEqual(doc["scores"][STAGES[2]]["correct"], 1)
        self.assertFalse(doc["errors"])

    def test_timing_is_first_and_preserves_measured_scope(self):
        gold = self.stages([self.record()])
        stages = deepcopy(gold)
        stages[STAGES[0]] = {}
        doc = self.evaluate(stages, gold, [STAGES[1]])
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            metadata = {"elapsed_seconds": 56.107, "entity_seconds": 54.733,
                        "veracity_and_materialization_seconds": 1.374,
                        "completeness_seconds": 99, "status": "complete"}
            (run / "experiment.json").write_text(json.dumps(metadata))
            summary = render_metrics([doc], run, Path("gold"))
            self.assertEqual(next(line for line in summary.splitlines() if line.startswith('|')),
                             "| Step | Elapsed time | Notes |")
            self.assertIn("| Overall | 56.11 s |", summary)
            self.assertIn("| Entities | 54.73 s |", summary)
            self.assertIn("| Veracity | 1.37 s | Includes prediction export", summary)
            self.assertIn("| Extraction | N/A |", summary)
            self.assertIn("| Completeness | N/A | Gold supplied", summary)
            self.assertNotIn("99.00 s", summary)
            # A separately recorded verification duration takes precedence.
            metadata["veracity_seconds"] = 0
            (run / "experiment.json").write_text(json.dumps(metadata))
            self.assertIn("| Veracity | 0.00 s | Recorded stage wall time.",
                          render_metrics([doc], run, Path("gold")))

    def test_absent_or_invalid_timing_is_unavailable_not_zero(self):
        doc = self.evaluate(self.stages([self.record()]), self.stages([self.record()]))
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            summary = render_metrics([doc], run, Path("gold"))
            self.assertIn("| Overall | N/A | Total duration was not recorded.", summary)
            self.assertIn("| Entities | N/A | Stage duration was not recorded.", summary)
            (run / "experiment.json").write_text(json.dumps({
                "elapsed_seconds": -1, "entity_seconds": True, "veracity_seconds": "unknown"}))
            summary = render_metrics([doc], run, Path("gold"))
            self.assertIn("| Overall | N/A |", summary)
            self.assertIn("| Entities | N/A |", summary)
            self.assertIn("| Veracity | N/A |", summary)

    def test_stage_scores_do_not_borrow_downstream_values(self):
        gold = self.stages([self.record()])
        stages = deepcopy(gold)
        stages[STAGES[1]]["gen_n2_001"]["items"][0]["completude"]["completa"] = False
        # Later checkpoints still contain the correct value.
        doc = self.evaluate(stages, gold)
        self.assertEqual(doc["scores"][STAGES[1]]["correct"], 0)
        self.assertEqual(doc["scores"][STAGES[2]]["correct"], 1)
        self.assertEqual(doc["scores"][STAGES[3]]["correct"], 1)
        self.assertEqual(doc["scores"]["overall"]["correct"], 0)

    def test_type_mismatch_fails_strict_overall_but_not_span_matching(self):
        gold = self.stages([self.record()])
        record = self.record()
        record["candidato"]["tipo"] = "jurisprudencia"
        record["campos_extraidos"] = {"tribunal": "STF"}
        doc = self.evaluate(self.stages([record]), gold)
        self.assertEqual(doc["scores"][STAGES[0]]["correct"], 1)
        self.assertEqual(doc["scores"][STAGES[2]]["correct"], 0)
        self.assertEqual(doc["scores"]["overall"]["correct"], 0)
        self.assertEqual(len(doc["errors"]), 1)
        self.assertFalse(doc["fields"])

    def test_identifier_sets_and_appeal_chains_use_evaluator_normalization(self):
        record = self.record()
        record["candidato"]["tipo"] = "jurisprudencia"
        record["campos_extraidos"] = {"numero_classe_tribunal": "123",
                                      "numero_registro_tribunal": "456", "cadeia_recursal": ["AgInt", "REsp"]}
        gold = self.stages([record])
        record["campos_extraidos"].update(numero_classe_tribunal="456", numero_registro_tribunal="123",
                                          cadeia_recursal=["REsp", "AgInt", "AgInt"])
        doc = self.evaluate(self.stages([record]), gold)
        self.assertEqual(doc["scores"]["overall"]["correct"], 1)
        self.assertEqual(doc["fields"]["sequencias_numericas_identificadoras"]["hits"], 1)
        self.assertFalse(doc["errors"])

    def test_gold_only_documents_are_reported_as_missed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run, gold, inputs = root / "run", root / "gold", root / "inputs"
            inputs.mkdir()
            for name in ("gen_n1_001", "gen_n2_001"):
                (inputs / f"{name}.txt").write_text("art. 1")
                sources = (run, gold) if name == "gen_n1_001" else (gold,)
                for source in sources:
                    folder = source / STAGES[2]
                    folder.mkdir(parents=True, exist_ok=True)
                    (folder / f"{name}.json").write_text(json.dumps({
                        "documento_id": name, "candidatos": [self.record()]}))
            destination = generate(run, gold, inputs, root / "clear")
            summary = (destination / "metrics_summary.md").read_text()
            self.assertIn("| Entities | 1 / 1 / 2 | 100% | 50% | 66.67% |", summary)
            self.assertTrue((destination / "gen_n2_001/evaluation.md").is_file())
            self.assertIn("Missing expected citation 1", (destination / "citation_errors.md").read_text())


if __name__ == "__main__":
    unittest.main()
