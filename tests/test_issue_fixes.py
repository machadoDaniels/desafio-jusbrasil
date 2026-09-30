import asyncio
import json
import os
import tempfile
import unittest
from importlib import import_module
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

from httpx import URL, Request, Response
from openai import AsyncOpenAI, BadRequestError, LengthFinishReasonError, OpenAI
from pydantic import ValidationError

from desafio_jusbrasil.contracts import (
    CandidatoCitacao,
    CandidatoCitacaoRequest,
    ConsultaJurisprudenciaAgente,
    ExtractionConfig,
    LoteCandidatosRequest,
    PipelineConfig,
)

extractor = import_module("desafio_jusbrasil.1_extractor")
entities = import_module("desafio_jusbrasil.3_entities")
orchestrator = import_module("desafio_jusbrasil.orchestrator")


def response_for(text):
    # Repeated citations are returned once by the model and located locally.
    candidates = [
        CandidatoCitacaoRequest(trecho=phrase, tipo="lei")
        for phrase in ("art. 1 do CPC", "art. 2 do CPC")
        if phrase in text
    ]
    response = Mock()
    response.choices = [
        Mock(message=Mock(parsed=LoteCandidatosRequest(candidatos=candidates)))
    ]
    response.model_dump.return_value = {"choices": [{"finish_reason": "stop"}]}
    return response


def user_text(messages):
    return messages[-1]["content"].split("\n\n", 1)[1]


def make_client(async_mode=False, server_limit=100):
    client = Mock(spec=AsyncOpenAI if async_mode else OpenAI)
    client.base_url = URL("http://localhost/v1/")

    def tokenize(*args, body, **kwargs):
        return {
            "count": len(user_text(body["messages"])),
            "max_model_len": server_limit,
        }

    def parse(**kwargs):
        return response_for(user_text(kwargs["messages"]))

    client.post = (
        AsyncMock(side_effect=tokenize) if async_mode else Mock(side_effect=tokenize)
    )
    client.chat.completions.parse = (
        AsyncMock(side_effect=parse) if async_mode else Mock(side_effect=parse)
    )
    return client


class ExtractionChunkTest(unittest.TestCase):
    def config(self, **overrides):
        return ExtractionConfig(
            model="test",
            context_window_tokens=100,
            max_output_tokens=10,
            token_margin=0,
            chunk_overlap_chars=40,
            tokenizer_path="/tokenize",
            **overrides,
        )

    def test_chunks_preserve_unicode_offsets_repetitions_and_audits(self):
        text = "é" * 83 + " art. 1 do CPC " + "x" * 83 + " art. 1 do CPC art. 2 do CPC "
        outputs = []
        for async_mode in (False, True):
            with self.subTest(async_mode=async_mode):
                client = make_client(async_mode)
                agent = extractor.AgenteExtrator(client, self.config())
                result, audits = (
                    asyncio.run(agent.extrair_auditada_async(text))
                    if async_mode
                    else agent.extrair_auditada(text)
                )
                self.assertEqual(len(result), 3)
                self.assertEqual(len({(c.inicio, c.fim) for c in result}), 3)
                self.assertGreater(len(audits), 1)
                self.assertEqual(len(audits), client.chat.completions.parse.call_count)
                for item in result:
                    self.assertEqual(text[item.inicio : item.fim], item.trecho)
                for audit in audits:
                    chunk = audit["chunk"]
                    self.assertLessEqual(chunk["prompt_tokens"] + 10, 100)
                    self.assertEqual(
                        user_text(audit["input"]["messages"]),
                        text[chunk["start"] : chunk["end"]],
                    )
                    self.assertEqual(audit["input"]["max_completion_tokens"], 10)
                outputs.append([c.model_dump() for c in result])
        self.assertEqual(*outputs)

    def test_effective_server_limit_overrides_larger_config(self):
        client = make_client(server_limit=40)
        agent = extractor.AgenteExtrator(client, self.config())
        _, audits = agent.extrair_auditada("x" * 100)
        self.assertGreater(len(audits), 1)
        self.assertTrue(all(a["chunk"]["prompt_tokens"] + 10 <= 40 for a in audits))

    def test_byte_estimate_needs_no_tokenizer_endpoint(self):
        client = make_client()
        agent = extractor.AgenteExtrator(client, ExtractionConfig(model="test"))
        result, audits = agent.extrair_auditada("art. 1 do CPC")
        self.assertEqual(len(result), 1)
        client.post.assert_not_called()
        self.assertEqual(audits[0]["chunk"]["count_method"], "utf8_byte_estimate")

    def test_capacity_rejection_splits_and_records_failed_call(self):
        for async_mode in (False, True):
            client = make_client(async_mode)
            error = BadRequestError(
                "maximum context length exceeded",
                response=Response(400, request=Request("POST", "http://localhost")),
                body=None,
            )
            result = response_for("art. 1 do CPC")
            client.chat.completions.parse.side_effect = [error, result, result]
            agent = extractor.AgenteExtrator(client, self.config())
            text = "art. 1 do CPC " * 4
            candidates, audits = (
                asyncio.run(agent.extrair_auditada_async(text))
                if async_mode
                else agent.extrair_auditada(text)
            )
            self.assertEqual(len(candidates), 4)
            self.assertEqual(len(audits), 3)
            self.assertEqual(audits[0]["action"], "split_after_capacity_error")

    def test_output_truncation_splits_instead_of_losing_candidates(self):
        client = make_client()
        error = LengthFinishReasonError(completion=Mock())
        client.chat.completions.parse.side_effect = [
            error,
            response_for("art. 1 do CPC"),
            response_for("art. 1 do CPC"),
        ]
        agent = extractor.AgenteExtrator(client, self.config())
        candidates, audits = agent.extrair_auditada("art. 1 do CPC " * 4)
        self.assertEqual(len(candidates), 4)
        self.assertEqual(audits[0]["erro"]["tipo"], "LengthFinishReasonError")

    def test_unrelated_bad_request_is_not_retried_as_chunking(self):
        client = make_client()
        error = BadRequestError(
            "Unknown model",
            response=Response(400, request=Request("POST", "http://localhost")),
            body=None,
        )
        client.chat.completions.parse.side_effect = error
        agent = extractor.AgenteExtrator(client, self.config())
        with self.assertRaises(BadRequestError):
            agent.extrair_auditada("art. 1 do CPC")
        self.assertEqual(client.chat.completions.parse.call_count, 1)

    def test_impossible_budget_stops_before_inference(self):
        client = make_client(server_limit=5)
        agent = extractor.AgenteExtrator(client, self.config())
        with self.assertRaisesRegex(ValueError, "cannot fit"):
            agent.extrair_auditada("abc")
        client.chat.completions.parse.assert_not_called()
        with self.assertRaises(ValidationError):
            ExtractionConfig(model="test", context_window_tokens=4096)

    def test_partial_boundary_span_is_replaced_by_complete_overlap_span(self):
        full = CandidatoCitacao(trecho="art. 1 do CPC", tipo="lei", inicio=90, fim=103)
        partial = CandidatoCitacao(trecho="art. 1", tipo="lei", inicio=90, fim=96)
        result = extractor.AgenteExtrator._merge_candidates(
            [(partial, True), (full, False), (full, True)]
        )
        self.assertEqual(result, [full])

    def test_checkpoint_writes_every_chunk_audit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "input").mkdir()
            (root / "input/test.txt").write_text("art. 1 do CPC " * 12)
            agent = extractor.AgenteExtrator(make_client(), self.config())
            extractor.executar_extracao(root / "input", root / "output", agent)
            audits = sorted((root / "output/test").glob("[0-9]*.json"))
            self.assertGreater(len(audits), 1)
            result = json.loads((root / "output/test/resultado.json").read_text())
            self.assertEqual(len(result["candidatos"]), 12)


class NormalizationEntryPointTest(unittest.TestCase):
    def test_both_entry_points_load_dictionary_and_normalize_names(self):
        mapping = {
            "dias toffoli": "josé antônio dias toffoli",
            "carlos augusto amaral oliveira": "carlos augusto amaral oliveira",
            "marco antonio": "marco antônio de farias",
        }
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.dict(os.environ, {"OPENAI_API_KEY": "test"}),
        ):
            root = Path(directory)
            (root / "relatores_padronizacao.json").write_text(json.dumps(mapping))
            stage = {"model": "test"}
            config = PipelineConfig(
                input_dir=root,
                workdir=root / "run",
                database=root / "test.db",
                extractor=stage,
                completeness=stage,
                entities=stage,
            )
            config_path = root / "config.json"
            config_path.write_text(config.model_dump_json())
            with patch.object(orchestrator, "Orquestrador") as runner:
                runner.return_value.executar = AsyncMock()
                asyncio.run(orchestrator._main(config_path))
                full_agent = runner.call_args.args[2]
            with patch.object(
                entities, "executar_entities_async", new_callable=AsyncMock
            ) as run_stage:
                asyncio.run(
                    entities._executar_entities_async(config, root / "entities")
                )
                standalone_agent = run_stage.call_args.args[2]
            for agent in (full_agent, standalone_agent):
                for source, canonical in mapping.items():
                    response = Mock()
                    response.choices = [
                        Mock(
                            message=Mock(
                                parsed=ConsultaJurisprudenciaAgente(relator=source)
                            )
                        )
                    ]
                    response.model_dump.return_value = {}
                    candidate = CandidatoCitacao(trecho=source, tipo="jurisprudencia")
                    query = agent.normalize(response.choices[0].message.parsed.model_dump(), candidate)
                    self.assertEqual(query.relator_norm, canonical)


if __name__ == "__main__":
    unittest.main()
