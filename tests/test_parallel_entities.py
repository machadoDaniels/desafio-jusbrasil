"""Concurrency, routing, failure, and normalization checks for focused extraction."""

import asyncio
import json
import tempfile
import unittest
from collections import Counter
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from desafio_jusbrasil.contracts import CandidatoCitacao, DocumentoEntidades, StageConfig
from desafio_jusbrasil.entity_extraction import EntityExtractor, EntityExtractionError
from desafio_jusbrasil.entity_extraction.shared import MAX_COMPLETION_TOKENS


class FakeClient:
    def __init__(self, fail=None, permanent=False):
        self.active = 0
        self.peak = 0
        self.requests = []
        self.counts = Counter()
        self.fail = fail
        self.permanent = permanent
        self.chat = SimpleNamespace(completions=SimpleNamespace(parse=self.parse))

    async def parse(self, **request):
        schema = request['response_format']
        name = schema.__name__
        self.requests.append(request)
        self.counts[name] += 1
        self.active += 1
        self.peak = max(self.peak, self.active)
        try:
            await asyncio.sleep(0.001)
            if name == self.fail and (self.permanent or self.counts[name] == 1):
                raise ValueError('Simulated invalid structured response')
            values = {
                'NaturezaFields': {'natureza': 'acordao'},
                'ProcessNumberFields': {'numero_processo_cnj': '12345672020241010001'},
                'ProcessClassFields': {'classe_processual': 'Rcl — Reclamação', 'cadeia_recursal': ['Rcl — Reclamação']},
                'TribunalFields': {'tribunal': 'STF'},
                'UFFields': {'uf': 'DF'},
                'YearFields': {'ano': 2024},
                'RelatorFields': {'relator': 'Judge Name'},
                'DiplomaFields': {'numero_artigo': '77', 'diploma': 'Código Civil'},
                'LawNumberFields': {'numero_diploma': None},
            }
            parsed = schema(**values[name])
            response = Mock()
            response.choices = [SimpleNamespace(message=SimpleNamespace(parsed=parsed))]
            response.model_dump.return_value = {'choices': [{'message': {'content': parsed.model_dump_json()}}]}
            return response
        finally:
            self.active -= 1


class ParallelEntitiesTests(unittest.IsolatedAsyncioTestCase):
    def config(self, **overrides):
        return StageConfig(model='test-model', max_concurrency=3, max_retries=2, **overrides)

    def candidate(self, kind='jurisprudencia'):
        return CandidatoCitacao(trecho='Rcl 1234567-20.2024.1.01.0001/DF', tipo=kind)

    async def test_global_limit_across_citations_and_group_merge(self):
        client = FakeClient()
        agent = EntityExtractor(client, self.config(), {'Judge Name': 'judge canonical'})
        results = await asyncio.gather(*(agent.extrair_auditada_async(self.candidate()) for _ in range(4)))
        self.assertEqual(client.peak, 3)
        self.assertEqual(len(client.requests), 28)
        for result, audits in results:
            self.assertEqual(result.classe_processual, 'Rcl')
            self.assertEqual(result.cadeia_recursal, ['Rcl'])
            self.assertEqual(result.relator_norm, 'judge canonical')
            self.assertEqual(result.numero_processo_cnj, '12345672020241010001')
            self.assertIsNone(result.ano)  # Existing CNJ normalization policy.
            self.assertEqual(len(audits), 7)
            self.assertEqual(len({a['extractor'] for a in audits}), 7)
            self.assertTrue(all(a['duration_seconds'] >= 0 for a in audits))
            self.assertTrue(all(a['candidate']['trecho'] == self.candidate().trecho for a in audits))
        for request in client.requests:
            self.assertEqual(request['max_completion_tokens'], MAX_COMPLETION_TOKENS)
            # Grammar constraints alone need not expose field definitions to the model.
            schema = request['response_format'].model_json_schema()
            self.assertIn(json.dumps(schema, ensure_ascii=False), request['messages'][0]['content'])
            self.assertEqual(request['messages'][1]['content'], 'Original citation:\n' + self.candidate().trecho)

    async def test_law_routes_only_two_extractors(self):
        client = FakeClient()
        result, audits = await EntityExtractor(client, self.config()).extrair_auditada_async(self.candidate('lei'))
        self.assertEqual(set(client.counts), {'DiplomaFields', 'LawNumberFields'})
        self.assertEqual(len(audits), 2)
        self.assertEqual(result.numero_artigo, '77')
        self.assertIsNone(result.numero_diploma)

    async def test_retry_only_failed_group(self):
        client = FakeClient(fail='TribunalFields')
        agent = EntityExtractor(client, self.config())
        _, audits = await agent.extrair_auditada_async(self.candidate())
        self.assertEqual(len(audits), 8)
        self.assertEqual(client.counts['TribunalFields'], 2)
        self.assertTrue(all(count == 1 for name, count in client.counts.items() if name != 'TribunalFields'))
        self.assertEqual(sum('erro' in audit for audit in audits), 1)

    async def test_failure_keeps_successful_group_audits_without_partial_entity(self):
        client = FakeClient(fail='YearFields', permanent=True)
        agent = EntityExtractor(client, StageConfig(model='test', max_retries=1))
        with self.assertRaises(EntityExtractionError) as raised:
            await agent.extrair_auditada_async(self.candidate())
        self.assertEqual(len(raised.exception.auditorias), 7)
        self.assertEqual(sum('erro' in audit for audit in raised.exception.auditorias), 1)
        self.assertEqual(client.active, 0)

    async def test_failure_persists_audits_in_stage_checkpoint(self):
        stage = import_module('desafio_jusbrasil.3_entities')
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / 'input'
            source.mkdir()
            candidate = self.candidate()
            (source / 'doc.json').write_text(json.dumps({'documento_id': 'doc', 'candidatos': [
                {'candidato': candidate.model_dump(mode='json'), 'completude': {'completa': True}},
            ]}))
            client = FakeClient(fail='YearFields', permanent=True)
            agent = EntityExtractor(client, StageConfig(model='test', max_retries=1))
            with self.assertRaises(EntityExtractionError):
                await stage.executar_entities_async(source, root / 'out', agent, 3)
            checkpoint = DocumentoEntidades.model_validate_json((root / 'out/doc/resultado.json').read_text())
            self.assertEqual(checkpoint.candidatos, [])
            self.assertEqual(len(list((root / 'out/doc').glob('[0-9]*.json'))), 7)

    async def test_sumula_normalization_does_not_infer_court(self):
        agent = EntityExtractor(FakeClient(), self.config())
        candidate = CandidatoCitacao(trecho='Súmula Vinculante 99', tipo='jurisprudencia')
        result = agent.normalize({'numero_sumula': 99, 'numero_classe_tribunal': '99'}, candidate)
        self.assertEqual(result.natureza, 'sumula')
        self.assertTrue(result.sumula_vinculante)
        self.assertIsNone(result.numero_classe_tribunal)
        self.assertIsNone(result.tribunal)


if __name__ == '__main__':
    unittest.main()
