import asyncio
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from pydantic import ValidationError

from desafio_jusbrasil.completeness import (
    AgenteCompletude,
    executar_completude,
    executar_completude_async,
)
from desafio_jusbrasil.contracts import (
    AuditoriaChamadaModelo,
    CandidatoAnalisado,
    CandidatoCitacao,
    CandidatoCitacaoRequest,
    Classificacao,
    ConsultaJurisprudencia,
    ConsultaLegislacao,
    DocumentoCompletude,
    DocumentoExtraido,
    DocumentoPredito,
    ResultadoCompletude,
    ResultadoVeracidade,
    StageConfig,
    TipoCitacao,
    escrever_manifesto_etapa,
)
from desafio_jusbrasil.extractor import (
    AgenteExtrator,
    executar_extracao,
    executar_extracao_async,
)
from desafio_jusbrasil.orchestrator import Orquestrador
from desafio_jusbrasil.veracity import (
    VerificadorVeracidade,
    _prompt_veracidade,
    executar_veracidade,
    executar_veracidade_async,
)


def _auditoria_fake() -> AuditoriaChamadaModelo:
    return AuditoriaChamadaModelo(input={"messages": []}, output={"choices": []})


class ExtratorFake:
    def extrair_auditada(
        self,
        texto: str,
    ) -> tuple[list[CandidatoCitacao], AuditoriaChamadaModelo]:
        trecho = "art. 373 do CPC"
        inicio = texto.index(trecho)
        return [
            CandidatoCitacao(
                tipo=TipoCitacao.LEI,
                trecho=trecho,
                inicio=inicio,
                fim=inicio + len(trecho),
            )
        ], _auditoria_fake()


class ExtratorAsyncFake:
    async def extrair_auditada_async(
        self,
        texto: str,
    ) -> tuple[list[CandidatoCitacao], AuditoriaChamadaModelo]:
        return ExtratorFake().extrair_auditada(texto)


class CompletudeFake:
    def classificar_auditada(
        self,
        candidato: CandidatoCitacao,
        contexto: str,
    ) -> tuple[ResultadoCompletude, AuditoriaChamadaModelo]:
        return ResultadoCompletude(completa=True), _auditoria_fake()


class CompletudeAsyncFake:
    async def classificar_auditada_async(
        self,
        candidato: CandidatoCitacao,
        contexto: str,
    ) -> tuple[ResultadoCompletude, AuditoriaChamadaModelo]:
        return CompletudeFake().classificar_auditada(candidato, contexto)


class CompletudeAsyncComFalha(CompletudeAsyncFake):
    async def classificar_auditada_async(
        self,
        candidato: CandidatoCitacao,
        contexto: str,
    ) -> tuple[ResultadoCompletude, AuditoriaChamadaModelo]:
        if candidato.trecho == "falha":
            await asyncio.sleep(0.01)
            raise RuntimeError("falha simulada")
        return await super().classificar_auditada_async(candidato, contexto)


class VeracidadeFake:
    def classificar_auditada(
        self,
        candidato: CandidatoCitacao,
    ) -> tuple[ResultadoVeracidade, AuditoriaChamadaModelo]:
        return ResultadoVeracidade(
            classificacao=Classificacao.REAL,
            id_canonico=28893055,
            justificativa="registro único",
        ), _auditoria_fake()


class VeracidadeAsyncFake:
    async def classificar_auditada_async(
        self,
        candidato: CandidatoCitacao,
    ) -> tuple[ResultadoVeracidade, AuditoriaChamadaModelo]:
        return VeracidadeFake().classificar_auditada(candidato)


class VeracidadeAsyncComFalha(VeracidadeAsyncFake):
    async def classificar_auditada_async(
        self,
        candidato: CandidatoCitacao,
    ) -> tuple[ResultadoVeracidade, AuditoriaChamadaModelo]:
        if candidato.trecho == "falha":
            await asyncio.sleep(0.01)
            raise RuntimeError("falha simulada")
        return await super().classificar_auditada_async(candidato)


class PipelineTest(unittest.TestCase):
    def test_llm_nao_recebe_campos_de_span(self) -> None:
        self.assertNotIn("inicio", CandidatoCitacaoRequest.model_fields)
        self.assertNotIn("fim", CandidatoCitacaoRequest.model_fields)

    def test_spans_sao_calculados_localmente(self) -> None:
        texto = "art. 373 do CPC e art. 373 do CPC"
        candidatos = AgenteExtrator._adicionar_spans(
            texto,
            [
                CandidatoCitacaoRequest(
                    trecho="art. 373 do CPC",
                    tipo=TipoCitacao.LEI,
                )
            ],
        )
        self.assertEqual(
            [(item.inicio, item.fim) for item in candidatos], [(0, 15), (18, 33)]
        )

    def test_spans_aceitam_normalizacao_de_whitespace(self) -> None:
        texto = (
            "Como reconhecido no julgado do STM proferido em 2023\n"
            "pela relatoria de CARLOS AUGUSTO AMARAL OLIVEIRA e na APL nº\n"
            "7000449-40.2023.7.00.0000/RS."
        )
        trechos = [
            (
                "julgado do STM proferido em 2023 pela relatoria de "
                "CARLOS AUGUSTO AMARAL OLIVEIRA"
            ),
            "APL nº 7000449-40.2023.7.00.0000/RS",
        ]
        candidatos = AgenteExtrator._adicionar_spans(
            texto,
            [
                CandidatoCitacaoRequest(
                    trecho=trecho,
                    tipo=TipoCitacao.JURISPRUDENCIA,
                )
                for trecho in trechos
            ],
        )
        self.assertEqual(len(candidatos), 2)
        for candidato in candidatos:
            assert candidato.inicio is not None
            assert candidato.fim is not None
            self.assertEqual(
                candidato.trecho,
                texto[candidato.inicio : candidato.fim],
            )
            self.assertIn("\n", candidato.trecho)

    def test_candidato_sem_trecho_literal_persiste_offsets_nulos(self) -> None:
        candidatos = AgenteExtrator._adicionar_spans(
            "Conforme o art. 373 do CPC.",
            [
                CandidatoCitacaoRequest(
                    trecho="art. 373 do Código de Processo Civil",
                    tipo=TipoCitacao.LEI,
                )
            ],
        )
        self.assertEqual(len(candidatos), 1)
        self.assertIsNone(candidatos[0].inicio)
        self.assertIsNone(candidatos[0].fim)
        self.assertIn('"inicio":null', candidatos[0].model_dump_json())

    def test_extrator_so_preserva_offsets_nulos_em_debug(self) -> None:
        pedido = CandidatoCitacaoRequest(
            trecho="trecho inexistente",
            tipo=TipoCitacao.JURISPRUDENCIA,
        )
        sem_debug = AgenteExtrator(  # type: ignore[arg-type]
            object(), StageConfig(model="modelo")
        )
        com_debug = AgenteExtrator(  # type: ignore[arg-type]
            object(), StageConfig(model="modelo", debug=True)
        )
        self.assertEqual(sem_debug._processar_candidatos("documento", [pedido]), [])
        self.assertEqual(len(com_debug._processar_candidatos("documento", [pedido])), 1)

    def test_checkpoint_inclui_input_e_output_do_modelo(self) -> None:
        documento = DocumentoExtraido(
            documento_id="doc",
            candidatos=[],
            texto="texto",
            chamadas_modelo=[
                AuditoriaChamadaModelo(
                    input={"messages": [{"role": "user", "content": "texto"}]},
                    output={"choices": []},
                )
            ],
        )
        serializado = documento.model_dump(mode="json")
        self.assertEqual(
            serializado["chamadas_modelo"][0],
            {
                "input": {"messages": [{"role": "user", "content": "texto"}]},
                "output": {"choices": []},
            },
        )

    def test_manifesto_registra_configuracao_da_etapa(self) -> None:
        with tempfile.TemporaryDirectory() as temporario:
            destino = Path(temporario)
            config = StageConfig(
                model="modelo",
                base_url="http://localhost:8000/v1",
                temperature=None,
                top_p=0.9,
                top_k=20,
                async_requests=True,
                max_concurrency=3,
            )
            escrever_manifesto_etapa(destino, "extractor", config)
            manifesto = json.loads((destino / "manifest.json").read_text())
            self.assertEqual(manifesto["etapa"], "extractor")
            self.assertEqual(
                manifesto["configuracao_modelo"], config.model_dump(mode="json")
            )

    def test_completude_nao_recebe_metadados_da_extracao(self) -> None:
        candidato = CandidatoCitacao(
            trecho="art. 373 do CPC",
            tipo=TipoCitacao.LEI,
            confianca_extracao=0.75,
            inicio=0,
            fim=15,
        )
        mensagens = AgenteCompletude._mensagens(candidato, candidato.trecho)
        conteudo = mensagens[1]["content"]
        self.assertNotIn("confianca_extracao", conteudo)
        self.assertNotIn('"inicio"', conteudo)
        self.assertNotIn('"fim"', conteudo)

    def test_completude_usa_prompt_por_tipo_e_retorna_so_booleano(self) -> None:
        lei = CandidatoCitacao(trecho="art. 1 do CPC", tipo=TipoCitacao.LEI)
        jurisprudencia = CandidatoCitacao(
            trecho="RE 123/SP",
            tipo=TipoCitacao.JURISPRUDENCIA,
        )
        prompt_lei = AgenteCompletude._mensagens(lei, lei.trecho)[0]["content"]
        prompt_jurisprudencia = AgenteCompletude._mensagens(
            jurisprudencia, jurisprudencia.trecho
        )[0]["content"]
        self.assertNotEqual(prompt_lei, prompt_jurisprudencia)
        self.assertEqual(set(ResultadoCompletude.model_fields), {"completa"})

    def test_veracidade_recebe_trecho_e_tipo_do_extractor(self) -> None:
        candidato = CandidatoCitacao(
            trecho="Súmula 331 do TST",
            tipo=TipoCitacao.JURISPRUDENCIA,
        )
        mensagens = VerificadorVeracidade._mensagens(candidato)
        self.assertIn(candidato.trecho, mensagens[1]["content"])
        self.assertNotEqual(
            _prompt_veracidade(TipoCitacao.JURISPRUDENCIA),
            _prompt_veracidade(TipoCitacao.LEI),
        )
        self.assertEqual(
            set(ConsultaJurisprudencia.model_fields),
            {"valores_fts", "natureza", "tribunal", "ano", "relator"},
        )
        self.assertEqual(set(ConsultaLegislacao.model_fields), {"valores_fts"})

    def test_veracidade_monta_uma_consulta_fts_parametrizada(self) -> None:
        with tempfile.TemporaryDirectory() as temporario:
            banco = Path(temporario) / "base.db"
            with sqlite3.connect(banco) as conexao:
                conexao.executescript(
                    """
                    CREATE TABLE documentos (
                        documento_id TEXT PRIMARY KEY,
                        id INTEGER NOT NULL UNIQUE,
                        tribunal TEXT,
                        ano INTEGER,
                        relator TEXT,
                        natureza TEXT NOT NULL,
                        tipo TEXT NOT NULL,
                        texto TEXT NOT NULL,
                        texto_len INTEGER NOT NULL
                    );
                    CREATE VIRTUAL TABLE documentos_fts USING fts5(
                        texto, content='documentos', content_rowid='rowid'
                    );
                    INSERT INTO documentos VALUES (
                        'doc_1', 42, 'STJ', 2023, 'Maria Silva', 'acordao',
                        'jurisprudencia', 'Agravo em Recurso Especial 1 996 496 RJ', 44
                    );
                    INSERT INTO documentos_fts(documentos_fts) VALUES ('rebuild');
                    """
                )

            verificador = object.__new__(VerificadorVeracidade)
            verificador._database = banco
            consulta = ConsultaJurisprudencia(
                valores_fts=["1 996 496", "RJ"],
                natureza="acordao",
                tribunal="STJ",
                ano=2023,
                relator="Maria Silva",
            )
            registros, sql, parametros = verificador._consultar_base(consulta)
            self.assertEqual(registros, [{"id": 42, "documento_id": "doc_1"}])
            self.assertEqual(parametros[0], '"1 996 496" AND "RJ"')
            self.assertIn("d.tribunal = ?", sql)
            self.assertEqual(
                verificador._classificar(registros).classificacao,
                Classificacao.REAL,
            )

    def test_real_exige_id_canonico(self) -> None:
        with self.assertRaises(ValidationError):
            ResultadoVeracidade(
                classificacao=Classificacao.REAL,
                justificativa="sem id",
            )

    def test_etapas_podem_ser_executadas_por_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temporario:
            raiz = Path(temporario)
            entrada = raiz / "txt"
            entrada.mkdir()
            texto = "Conforme art. 373 do CPC, cabe à parte provar o fato."
            (entrada / "doc_001.txt").write_text(texto, encoding="utf-8")

            extracao = raiz / "01-extraction"
            completude = raiz / "02-completeness"
            veracidade = raiz / "03-veracity"

            executar_extracao(entrada, extracao, ExtratorFake())
            (extracao / "manifest.json").write_text("{}", encoding="utf-8")
            executar_completude(extracao, completude, CompletudeFake())
            (completude / "manifest.json").write_text("{}", encoding="utf-8")
            executar_veracidade(completude, veracidade, VeracidadeFake())

            resultado = json.loads(
                (veracidade / "doc_001.json").read_text(encoding="utf-8")
            )
            self.assertEqual(resultado["documento_id"], "doc_001")
            self.assertEqual(
                resultado["candidatos"][0]["veracidade"]["classificacao"],
                "real",
            )
            self.assertEqual(len(resultado["chamadas_modelo"]), 1)

    def test_extracao_async_preserva_ordem_dos_documentos(self) -> None:
        with tempfile.TemporaryDirectory() as temporario:
            raiz = Path(temporario)
            entrada = raiz / "txt"
            entrada.mkdir()
            for nome in ("b", "a"):
                (entrada / f"{nome}.txt").write_text(
                    "Conforme art. 373 do CPC.",
                    encoding="utf-8",
                )

            destino = raiz / "01-extraction"
            asyncio.run(
                executar_extracao_async(
                    entrada,
                    destino,
                    ExtratorAsyncFake(),
                    max_concurrency=2,
                )
            )

            ids = [
                json.loads(arquivo.read_text())["documento_id"]
                for arquivo in sorted(destino.glob("*.json"))
            ]
            self.assertEqual(ids, ["a", "b"])

    def test_completude_async_salva_documento_antes_de_falha_posterior(self) -> None:
        with tempfile.TemporaryDirectory() as temporario:
            raiz = Path(temporario)
            entrada = raiz / "entrada"
            saida = raiz / "saida"
            entrada.mkdir()
            for documento_id, texto in (("a", "art. 373 do CPC"), ("b", "falha")):
                documento = DocumentoExtraido(
                    documento_id=documento_id,
                    texto=texto,
                    candidatos=[
                        CandidatoCitacao(
                            trecho=texto,
                            tipo=TipoCitacao.LEI,
                            inicio=0,
                            fim=len(texto),
                        )
                    ],
                )
                (entrada / f"{documento_id}.json").write_text(
                    documento.model_dump_json(),
                    encoding="utf-8",
                )

            with self.assertRaisesRegex(RuntimeError, "falha simulada"):
                asyncio.run(
                    executar_completude_async(
                        entrada,
                        saida,
                        CompletudeAsyncComFalha(),
                        max_concurrency=2,
                    )
                )

            self.assertTrue((saida / "a.json").exists())
            self.assertFalse((saida / "b.json").exists())

    def test_veracidade_async_salva_documento_antes_de_falha_posterior(self) -> None:
        with tempfile.TemporaryDirectory() as temporario:
            raiz = Path(temporario)
            entrada = raiz / "entrada"
            saida = raiz / "saida"
            entrada.mkdir()
            for documento_id, texto in (("a", "art. 373 do CPC"), ("b", "falha")):
                candidato = CandidatoCitacao(
                    trecho=texto,
                    tipo=TipoCitacao.LEI,
                    inicio=0,
                    fim=len(texto),
                )
                documento = DocumentoCompletude(
                    documento_id=documento_id,
                    texto=texto,
                    candidatos=[
                        CandidatoAnalisado(
                            candidato=candidato,
                            completude=ResultadoCompletude(completa=True),
                        )
                    ],
                )
                (entrada / f"{documento_id}.json").write_text(
                    documento.model_dump_json(),
                    encoding="utf-8",
                )

            with self.assertRaisesRegex(RuntimeError, "falha simulada"):
                asyncio.run(
                    executar_veracidade_async(
                        entrada,
                        saida,
                        VeracidadeAsyncComFalha(),
                        max_concurrency=2,
                    )
                )

            self.assertTrue((saida / "a.json").exists())
            self.assertFalse((saida / "b.json").exists())

    def test_completude_e_veracidade_async(self) -> None:
        with tempfile.TemporaryDirectory() as temporario:
            raiz = Path(temporario)
            entrada = raiz / "txt"
            entrada.mkdir()
            (entrada / "doc.txt").write_text(
                "Conforme art. 373 do CPC.",
                encoding="utf-8",
            )
            extracao = raiz / "01"
            completude = raiz / "02"
            veracidade = raiz / "03"
            executar_extracao(entrada, extracao, ExtratorFake())
            asyncio.run(
                executar_completude_async(
                    extracao,
                    completude,
                    CompletudeAsyncFake(),
                    max_concurrency=2,
                )
            )
            asyncio.run(
                executar_veracidade_async(
                    completude,
                    veracidade,
                    VeracidadeAsyncFake(),
                    max_concurrency=2,
                )
            )
            resultado = json.loads(
                (veracidade / "doc.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                resultado["candidatos"][0]["veracidade"]["classificacao"],
                "real",
            )

    def test_orquestrador_materializa_contrato_final(self) -> None:
        with tempfile.TemporaryDirectory() as temporario:
            raiz = Path(temporario)
            entrada = raiz / "txt"
            entrada.mkdir()
            (entrada / "doc_001.txt").write_text(
                "Conforme art. 373 do CPC.", encoding="utf-8"
            )

            orquestrador = Orquestrador(
                ExtratorFake(),
                CompletudeFake(),
                VeracidadeFake(),
            )
            orquestrador.executar(entrada, raiz / "run")

            destino = raiz / "run" / "predictions" / "doc_001.json"
            documento = DocumentoPredito.model_validate_json(
                destino.read_text(encoding="utf-8")
            )
            self.assertEqual(documento.citacoes[0].classificacao, Classificacao.REAL)
            self.assertEqual(documento.citacoes[0].resolucao.id_canonico, 28893055)


if __name__ == "__main__":
    unittest.main()
