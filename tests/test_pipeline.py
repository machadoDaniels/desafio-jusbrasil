import asyncio
import json
import tempfile
import unittest
from pathlib import Path

from pydantic import ValidationError

from desafio_jusbrasil.completeness import (
    executar_completude,
    executar_completude_async,
)
from desafio_jusbrasil.contracts import (
    CandidatoCitacao,
    CandidatoCitacaoRequest,
    Classificacao,
    ConsultaCanonica,
    DocumentoPredito,
    ResultadoCompletude,
    ResultadoVeracidade,
    TipoCitacao,
)
from desafio_jusbrasil.extractor import (
    AgenteExtrator,
    executar_extracao,
    executar_extracao_async,
)
from desafio_jusbrasil.orchestrator import Orquestrador
from desafio_jusbrasil.veracity import executar_veracidade, executar_veracidade_async


class ExtratorFake:
    def extrair(self, texto: str) -> list[CandidatoCitacao]:
        trecho = "art. 373 do CPC"
        inicio = texto.index(trecho)
        return [
            CandidatoCitacao(
                tipo=TipoCitacao.LEI,
                trecho=trecho,
                inicio=inicio,
                fim=inicio + len(trecho),
            )
        ]


class ExtratorAsyncFake:
    async def extrair_async(self, texto: str) -> list[CandidatoCitacao]:
        return ExtratorFake().extrair(texto)


class CompletudeFake:
    def classificar(self, candidato, contexto) -> ResultadoCompletude:
        return ResultadoCompletude(
            completa=True,
            consulta=ConsultaCanonica(
                tipo=candidato.tipo,
                dispositivo=candidato.trecho,
            ),
            justificativa="consulta específica",
        )


class CompletudeAsyncFake:
    async def classificar_async(self, candidato, contexto) -> ResultadoCompletude:
        return CompletudeFake().classificar(candidato, contexto)


class VeracidadeFake:
    def classificar(self, consulta) -> ResultadoVeracidade:
        return ResultadoVeracidade(
            classificacao=Classificacao.REAL,
            id_canonico=28893055,
            justificativa="registro único",
        )


class VeracidadeAsyncFake:
    async def classificar_async(self, consulta) -> ResultadoVeracidade:
        return VeracidadeFake().classificar(consulta)


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
            executar_completude(extracao, completude, CompletudeFake())
            executar_veracidade(completude, veracidade, VeracidadeFake())

            resultado = json.loads(
                (veracidade / "doc_001.json").read_text(encoding="utf-8")
            )
            self.assertEqual(resultado["documento_id"], "doc_001")
            self.assertEqual(
                resultado["candidatos"][0]["veracidade"]["classificacao"],
                "real",
            )

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
