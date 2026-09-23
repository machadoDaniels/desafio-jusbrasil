import asyncio
import json
import sqlite3
import tempfile
import unittest
from importlib import import_module
from pathlib import Path
from unittest.mock import Mock

from openai import OpenAI
from pydantic import ValidationError

from desafio_jusbrasil.contracts import (
    CandidatoAnalisado,
    CandidatoCitacao,
    CandidatoCitacaoRequest,
    Classificacao,
    ConsultaJurisprudencia,
    ConsultaJurisprudenciaAgente,
    ConsultaLegislacao,
    ConsultaLegislacaoAgente,
    DocumentoCompletude,
    DocumentoExtraido,
    DocumentoPredito,
    LoteCandidatosRequest,
    ResultadoCompletude,
    ResultadoVeracidade,
    StageConfig,
    TipoCitacao,
)
from desafio_jusbrasil.orchestrator import Orquestrador
from desafio_jusbrasil.utils import (
    escrever_manifesto_etapa,
    escrever_saida_documento,
    normalizar_numero_cnj,
)

_extractor = import_module("desafio_jusbrasil.1_extractor")
_completeness = import_module("desafio_jusbrasil.2_completeness")
_entities = import_module("desafio_jusbrasil.3_entities")
_veracity = import_module("desafio_jusbrasil.4_veracity")
AgenteExtrator = _extractor.AgenteExtrator
executar_extracao = _extractor.executar_extracao
executar_extracao_async = _extractor.executar_extracao_async
AgenteCompletude = _completeness.AgenteCompletude
executar_completude = _completeness.executar_completude
executar_completude_async = _completeness.executar_completude_async
AgenteExtratorEntidades = _entities.AgenteExtratorEntidades
_prompt_veracidade = _entities._prompt_veracidade
executar_entities_async = _entities.executar_entities_async
VerificadorVeracidade = _veracity.VerificadorVeracidade
executar_veracidade = _veracity.executar_veracidade


def _auditoria_fake() -> dict:
    return {"input": {"messages": []}, "output": {"choices": []}}


class ExtratorFake:
    def extrair_auditada(
        self,
        texto: str,
    ) -> tuple[list[CandidatoCitacao], dict]:
        trecho = "art. 373 do CPC"
        inicio = texto.index(trecho)
        return [
            CandidatoCitacao(
                tipo=TipoCitacao.LEI,
                trecho=trecho,
                confianca_extracao=0.75,
                inicio=inicio,
                fim=inicio + len(trecho),
            )
        ], _auditoria_fake()


class ExtratorAsyncFake:
    async def extrair_auditada_async(
        self,
        texto: str,
    ) -> tuple[list[CandidatoCitacao], dict]:
        return ExtratorFake().extrair_auditada(texto)


class CompletudeFake:
    def classificar_auditada(
        self,
        candidato: CandidatoCitacao,
        contexto: str,
    ) -> tuple[ResultadoCompletude, list[dict]]:
        return ResultadoCompletude(completa=True), [_auditoria_fake()]


class CompletudeAsyncFake:
    async def classificar_auditada_async(
        self,
        candidato: CandidatoCitacao,
        contexto: str,
    ) -> tuple[ResultadoCompletude, list[dict]]:
        return CompletudeFake().classificar_auditada(candidato, contexto)


class CompletudeAsyncComFalha(CompletudeAsyncFake):
    async def classificar_auditada_async(
        self,
        candidato: CandidatoCitacao,
        contexto: str,
    ) -> tuple[ResultadoCompletude, list[dict]]:
        if candidato.trecho == "falha":
            await asyncio.sleep(0.01)
            raise RuntimeError("falha simulada")
        return await super().classificar_auditada_async(candidato, contexto)


class EntidadesAsyncFake:
    def __init__(self) -> None:
        self.chamadas = 0

    async def extrair_auditada_async(
        self,
        candidato: CandidatoCitacao,
    ) -> tuple[ConsultaJurisprudencia | ConsultaLegislacao, dict]:
        self.chamadas += 1
        if candidato.tipo == TipoCitacao.LEI:
            campos = ConsultaLegislacao(
                numero_artigo="373",
                diploma="Código de Processo Civil",
            )
        else:
            campos = ConsultaJurisprudencia(
                natureza="acordao",
                numero_classe_tribunal="123",
            )
        return campos, _auditoria_fake()


class EntidadesAsyncComFalha(EntidadesAsyncFake):
    async def extrair_auditada_async(
        self,
        candidato: CandidatoCitacao,
    ) -> tuple[ConsultaJurisprudencia | ConsultaLegislacao, dict]:
        if candidato.trecho == "falha":
            await asyncio.sleep(0.01)
            raise RuntimeError("falha simulada")
        return await super().extrair_auditada_async(candidato)


class VerificadorFake:
    def verificar(
        self,
        consulta: ConsultaJurisprudencia | ConsultaLegislacao,
    ) -> tuple[ResultadoVeracidade, dict]:
        resultado = ResultadoVeracidade(
            classificacao=Classificacao.REAL,
            id_canonico=28893055,
            justificativa="registro único",
        )
        return resultado, {
            "campos_extraidos": consulta.model_dump(mode="json"),
            "sql": "SELECT 1",
            "parametros": [],
            "registros": [{"id": 28893055}],
            "resultado": resultado.model_dump(mode="json"),
        }


class PipelineTest(unittest.TestCase):
    def test_llm_nao_recebe_campos_de_span(self) -> None:
        self.assertNotIn("inicio", CandidatoCitacaoRequest.model_fields)
        self.assertNotIn("fim", CandidatoCitacaoRequest.model_fields)

    def test_extracao_repete_resposta_invalida(self) -> None:
        payload_invalido = {
            "candidatos": [{"trecho": "Rcl 44.960/PE", "type": "jurisprudencia"}]
        }
        with self.assertRaises(ValidationError) as contexto:
            LoteCandidatosRequest.model_validate(payload_invalido)

        lote = LoteCandidatosRequest(
            candidatos=[
                CandidatoCitacaoRequest(
                    trecho="Rcl 44.960/PE",
                    tipo=TipoCitacao.JURISPRUDENCIA,
                )
            ]
        )
        resposta = Mock()
        resposta.choices = [Mock(message=Mock(parsed=lote))]
        resposta.model_dump.return_value = {"choices": []}
        cliente = Mock(spec=OpenAI)
        cliente.chat.completions.parse.side_effect = [contexto.exception, resposta]

        with self.assertLogs("desafio_jusbrasil.1_extractor", level="WARNING") as logs:
            candidatos, auditoria = AgenteExtrator(
                cliente, StageConfig(model="modelo")
            )._consultar_modelo("Rcl 44.960/PE")

        self.assertEqual(cliente.chat.completions.parse.call_count, 2)
        self.assertEqual(candidatos, lote.candidatos)
        self.assertEqual(len(logs.output), 1)
        self.assertIn("retry 1/4", logs.output[0])
        self.assertEqual(auditoria["campos_extraidos"][0]["tipo"], "jurisprudencia")
        self.assertEqual(list(auditoria)[-2:], ["input", "output"])

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

    def test_checkpoint_nao_inclui_chamadas_do_modelo(self) -> None:
        documento = DocumentoExtraido(
            documento_id="doc",
            candidatos=[],
            texto="texto",
        )
        serializado = documento.model_dump(mode="json")
        self.assertNotIn("chamadas_modelo", serializado)
        self.assertNotIn("texto", serializado)

    def test_saida_separa_resultado_de_cada_chamada(self) -> None:
        with tempfile.TemporaryDirectory() as temporario:
            destino = Path(temporario)
            documento = DocumentoExtraido(
                documento_id="doc",
                candidatos=[],
                texto="texto",
            )
            chamadas = [_auditoria_fake(), _auditoria_fake()]

            escrever_saida_documento(destino, documento, chamadas)

            pasta = destino / "doc"
            self.assertEqual(
                sorted(arquivo.name for arquivo in pasta.glob("*.json")),
                ["0001.json", "0002.json", "resultado.json"],
            )
            resultado = json.loads((pasta / "resultado.json").read_text())
            self.assertNotIn("chamadas_modelo", resultado)
            self.assertNotIn("texto", resultado)
            self.assertEqual(
                json.loads((pasta / "0001.json").read_text()),
                _auditoria_fake(),
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
        mensagens = AgenteExtratorEntidades._mensagens(candidato)
        self.assertIn(candidato.trecho, mensagens[1]["content"])
        self.assertIn(
            "EDcl no AgInt no Agravo em Recurso Especial", mensagens[0]["content"]
        )
        self.assertIn(
            '"classe_processual": "AREsp — Agravo em Recurso Especial"',
            mensagens[0]["content"],
        )
        prompt_lei = AgenteExtratorEntidades._mensagens(
            CandidatoCitacao(
                trecho="art. 1.134 da Lei nº 13.105/2015", tipo=TipoCitacao.LEI
            )
        )[0]["content"]
        self.assertIn("CPC — Código de Processo Civil", prompt_lei)
        schema_jurisprudencia = ConsultaJurisprudencia.model_json_schema()["properties"]
        self.assertEqual(
            schema_jurisprudencia["numero_classe_tribunal"]["examples"],
            ["68244", "1741784"],
        )
        self.assertEqual(
            schema_jurisprudencia["numero_registro_tribunal"]["examples"],
            ["202201873194", "201801163041"],
        )
        self.assertNotEqual(
            _prompt_veracidade(TipoCitacao.JURISPRUDENCIA),
            _prompt_veracidade(TipoCitacao.LEI),
        )
        self.assertEqual(
            set(ConsultaJurisprudencia.model_fields),
            {
                "natureza",
                "numero_processo_cnj",
                "numero_classe_tribunal",
                "numero_registro_tribunal",
                "classe_processual",
                "cadeia_recursal",
                "tribunal",
                "uf",
                "ano",
                "relator",
                "relator_norm",
                "numero_sumula",
                "sumula_vinculante",
            },
        )
        self.assertEqual(
            set(ConsultaLegislacao.model_fields),
            {
                "numero_artigo",
                "diploma",
                "numero_diploma",
            },
        )
        self.assertTrue(
            all(
                not campo.is_required()
                for contrato in (ConsultaJurisprudencia, ConsultaLegislacao)
                for campo in contrato.model_fields.values()
            )
        )
        self.assertEqual(
            ConsultaJurisprudencia(
                cadeia_recursal=["REsp", "AgInt", "REsp"]
            ).cadeia_recursal,
            ["REsp", "AgInt"],
        )
        self.assertIsNone(ConsultaJurisprudencia(cadeia_recursal=[]).cadeia_recursal)
        classe_agente = ConsultaJurisprudenciaAgente.model_json_schema()["properties"][
            "classe_processual"
        ]["anyOf"][0]["enum"]
        self.assertIn("Rcl — Reclamação", classe_agente)
        self.assertIsNone(
            ConsultaJurisprudencia(
                numero_processo_cnj="00003788220166050151", ano=2016
            ).ano
        )
        self.assertEqual(
            ConsultaJurisprudencia(
                classe_processual="Rcl — Reclamação",
                cadeia_recursal=[
                    "AgInt — Agravo Interno",
                    "Rcl — Reclamação",
                ],
            ).model_dump(include={"classe_processual", "cadeia_recursal"}),
            {
                "classe_processual": "Rcl",
                "cadeia_recursal": ["AgInt", "Rcl"],
            },
        )
        self.assertEqual(
            ConsultaLegislacao(numero_artigo="Art. 1.105").numero_artigo,
            "1105",
        )
        self.assertEqual(
            ConsultaLegislacao(numero_diploma="Lei nº 13.105").numero_diploma,
            "13105",
        )
        diplomas_agente = ConsultaLegislacaoAgente.model_json_schema()["properties"][
            "diploma"
        ]["anyOf"][0]["enum"]
        self.assertIn("CLT — Consolidação das Leis do Trabalho", diplomas_agente)
        self.assertEqual(
            ConsultaLegislacao(
                diploma="CLT — Consolidação das Leis do Trabalho"
            ).diploma,
            "Consolidação das Leis do Trabalho",
        )
        self.assertEqual(
            ConsultaJurisprudencia(
                numero_classe_tribunal="Rcl 68.244",
                numero_registro_tribunal="2022/0187319-4",
            ).model_dump(
                include={"numero_classe_tribunal", "numero_registro_tribunal"}
            ),
            {
                "numero_classe_tribunal": "68244",
                "numero_registro_tribunal": "202201873194",
            },
        )

    def test_identificadores_do_agente_sao_normalizados_no_contrato(self) -> None:
        consulta = ConsultaJurisprudenciaAgente(
            numero_classe_tribunal="Rcl 55.626",
            numero_registro_tribunal="2022/0187319-4",
        )
        self.assertEqual(consulta.numero_classe_tribunal, "55626")
        self.assertEqual(consulta.numero_registro_tribunal, "202201873194")

    def test_numero_cnj_recupera_zeros_do_trecho(self) -> None:
        self.assertEqual(
            normalizar_numero_cnj(
                "3788220166050151",
                "REspe nº 378-82.2016.6.05.0151",
            ),
            "00003788220166050151",
        )

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
                        'jurisprudencia', 'AgInt no Agravo em Recurso Especial 1 996 496 RJ', 53
                    );
                    INSERT INTO documentos_fts(documentos_fts) VALUES ('rebuild');
                    """
                )

            verificador = object.__new__(VerificadorVeracidade)
            verificador._database = banco
            consulta = ConsultaJurisprudencia(
                natureza="acordao",
                numero_classe_tribunal="1996496",
                classe_processual="AREsp",
                cadeia_recursal=["AgInt", "AREsp"],
                tribunal="STJ",
                uf="RJ",
                ano=2023,
                relator="Maria Silva",
            )
            registros, sql, parametros = verificador._consultar_base(consulta)
            self.assertEqual(registros, [{"id": 42, "documento_id": "doc_1"}])
            self.assertEqual(
                parametros[0],
                'NEAR("AgInt" "AGRAVO EM RECURSO ESPECIAL" "1 996 496", 20)',
            )
            self.assertNotIn("d.tribunal = ?", sql)
            self.assertEqual(
                verificador._classificar(registros).classificacao,
                Classificacao.REAL,
            )

    def test_veracidade_prioriza_colunas_estruturadas(self) -> None:
        with tempfile.TemporaryDirectory() as temporario:
            banco = Path(temporario) / "base.db"
            with sqlite3.connect(banco) as conexao:
                conexao.executescript(
                    """
                    CREATE TABLE documentos (
                        documento_id TEXT PRIMARY KEY, id INTEGER UNIQUE,
                        natureza TEXT, tipo TEXT, texto TEXT,
                        numero_processo_cnj TEXT, numero_sumula INTEGER,
                        sumula_vinculante INTEGER, numero_artigo TEXT,
                        numero_diploma TEXT, diploma TEXT, ano_diploma INTEGER
                    );
                    CREATE VIRTUAL TABLE documentos_fts USING fts5(
                        texto, content='documentos', content_rowid='rowid'
                    );
                    INSERT INTO documentos VALUES (
                        'doc_1', 42, 'acordao', 'jurisprudencia', 'texto sem número',
                        '00003788220166050151', NULL, NULL, NULL, NULL, NULL, NULL
                    );
                    INSERT INTO documentos_fts(documentos_fts) VALUES ('rebuild');
                    """
                )
            verificador = VerificadorVeracidade(banco)
            registros, sql, parametros = verificador._consultar_base(
                ConsultaJurisprudencia(
                    natureza="acordao",
                    numero_processo_cnj="00003788220166050151",
                )
            )
            self.assertEqual(registros, [{"id": 42, "documento_id": "doc_1"}])
            self.assertIn("numero_processo_cnj = ?", sql)
            self.assertEqual(parametros, ["00003788220166050151"])

    def test_veracidade_busca_classe_e_registro_nas_duas_colunas(self) -> None:
        colunas = {"numero_classe_tribunal", "numero_registro_tribunal"}
        for consulta in (
            ConsultaJurisprudencia(numero_classe_tribunal="68244"),
            ConsultaJurisprudencia(numero_registro_tribunal="68244"),
        ):
            sql, parametros = VerificadorVeracidade._consulta_estruturada(
                consulta, colunas
            )
            self.assertIn("numero_classe_tribunal IN (?)", sql)
            self.assertIn("numero_registro_tribunal IN (?)", sql)
            self.assertEqual(parametros, ["68244", "68244"])

    def test_veracidade_sem_identificador_retorna_incompleta(self) -> None:
        verificador = object.__new__(VerificadorVeracidade)

        resultado, verificacao = verificador.verificar(ConsultaJurisprudencia())

        self.assertEqual(resultado.classificacao, Classificacao.INCOMPLETA)
        self.assertIsNone(verificacao["sql"])
        self.assertEqual(verificacao["parametros"], [])
        self.assertEqual(verificacao["registros"], [])

    def test_veracidade_deriva_termos_dos_campos_juridicos(self) -> None:
        self.assertEqual(
            VerificadorVeracidade._valores_fts(
                ConsultaJurisprudencia(
                    natureza="sumula",
                    numero_sumula=10,
                    sumula_vinculante=True,
                )
            ),
            ["sumula", "vinculante", "10"],
        )
        self.assertEqual(
            VerificadorVeracidade._valores_fts(
                ConsultaLegislacao(
                    numero_artigo="373",
                    diploma="Código de Processo Civil",
                    numero_diploma="13.105",
                )
            ),
            ["Artigo 373", "13 105"],
        )
        self.assertEqual(
            VerificadorVeracidade._normalizar_numero("AgInt no REsp 21737l8-SP"),
            "2 173 718",
        )
        self.assertEqual(
            VerificadorVeracidade._normalizar_numero("7000553-0320217000000"),
            "7000553 03 2021 7 00 0000",
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
            entities = raiz / "03-entities"
            veracidade = raiz / "04-veracity"

            executar_extracao(entrada, extracao, ExtratorFake())
            (extracao / "manifest.json").write_text("{}", encoding="utf-8")
            executar_completude(extracao, completude, CompletudeFake(), entrada)
            (completude / "manifest.json").write_text("{}", encoding="utf-8")
            asyncio.run(
                executar_entities_async(
                    completude,
                    entities,
                    EntidadesAsyncFake(),
                    max_concurrency=2,
                )
            )
            executar_veracidade(entities, veracidade, VerificadorFake())

            resultado = json.loads(
                (veracidade / "doc_001" / "resultado.json").read_text(encoding="utf-8")
            )
            self.assertEqual(resultado["documento_id"], "doc_001")
            self.assertEqual(
                resultado["candidatos"][0]["veracidade"]["classificacao"],
                "real",
            )
            self.assertNotIn("chamadas_modelo", resultado)
            self.assertTrue((veracidade / "doc_001" / "0001.json").exists())

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
                for arquivo in sorted(destino.glob("*/resultado.json"))
            ]
            self.assertEqual(ids, ["a", "b"])

    def test_completude_async_salva_documento_antes_de_falha_posterior(self) -> None:
        with tempfile.TemporaryDirectory() as temporario:
            raiz = Path(temporario)
            entrada = raiz / "entrada"
            textos = raiz / "txt"
            saida = raiz / "saida"
            entrada.mkdir()
            textos.mkdir()
            for documento_id, texto in (("a", "art. 373 do CPC"), ("b", "falha")):
                (textos / f"{documento_id}.txt").write_text(texto, encoding="utf-8")
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
                        input_dir=textos,
                    )
                )

            self.assertTrue((saida / "a" / "resultado.json").exists())
            self.assertFalse((saida / "b" / "resultado.json").exists())

    def test_entities_extrai_campos_parciais_de_citacao_incompleta(self) -> None:
        with tempfile.TemporaryDirectory() as temporario:
            raiz = Path(temporario)
            entrada, saida = raiz / "entrada", raiz / "saida"
            entrada.mkdir()
            candidato = CandidatoCitacao(
                trecho="art. 373 do CPC",
                tipo=TipoCitacao.LEI,
                inicio=0,
                fim=15,
            )
            documento = DocumentoCompletude(
                documento_id="doc",
                candidatos=[
                    CandidatoAnalisado(
                        candidato=candidato,
                        completude=ResultadoCompletude(completa=False),
                    )
                ],
            )
            (entrada / "doc.json").write_text(
                documento.model_dump_json(), encoding="utf-8"
            )

            asyncio.run(
                executar_entities_async(
                    entrada,
                    saida,
                    EntidadesAsyncFake(),
                    max_concurrency=1,
                )
            )

            resultado = json.loads(
                (saida / "doc" / "resultado.json").read_text(encoding="utf-8")
            )
            self.assertFalse(resultado["candidatos"][0]["completude"]["completa"])
            self.assertEqual(
                resultado["candidatos"][0]["campos_extraidos"]["numero_artigo"],
                "373",
            )
            self.assertTrue((saida / "doc" / "0001.json").is_file())

    def test_entities_reprocessa_documentos_ja_existentes(self) -> None:
        with tempfile.TemporaryDirectory() as temporario:
            raiz = Path(temporario)
            entrada, saida = raiz / "entrada", raiz / "saida"
            entrada.mkdir()
            candidato = CandidatoCitacao(
                trecho="art. 373 do CPC",
                tipo=TipoCitacao.LEI,
                inicio=0,
                fim=15,
            )
            documento = DocumentoCompletude(
                documento_id="doc",
                candidatos=[
                    CandidatoAnalisado(
                        candidato=candidato,
                        completude=ResultadoCompletude(completa=True),
                    )
                ],
            )
            (entrada / "doc.json").write_text(
                documento.model_dump_json(), encoding="utf-8"
            )
            extrator = EntidadesAsyncFake()

            for _ in range(2):
                asyncio.run(
                    executar_entities_async(
                        entrada,
                        saida,
                        extrator,
                        max_concurrency=1,
                    )
                )

            self.assertEqual(extrator.chamadas, 2)

    def test_entities_async_salva_documento_antes_de_falha_posterior(self) -> None:
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
                    executar_entities_async(
                        entrada,
                        saida,
                        EntidadesAsyncComFalha(),
                        max_concurrency=2,
                    )
                )

            self.assertTrue((saida / "a" / "resultado.json").exists())
            self.assertFalse((saida / "b" / "resultado.json").exists())

    def test_entities_salva_citacao_antes_de_falha_posterior_no_documento(self) -> None:
        with tempfile.TemporaryDirectory() as temporario:
            raiz = Path(temporario)
            entrada = raiz / "entrada"
            saida = raiz / "saida"
            entrada.mkdir()
            candidatos = [
                CandidatoAnalisado(
                    candidato=CandidatoCitacao(
                        trecho=texto,
                        tipo=TipoCitacao.LEI,
                        inicio=inicio,
                        fim=inicio + len(texto),
                    ),
                    completude=ResultadoCompletude(completa=True),
                )
                for texto, inicio in (("art. 373 do CPC", 0), ("falha", 20))
            ]
            documento = DocumentoCompletude(
                documento_id="doc", texto="", candidatos=candidatos
            )
            (entrada / "doc.json").write_text(
                documento.model_dump_json(), encoding="utf-8"
            )

            with self.assertRaisesRegex(RuntimeError, "falha simulada"):
                asyncio.run(
                    executar_entities_async(
                        entrada,
                        saida,
                        EntidadesAsyncComFalha(),
                        max_concurrency=1,
                    )
                )

            resultado = json.loads((saida / "doc" / "resultado.json").read_text())
            self.assertEqual(len(resultado["candidatos"]), 1)
            self.assertTrue((saida / "doc" / "0001.json").exists())

    def test_completude_entities_e_veracidade(self) -> None:
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
            entities = raiz / "03"
            veracidade = raiz / "04"
            executar_extracao(entrada, extracao, ExtratorFake())
            asyncio.run(
                executar_completude_async(
                    extracao,
                    completude,
                    CompletudeAsyncFake(),
                    max_concurrency=2,
                    input_dir=entrada,
                )
            )
            asyncio.run(
                executar_entities_async(
                    completude,
                    entities,
                    EntidadesAsyncFake(),
                    max_concurrency=2,
                )
            )
            executar_veracidade(entities, veracidade, VerificadorFake())
            resultado = json.loads(
                (veracidade / "doc" / "resultado.json").read_text(encoding="utf-8")
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
                EntidadesAsyncFake(),
                VerificadorFake(),
            )
            asyncio.run(orquestrador.executar(entrada, raiz / "run"))

            destino = raiz / "run" / "predictions" / "doc_001.json"
            documento = DocumentoPredito.model_validate_json(
                destino.read_text(encoding="utf-8")
            )
            self.assertEqual(documento.citacoes[0].classificacao, Classificacao.REAL)
            self.assertEqual(documento.citacoes[0].resolucao.id_canonico, 28893055)
            self.assertEqual(documento.citacoes[0].confianca, 0.75)


if __name__ == "__main__":
    unittest.main()
