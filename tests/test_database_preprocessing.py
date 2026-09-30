from __future__ import annotations

import asyncio
import hashlib
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from typing import Self
from unittest.mock import patch

from pydantic import ValidationError

from desafio_jusbrasil.database_preprocessing import (
    DocumentoEnriquecido,
    DocumentoFonte,
    MetadadosAcordao,
    MetadadosDispositivo,
    MetadadosSumula,
    agent,
    database,
    pipeline,
)
from desafio_jusbrasil.database_preprocessing import __main__ as cli
from desafio_jusbrasil.database_preprocessing.contracts import (
    _MetadadosAcordaoAgente,
    _MetadadosDispositivoAgente,
)
from desafio_jusbrasil.utils import normalizar_relator


def _cnj_valido() -> str:
    """Gera um CNJ cujo dígito verificador atende ao contrato."""
    inicio = "0000001"
    fim = "20241234567"
    base = int(inicio + fim + "00")
    return inicio + f"{98 - (base % 97):02d}" + fim


def _sha256(caminho: Path) -> str:
    return hashlib.sha256(caminho.read_bytes()).hexdigest()


def _criar_banco(caminho: Path) -> None:
    with sqlite3.connect(caminho) as conexao:
        conexao.executescript(
            """
            CREATE TABLE documentos (
                documento_id TEXT PRIMARY KEY,
                id INTEGER NOT NULL UNIQUE,
                natureza TEXT NOT NULL,
                tribunal TEXT,
                ano INTEGER,
                relator TEXT,
                texto TEXT NOT NULL
            );
            CREATE VIRTUAL TABLE documentos_fts USING fts5(
                texto, content='documentos', content_rowid='rowid',
                tokenize='unicode61 remove_diacritics 2'
            );
            INSERT INTO documentos VALUES
                ('acordao-1', 1, 'acordao', 'STJ', 2024, 'Ministra Ada', 'REsp julgamento'),
                ('sumula-1', 2, 'sumula', 'STF', NULL, NULL, 'Súmula vinculante 10'),
                ('dispositivo-1', 3, 'dispositivo', NULL, NULL, NULL, 'Art. 373 do CPC');
            INSERT INTO documentos_fts(documentos_fts) VALUES ('rebuild');
            """
        )


def _resultados() -> list[DocumentoEnriquecido]:
    return [
        DocumentoEnriquecido(
            documento_id="acordao-1",
            id=1,
            natureza="acordao",
            campos=MetadadosAcordao(
                numero_processo_cnj=_cnj_valido(),
                classe_processual="REsp",
                cadeia_recursal=["AgInt", "REsp"],
                uf="MG",
                relator_norm="ministra ada",
            ),
        ),
        DocumentoEnriquecido(
            documento_id="sumula-1",
            id=2,
            natureza="sumula",
            campos=MetadadosSumula(numero_sumula=10, sumula_vinculante=True),
        ),
        DocumentoEnriquecido(
            documento_id="dispositivo-1",
            id=3,
            natureza="dispositivo",
            campos=MetadadosDispositivo(
                diploma="Código de Processo Civil",
                numero_diploma="13105",
                ano_diploma=2015,
                numero_artigo="373-A",
            ),
        ),
    ]


class _RespostaFake:
    def __init__(self, campos: object) -> None:
        self.choices = [
            type("Choice", (), {"message": type("Message", (), {"parsed": campos})()})()
        ]
        self.usage = type(
            "Usage", (), {"model_dump": lambda self, **_: {"total_tokens": 7}}
        )()

    def model_dump(self, **_: object) -> dict[str, object]:
        return {"choices": [{"message": {"content": "fake"}}]}


class _ClienteFake:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.chat = type("Chat", (), {"completions": self})()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *_: object) -> None:
        return None

    async def parse(self, **requisicao: object) -> _RespostaFake:
        self.calls.append(requisicao)
        entrada = json.loads(requisicao["messages"][1]["content"])
        if "REsp" in entrada["texto"]:
            campos = _resultados()[0].campos
        elif "Súmula" in entrada["texto"]:
            campos = _resultados()[1].campos
        else:
            campos = _resultados()[2].campos
        return _RespostaFake(campos)


class DatabasePreprocessingTest(unittest.TestCase):
    def test_texto_enviado_ao_agente_respeita_limite(self) -> None:
        documento = DocumentoFonte(
            documento_id="doc",
            id=1,
            natureza="acordao",
            texto="abcdefghij",
        )
        entrada = agent._entrada_agente(
            documento,
            text_start_char_limit=3,
            text_end_char_limit=2,
        )
        self.assertEqual(entrada["texto"], "abcij")
        self.assertEqual(
            agent._entrada_agente(
                documento,
                text_start_char_limit=10,
                text_end_char_limit=10,
            )["texto"],
            "abcdefghij",
        )
        self.assertEqual(
            agent._entrada_agente(
                documento,
                text_start_char_limit=None,
                text_end_char_limit=None,
            )["texto"],
            "abcdefghij",
        )
        self.assertEqual(
            agent._entrada_agente(
                documento,
                text_start_char_limit=3,
                text_end_char_limit=None,
            )["texto"],
            "abc",
        )
        self.assertEqual(
            agent._entrada_agente(
                documento,
                text_start_char_limit=None,
                text_end_char_limit=2,
            )["texto"],
            "ij",
        )
        self.assertEqual(documento.texto, "abcdefghij")

    def test_contratos_dos_tres_tipos(self) -> None:
        acordao = MetadadosAcordao(
            numero_processo_cnj=_cnj_valido(),
            classe_processual="REsp",
            cadeia_recursal=["AgInt", "REsp"],
            uf="MG",
            relator_norm="ministra ada",
        )
        sumula = MetadadosSumula(numero_sumula=1, sumula_vinculante=False)
        dispositivo = MetadadosDispositivo(
            diploma="Lei", numero_diploma="123", ano_diploma=2024, numero_artigo="1º"
        )
        self.assertEqual(acordao.uf, "MG")
        self.assertFalse(sumula.sumula_vinculante)
        self.assertEqual(dispositivo.numero_artigo, "1º")
        with self.assertRaises(ValidationError):
            DocumentoEnriquecido(
                documento_id="x", id=1, natureza="sumula", campos=acordao
            )
        with self.assertRaises(ValidationError):
            MetadadosSumula(numero_sumula=1, inesperado=True)

    def test_numeros_e_cadeia_respeitam_contrato(self) -> None:
        self.assertEqual(
            MetadadosAcordao(numero_processo_cnj=_cnj_valido()).numero_processo_cnj,
            _cnj_valido(),
        )
        with self.assertRaisesRegex(ValidationError, "20 dígitos"):
            MetadadosAcordao(numero_processo_cnj="1234567")
        acordao_com_repeticao = MetadadosAcordao(
            numero_classe_tribunal="1234",
            numero_registro_tribunal="202201873194",
            classe_processual="REsp",
            cadeia_recursal=["REsp", "AgInt", "REsp"],
        )
        self.assertEqual(acordao_com_repeticao.classe_processual, "REsp")
        self.assertEqual(acordao_com_repeticao.cadeia_recursal, ["REsp", "AgInt"])
        self.assertEqual(
            MetadadosAcordao(
                classe_processual="REsp", cadeia_recursal=["AgInt"]
            ).cadeia_recursal,
            ["AgInt", "REsp"],
        )
        self.assertEqual(
            MetadadosAcordao(classe_processual="HC").cadeia_recursal, ["HC"]
        )
        self.assertIsNone(MetadadosAcordao().cadeia_recursal)
        self.assertIsNone(MetadadosAcordao(cadeia_recursal=[]).cadeia_recursal)

    def test_nomes_completos_do_agente_viram_siglas(self) -> None:
        acordao = MetadadosAcordao(
            classe_processual="AgInt — Agravo Interno",
            cadeia_recursal=["REsp — Recurso Especial"],
        )
        self.assertEqual(acordao.classe_processual, "AgInt")
        self.assertEqual(acordao.cadeia_recursal, ["REsp", "AgInt"])
        self.assertEqual(
            MetadadosDispositivo(diploma="CLT — Consolidação das Leis do Trabalho").diploma,
            "Consolidação das Leis do Trabalho",
        )
        self.assertEqual(MetadadosDispositivo(diploma="Lei").diploma, "Lei")

    def test_schema_expoe_descricoes_semanticas_ao_modelo(self) -> None:
        schema_acordao = _MetadadosAcordaoAgente.model_json_schema()
        schema_sumula = MetadadosSumula.model_json_schema()
        schema_dispositivo = _MetadadosDispositivoAgente.model_json_schema()
        self.assertIn(
            "20 dígitos",
            schema_acordao["properties"]["numero_processo_cnj"]["description"],
        )
        self.assertIn(
            "ordem", schema_acordao["properties"]["cadeia_recursal"]["description"]
        )
        self.assertIn(
            "false", schema_sumula["properties"]["sumula_vinculante"]["description"]
        )
        self.assertIn(
            "5.452", schema_dispositivo["properties"]["diploma"]["description"]
        )

    def test_relator_normalizado_por_regra(self) -> None:
        relatores = {
            "min. joão da silva": "joão da silva",
            "joão da silva": "joão da silva",
        }
        self.assertEqual(
            normalizar_relator("MIN. JOÃO DA SILVA", relatores),
            "joão da silva",
        )
        self.assertIsNone(normalizar_relator("Nome desconhecido", relatores))
        self.assertIsNone(normalizar_relator(None, relatores))

    def test_leitura_e_estritamente_read_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporario:
            origem = Path(temporario) / "origem.db"
            _criar_banco(origem)
            antes = _sha256(origem)
            self.assertEqual(len(database.listar_documentos(origem)), 3)
            with (
                database._conectar_somente_leitura(origem) as conexao,
                self.assertRaises(sqlite3.OperationalError),
            ):
                conexao.execute("DELETE FROM documentos")
            self.assertEqual(_sha256(origem), antes)

    def test_materializacao_copia_preserva_origem_fts_colunas_e_indices(self) -> None:
        with tempfile.TemporaryDirectory() as temporario:
            raiz = Path(temporario)
            origem, destino = raiz / "origem.db", raiz / "enriquecido.db"
            _criar_banco(origem)
            hash_origem, schema_origem = _sha256(origem), None
            with sqlite3.connect(origem) as conexao:
                schema_origem = conexao.execute(
                    "SELECT sql FROM sqlite_master WHERE name = 'documentos_fts'"
                ).fetchone()[0]

            diagnostico = database.materializar_banco(origem, destino, _resultados())

            self.assertEqual(diagnostico["documentos"], 3)
            self.assertEqual(diagnostico["documentos_fts"], 3)
            self.assertEqual(_sha256(origem), hash_origem)
            with sqlite3.connect(origem) as conexao:
                self.assertEqual(
                    conexao.execute("PRAGMA table_info(documentos)")
                    .fetchall()
                    .__len__(),
                    7,
                )
                self.assertEqual(
                    conexao.execute(
                        "SELECT sql FROM sqlite_master WHERE name = 'documentos_fts'"
                    ).fetchone()[0],
                    schema_origem,
                )
            with sqlite3.connect(destino) as conexao:
                colunas = {
                    linha[1]
                    for linha in conexao.execute("PRAGMA table_info(documentos)")
                }
                self.assertTrue(set(database._COLUNAS_PLANEJADAS).issubset(colunas))
                indices = {
                    linha[1]
                    for linha in conexao.execute("PRAGMA index_list(documentos)")
                }
                self.assertTrue(set(database._INDICES_PLANEJADOS).issubset(indices))
                acordao = conexao.execute(
                    "SELECT numero_processo_cnj, cadeia_recursal, relator_norm FROM documentos WHERE id = 1"
                ).fetchone()
                self.assertEqual(
                    acordao, (_cnj_valido(), '["AgInt","REsp"]', "ministra ada")
                )
                self.assertEqual(
                    conexao.execute(
                        "SELECT sumula_vinculante FROM documentos WHERE id = 2"
                    ).fetchone()[0],
                    1,
                )
                self.assertEqual(
                    conexao.execute(
                        "SELECT count(*) FROM documentos_fts WHERE documentos_fts MATCH 'REsp'"
                    ).fetchone()[0],
                    1,
                )
                conexao.execute(
                    "UPDATE documentos SET numero_processo_cnj = ? WHERE id = 1",
                    ("00000010020241234567",),
                )
                with self.assertRaises(sqlite3.IntegrityError):
                    conexao.execute(
                        "UPDATE documentos SET numero_processo_cnj = ? WHERE id = 1",
                        ("123",),
                    )
                with self.assertRaises(sqlite3.IntegrityError):
                    conexao.execute(
                        "UPDATE documentos SET cadeia_recursal = '{}' WHERE id = 1"
                    )

    def test_destino_existente_e_substituido(self) -> None:
        with tempfile.TemporaryDirectory() as temporario:
            raiz = Path(temporario)
            origem, destino = raiz / "origem.db", raiz / "destino.db"
            _criar_banco(origem)
            database.materializar_banco(origem, destino, _resultados())
            self.assertEqual(
                database.materializar_banco(origem, destino, _resultados())[
                    "integrity_check"
                ],
                "ok",
            )

    def test_cli_carrega_yaml(self) -> None:
        with tempfile.TemporaryDirectory() as temporario:
            caminho = Path(temporario) / "pipeline.yaml"
            caminho.write_text(
                "database_preprocessing:\n  model: modelo\n", encoding="utf-8"
            )
            self.assertEqual(cli._carregar_configuracao(caminho), {"model": "modelo"})
            caminho.write_text("extractor: {}\n", encoding="utf-8")
            with self.assertRaises(TypeError):
                cli._carregar_configuracao(caminho)

    def test_pipeline_fake_sem_rede_checkpoint_e_auditoria(self) -> None:
        with tempfile.TemporaryDirectory() as temporario:
            raiz = Path(temporario)
            origem, destino, auditoria = (
                raiz / "origem.db",
                raiz / "destino.db",
                raiz / "audit",
            )
            _criar_banco(origem)
            (raiz / "relatores_padronizacao.json").write_text(
                json.dumps({"Ministra Ada": "ministra ada"}), encoding="utf-8"
            )
            config = {
                "model": "modelo-fake",
                "base_url": None,
                "temperature": None,
                "top_p": None,
                "top_k": None,
                "reasoning_effort": None,
                "max_concurrency": 2,
                "max_retries": 1,
                "retry_delay_seconds": 0,
                "text_start_char_limit": 10_000,
                "text_end_char_limit": 10_000,
                "prompts": {
                    "acordao": "prompt acordao",
                    "sumula": "prompt sumula",
                    "dispositivo": "prompt dispositivo",
                },
            }
            cliente = _ClienteFake()
            with patch.object(pipeline, "AsyncOpenAI", return_value=cliente):
                relatorio = asyncio.run(
                    pipeline.executar_async(
                        origem=origem,
                        destino=destino,
                        diretorio_auditoria=auditoria,
                        configuracao=config,
                    )
                )
                relatorio_retomado = asyncio.run(
                    pipeline.executar_async(
                        origem=origem,
                        destino=destino,
                        diretorio_auditoria=auditoria,
                        configuracao=config,
                    )
                )

            self.assertEqual(len(cliente.calls), 3)
            entrada_acordao = json.loads(cliente.calls[0]["messages"][1]["content"])
            self.assertNotIn("documento_id", entrada_acordao)
            self.assertNotIn("id", entrada_acordao)
            self.assertNotIn("relator", entrada_acordao)
            self.assertNotIn("relatores_padronizados", entrada_acordao)
            self.assertEqual(relatorio["processados"], 3)
            self.assertEqual(relatorio["invalidos"], 0)
            self.assertEqual(relatorio_retomado["reutilizados"], 3)
            self.assertTrue((auditoria / "manifest.json").is_file())
            self.assertTrue((auditoria / "relatorio.json").is_file())
            self.assertEqual(
                (auditoria / "revisao.jsonl").read_text(encoding="utf-8"), ""
            )
            for documento_id in ("acordao-1", "sumula-1", "dispositivo-1"):
                pasta = auditoria / "documentos" / documento_id
                resultado = json.loads(
                    (pasta / "resultado.json").read_text(encoding="utf-8")
                )
                chamada = json.loads((pasta / "0001.json").read_text(encoding="utf-8"))
                self.assertNotIn("texto", resultado)
                self.assertEqual(list(chamada)[-2:], ["input", "output"])
            self.assertNotIn("avaliacao_gold", relatorio)
            self.assertEqual(relatorio["materializacao"]["integrity_check"], "ok")


if __name__ == "__main__":
    unittest.main()
