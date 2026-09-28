"""Leitura e materialização segura do SQLite canônico enriquecido."""

from __future__ import annotations

import json
import os
import sqlite3
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any
from urllib.parse import quote

from .contracts import DocumentoEnriquecido, DocumentoFonte

_CLASSES = (
    "'AI', 'APL', 'AR', 'AREsp', 'AREspe', 'ARR', 'AgInt', 'AgRg', 'E', "
    "'EDcl', 'HC', 'RE', 'REsp', 'REspe', 'RHC', 'RMS', 'RR', 'RSE', "
    "'Rcl', 'Rp', 'SLS'"
)
_UFS = (
    "'AC', 'AL', 'AP', 'AM', 'BA', 'CE', 'DF', 'ES', 'GO', 'MA', 'MT', "
    "'MS', 'MG', 'PA', 'PB', 'PR', 'PE', 'PI', 'RJ', 'RN', 'RS', 'RO', "
    "'RR', 'SC', 'SP', 'SE', 'TO'"
)
_DIPLOMAS = (
    "'Constituição Federal', 'Código Civil', 'Código de Defesa do Consumidor', "
    "'Código de Processo Civil', 'Código de Processo Penal', 'Código Penal Militar', "
    "'Código Eleitoral', 'Consolidação das Leis do Trabalho', 'Lei Complementar', 'Lei'"
)
_COLUNAS_PLANEJADAS = {
    "numero_processo_cnj": (
        "TEXT CHECK (numero_processo_cnj IS NULL OR "
        "(numero_processo_cnj NOT GLOB '*[^0-9]*' AND length(numero_processo_cnj) = 20))"
    ),
    "numero_classe_tribunal": (
        "TEXT CHECK (numero_classe_tribunal IS NULL OR "
        "(numero_classe_tribunal <> '' AND numero_classe_tribunal NOT GLOB '*[^0-9]*'))"
    ),
    "numero_registro_tribunal": (
        "TEXT CHECK (numero_registro_tribunal IS NULL OR "
        "(numero_registro_tribunal <> '' AND numero_registro_tribunal NOT GLOB '*[^0-9]*'))"
    ),
    "classe_processual": f"TEXT CHECK (classe_processual IS NULL OR classe_processual IN ({_CLASSES}))",
    "cadeia_recursal": (
        "TEXT CHECK (cadeia_recursal IS NULL OR "
        "(json_valid(cadeia_recursal) AND json_type(cadeia_recursal) = 'array'))"
    ),
    "uf": f"TEXT CHECK (uf IS NULL OR uf IN ({_UFS}))",
    "relator_norm": "TEXT",
    "numero_sumula": "INTEGER CHECK (numero_sumula IS NULL OR numero_sumula >= 1)",
    "sumula_vinculante": (
        "INTEGER CHECK (sumula_vinculante IS NULL OR sumula_vinculante IN (0, 1))"
    ),
    "diploma": f"TEXT CHECK (diploma IS NULL OR diploma IN ({_DIPLOMAS}))",
    "numero_diploma": (
        "TEXT CHECK (numero_diploma IS NULL OR "
        "(numero_diploma <> '' AND numero_diploma NOT GLOB '*[^0-9]*'))"
    ),
    "ano_diploma": "INTEGER CHECK (ano_diploma IS NULL OR ano_diploma BETWEEN 1 AND 9999)",
    "numero_artigo": "TEXT CHECK (numero_artigo IS NULL OR trim(numero_artigo) <> '')",
}
_INDICES_PLANEJADOS = {
    "idx_documentos_processo_cnj": (
        "ON documentos(numero_processo_cnj) WHERE natureza = 'acordao'"
    ),
    "idx_documentos_classe_tribunal": (
        "ON documentos(numero_classe_tribunal, classe_processual, tribunal) "
        "WHERE natureza = 'acordao'"
    ),
    "idx_documentos_registro_tribunal": (
        "ON documentos(numero_registro_tribunal, tribunal) WHERE natureza = 'acordao'"
    ),
    "idx_documentos_sumula": (
        "ON documentos(tribunal, numero_sumula, sumula_vinculante) "
        "WHERE natureza = 'sumula'"
    ),
    "idx_documentos_dispositivo": (
        "ON documentos(numero_diploma, ano_diploma, numero_artigo) "
        "WHERE natureza = 'dispositivo'"
    ),
    "idx_documentos_relator_norm": (
        "ON documentos(relator_norm) WHERE natureza = 'acordao'"
    ),
}


def _uri_somente_leitura(caminho: Path) -> str:
    return f"file:{quote(str(caminho.resolve()))}?mode=ro"


def _conectar_somente_leitura(caminho: Path) -> sqlite3.Connection:
    if not caminho.is_file():
        raise FileNotFoundError(f"banco de origem não encontrado: {caminho}")
    conexao = sqlite3.connect(_uri_somente_leitura(caminho), uri=True)
    conexao.row_factory = sqlite3.Row
    conexao.execute("PRAGMA query_only = ON")
    return conexao


def _objetos_fts(conexao: sqlite3.Connection) -> list[tuple[str, str, str, str | None]]:
    return [
        tuple(linha)
        for linha in conexao.execute(
            "SELECT type, name, tbl_name, sql FROM sqlite_master "
            "WHERE name = 'documentos_fts' OR name GLOB 'documentos_fts_*' "
            "ORDER BY type, name"
        )
    ]


def _validar_schema_fts(
    conexao: sqlite3.Connection,
) -> list[tuple[str, str, str, str | None]]:
    objetos = _objetos_fts(conexao)
    sql = next(
        (
            objeto[3]
            for objeto in objetos
            if objeto[0] == "table" and objeto[1] == "documentos_fts"
        ),
        None,
    )
    normalizado = "" if sql is None else "".join(sql.lower().split()).replace("'", "")
    esperado = (
        "createvirtualtabledocumentos_ftsusingfts5("
        "texto,content=documentos,content_rowid=rowid,"
        "tokenize=unicode61remove_diacritics2)"
    )
    if normalizado != esperado:
        raise ValueError("schema FTS documentos_fts inválido")
    return objetos


def listar_documentos(origem: Path) -> list[DocumentoFonte]:
    """Lista os documentos canônicos usando uma conexão SQLite estritamente read-only."""
    with _conectar_somente_leitura(origem) as conexao:
        linhas = conexao.execute(
            "SELECT documento_id, id, natureza, tribunal, ano, relator, texto "
            "FROM documentos ORDER BY id"
        ).fetchall()
    return [DocumentoFonte.model_validate(dict(linha)) for linha in linhas]


def _validar_resultados(
    origem: Path,
    resultados: Iterable[DocumentoEnriquecido],
    permitir_parcial: bool = False,
) -> list[DocumentoEnriquecido]:
    documentos = listar_documentos(origem)
    por_documento = {documento.documento_id: documento for documento in documentos}
    resultados_validados = [
        DocumentoEnriquecido.model_validate(resultado) for resultado in resultados
    ]
    if not permitir_parcial and len(resultados_validados) != len(por_documento):
        raise ValueError(
            "a quantidade de resultados não corresponde à quantidade de documentos"
        )

    vistos: set[str] = set()
    for resultado in resultados_validados:
        if resultado.documento_id in vistos:
            raise ValueError(f"resultado duplicado: {resultado.documento_id}")
        vistos.add(resultado.documento_id)
        documento = por_documento.get(resultado.documento_id)
        if documento is None:
            raise ValueError(
                f"resultado sem documento de origem: {resultado.documento_id}"
            )
        if (resultado.id, resultado.natureza) != (documento.id, documento.natureza):
            raise ValueError(f"identidade divergente para {resultado.documento_id}")
    return resultados_validados


def _adicionar_schema_planejado(
    conexao: sqlite3.Connection,
) -> tuple[list[str], list[str]]:
    existentes = {
        linha[1] for linha in conexao.execute("PRAGMA table_info(documentos)")
    }
    colunas_adicionadas = []
    for coluna, tipo in _COLUNAS_PLANEJADAS.items():
        if coluna not in existentes:
            conexao.execute(f"ALTER TABLE documentos ADD COLUMN {coluna} {tipo}")
            colunas_adicionadas.append(coluna)

    indices_existentes = {
        linha[0]
        for linha in conexao.execute(
            "SELECT name FROM sqlite_master WHERE type = 'index'"
        )
    }
    indices_adicionados = []
    for indice, definicao in _INDICES_PLANEJADOS.items():
        if indice not in indices_existentes:
            conexao.execute(f"CREATE INDEX {indice} {definicao}")
            indices_adicionados.append(indice)
    return colunas_adicionadas, indices_adicionados


def _valores_materializados(resultado: DocumentoEnriquecido) -> dict[str, Any]:
    valores: dict[str, Any] = {coluna: None for coluna in _COLUNAS_PLANEJADAS}
    campos = resultado.campos.model_dump(mode="json")
    for coluna in valores:
        if coluna in campos:
            valores[coluna] = campos[coluna]
    if valores["cadeia_recursal"] is not None:
        valores["cadeia_recursal"] = json.dumps(
            valores["cadeia_recursal"], ensure_ascii=False, separators=(",", ":")
        )
    if valores["sumula_vinculante"] is not None:
        valores["sumula_vinculante"] = int(valores["sumula_vinculante"])
    return valores


def _atualizar_documentos(
    conexao: sqlite3.Connection, resultados: list[DocumentoEnriquecido]
) -> None:
    atribuicoes = ", ".join(f"{coluna} = :{coluna}" for coluna in _COLUNAS_PLANEJADAS)
    sql = (
        f"UPDATE documentos SET {atribuicoes} "
        "WHERE documento_id = :documento_id AND id = :id AND natureza = :natureza"
    )
    for resultado in resultados:
        parametros = _valores_materializados(resultado)
        parametros.update(
            documento_id=resultado.documento_id,
            id=resultado.id,
            natureza=resultado.natureza,
        )
        cursor = conexao.execute(sql, parametros)
        if cursor.rowcount != 1:
            raise RuntimeError(f"falha ao atualizar {resultado.documento_id}")


def _validar_copia(
    conexao: sqlite3.Connection,
    *,
    documentos_origem: list[tuple[str, int, str]],
    objetos_fts_origem: list[tuple[str, str, str, str | None]],
    contagem_fts_origem: int,
    resultados: list[DocumentoEnriquecido],
) -> None:
    documentos_copia = [
        tuple(linha)
        for linha in conexao.execute(
            "SELECT documento_id, id, natureza FROM documentos ORDER BY id"
        )
    ]
    if documentos_copia != documentos_origem:
        raise RuntimeError("contagens ou IDs dos documentos divergem da origem")
    if _validar_schema_fts(conexao) != objetos_fts_origem:
        raise RuntimeError("objetos FTS foram alterados")
    contagem_fts = conexao.execute("SELECT count(*) FROM documentos_fts").fetchone()[0]
    if contagem_fts != contagem_fts_origem:
        raise RuntimeError("cardinalidade FTS foi alterada")

    for resultado in resultados:
        esperado = _valores_materializados(resultado)
        colunas = ", ".join(_COLUNAS_PLANEJADAS)
        linha = conexao.execute(
            f"SELECT {colunas} FROM documentos WHERE documento_id = ?",
            (resultado.documento_id,),
        ).fetchone()
        if linha is None or dict(linha) != esperado:
            raise RuntimeError(
                f"valores materializados inválidos: {resultado.documento_id}"
            )

    integridade = conexao.execute("PRAGMA integrity_check").fetchone()[0]
    if integridade != "ok":
        raise RuntimeError(f"integrity_check falhou: {integridade}")


def materializar_banco(
    origem: Path,
    destino: Path,
    resultados: Iterable[DocumentoEnriquecido],
    permitir_parcial: bool = False,
) -> dict[str, Any]:
    """Copia, enriquece, valida e publica o SQLite por ``rename`` atômico.

    Com ``permitir_parcial``, documentos sem resultado mantêm as colunas novas nulas.
    """
    destino.parent.mkdir(parents=True, exist_ok=True)
    resultados_validados = _validar_resultados(origem, resultados, permitir_parcial)

    with _conectar_somente_leitura(origem) as conexao_origem:
        objetos_fts_origem = _validar_schema_fts(conexao_origem)
        contagem_fts_origem = conexao_origem.execute(
            "SELECT count(*) FROM documentos_fts"
        ).fetchone()[0]
        documentos_origem = [
            tuple(linha)
            for linha in conexao_origem.execute(
                "SELECT documento_id, id, natureza FROM documentos ORDER BY id"
            )
        ]
        descritor, nome_temporario = tempfile.mkstemp(
            prefix=f".{destino.name}.", suffix=".tmp", dir=destino.parent
        )
        os.close(descritor)
        temporario = Path(nome_temporario)
        try:
            with sqlite3.connect(temporario) as conexao_destino:
                conexao_destino.row_factory = sqlite3.Row
                conexao_origem.backup(conexao_destino)
                colunas_adicionadas, indices_adicionados = _adicionar_schema_planejado(
                    conexao_destino
                )
                _atualizar_documentos(conexao_destino, resultados_validados)
                _validar_copia(
                    conexao_destino,
                    documentos_origem=documentos_origem,
                    objetos_fts_origem=objetos_fts_origem,
                    contagem_fts_origem=contagem_fts_origem,
                    resultados=resultados_validados,
                )
            os.replace(temporario, destino)
        finally:
            temporario.unlink(missing_ok=True)

    return {
        "documentos": len(documentos_origem),
        "resultados": len(resultados_validados),
        "colunas_adicionadas": colunas_adicionadas,
        "indices_adicionados": indices_adicionados,
        "integrity_check": "ok",
        "documentos_fts": contagem_fts_origem,
    }
