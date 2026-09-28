"""Fallback FTS5 para consultas canônicas sem chave estruturada utilizável."""

from __future__ import annotations

import argparse
import re
import sqlite3
from importlib import import_module
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from .contracts import (
    ConsultaJurisprudencia,
    ConsultaLegislacao,
    PipelineConfig,
    TipoCitacao,
)
from .utils import escrever_manifesto_etapa, materializar

_veracity = import_module(".4_veracity", __package__)


class VerificadorVeracidadeFts(_veracity.VerificadorVeracidade):
    """Verificador alternativo que consulta exclusivamente o índice FTS5."""

    def _consultar_base(
        self,
        consulta: ConsultaJurisprudencia | ConsultaLegislacao,
    ) -> tuple[list[dict[str, Any]], str, list[Any]]:
        uri = f"file:{self._database}?mode=ro"
        with sqlite3.connect(uri, uri=True) as conexao:
            conexao.row_factory = sqlite3.Row
            conexao.execute("PRAGMA query_only = ON")
            return consultar_fts(conexao, consulta)


def consultar_fts(
    conexao: sqlite3.Connection,
    consulta: ConsultaJurisprudencia | ConsultaLegislacao,
) -> tuple[list[dict[str, Any]], str, list[Any]]:
    """Executa o fallback FTS5 usando os campos já extraídos da citação."""
    expressao = montar_expressao(consulta)
    filtros = ["documentos_fts MATCH ?", "d.tipo = ?", "d.natureza = ?"]
    if isinstance(consulta, ConsultaJurisprudencia):
        natureza = consulta.natureza or (
            "sumula" if consulta.numero_sumula else "acordao"
        )
        parametros = [expressao, TipoCitacao.JURISPRUDENCIA.value, natureza]
        if consulta.numero_sumula and consulta.tribunal:
            filtros.append("d.tribunal = ? COLLATE NOCASE")
            parametros.append(consulta.tribunal)
    else:
        parametros = [expressao, TipoCitacao.LEI.value, "dispositivo"]
    sql = (
        "SELECT d.id, d.documento_id FROM documentos_fts "
        "JOIN documentos AS d ON d.rowid = documentos_fts.rowid WHERE "
        + " AND ".join(filtros)
        + " LIMIT 10"
    )
    linhas = conexao.execute(sql, parametros).fetchall()
    return [dict(linha) for linha in linhas], sql, parametros


def montar_expressao(
    consulta: ConsultaJurisprudencia | ConsultaLegislacao,
) -> str:
    """Monta a expressão FTS5 a partir dos campos disponíveis."""
    valores = valores_fts(consulta)
    if (
        isinstance(consulta, ConsultaJurisprudencia)
        and _numero_jurisprudencia(consulta)
        and consulta.cadeia_recursal
        and len(consulta.cadeia_recursal) > 1
    ):
        classes = [normalizar_classe(item) for item in consulta.cadeia_recursal]
        termos = " ".join(f'"{valor}"' for valor in [*classes, valores[0]])
        return f"NEAR({termos}, 20)"
    return " AND ".join(f'"{valor}"' for valor in valores)


def valores_fts(
    consulta: ConsultaJurisprudencia | ConsultaLegislacao,
) -> list[str]:
    """Converte campos jurídicos em termos pesquisáveis no FTS5."""
    if isinstance(consulta, ConsultaJurisprudencia):
        numero = _numero_jurisprudencia(consulta)
        if numero:
            return [normalizar_numero(numero)]
        if consulta.numero_sumula is not None:
            valores = ["sumula", normalizar_numero(str(consulta.numero_sumula))]
            if consulta.sumula_vinculante:
                valores.insert(1, "vinculante")
            return valores
        raise ValueError("jurisprudência sem processo ou súmula")
    if not consulta.numero_artigo:
        raise ValueError("legislação sem número de artigo")
    valores = [f"Artigo {normalizar_numero(consulta.numero_artigo)}"]
    diploma = normalizar_diploma(consulta)
    if diploma:
        valores.append(diploma)
    return valores


def normalizar_numero(valor: str) -> str:
    """Converte um identificador publicado em termos numéricos FTS5."""
    trecho_numerico = re.search(r"\d.*", valor)
    if not trecho_numerico:
        raise ValueError(f"número inválido: {valor}")
    valor = re.sub(
        r"[/\-\s(]+[A-Za-z]{2}\)?\s*$",
        "",
        trecho_numerico.group(),
    ).translate(
        str.maketrans(
            {"O": "0", "o": "0", "I": "1", "l": "1", "S": "5", "g": "9", "G": "6"}
        )
    )
    digitos = "".join(re.findall(r"\d", valor))
    if not digitos:
        raise ValueError(f"número inválido: {valor}")
    if len(digitos) == 20:
        tamanhos = (7, 2, 4, 1, 2, 4)
        partes = []
        inicio = 0
        for tamanho in tamanhos:
            partes.append(digitos[inicio : inicio + tamanho])
            inicio += tamanho
        return " ".join(partes)
    primeira = len(digitos) % 3 or 3
    return " ".join(
        [digitos[:primeira]]
        + [digitos[indice : indice + 3] for indice in range(primeira, len(digitos), 3)]
    )


def normalizar_classe(valor: str) -> str:
    """Expande siglas de classe para melhorar a recuperação textual."""
    classes = {
        "ARESP": "AGRAVO EM RECURSO ESPECIAL",
        "RESP": "RECURSO ESPECIAL",
        "RHC": "RECURSO EM HABEAS CORPUS",
        "RMS": "RECURSO EM MANDADO DE SEGURANCA",
        "RCL": "RECLAMACAO",
        "RR": "RECURSO DE REVISTA",
    }
    return classes.get(valor.upper(), valor)


def normalizar_diploma(consulta: ConsultaLegislacao) -> str | None:
    """Converte diploma em um termo textual de fallback."""
    if consulta.numero_diploma:
        return normalizar_numero(consulta.numero_diploma)
    if not consulta.diploma:
        return None
    diploma = " ".join(re.findall(r"\w+", consulta.diploma)).casefold()
    if "constituição" in diploma or "constituicao" in diploma:
        return "Constituicao Federal"
    return None


def _numero_jurisprudencia(consulta: ConsultaJurisprudencia) -> str | None:
    return (
        consulta.numero_processo_cnj
        or consulta.numero_registro_tribunal
        or consulta.numero_classe_tribunal
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("./configs/pipeline.yaml"))
    args = parser.parse_args()
    load_dotenv()
    config = PipelineConfig.from_yaml(args.config)
    entrada = config.workdir / "03-entities"
    destino = config.workdir / "04-veracity"
    escrever_manifesto_etapa(destino, "veracity-fts")
    _veracity.executar_veracidade(
        entrada,
        destino,
        VerificadorVeracidadeFts(config.database),
    )
    print(f"{destino}: veracidade FTS concluída")
    materializar(destino, config.workdir / "predictions")


if __name__ == "__main__":
    main()
