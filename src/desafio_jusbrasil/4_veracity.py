"""Etapa 4: consulta a base SQLite e classifica a veracidade das citações."""

from __future__ import annotations

import argparse
import re
import sqlite3
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from tqdm import tqdm

from .contracts import (
    CandidatoClassificado,
    Classificacao,
    ConsultaJurisprudencia,
    ConsultaLegislacao,
    DocumentoClassificado,
    DocumentoEntidades,
    PipelineConfig,
    ResultadoVeracidade,
    VerificadorConsulta,
)
from .utils import (
    escrever_manifesto_etapa,
    escrever_saida_documento,
    ler_documento,
    listar_resultados,
    materializar,
)

type ConsultaSql = tuple[str, list[Any]]


class VerificadorVeracidade:
    """Monta e executa a consulta SQL sem chamar modelos."""

    def __init__(self, database: Path) -> None:
        self._database = database

    def verificar(
        self,
        consulta: ConsultaJurisprudencia | ConsultaLegislacao,
    ) -> tuple[ResultadoVeracidade, dict[str, Any]]:
        try:
            registros, sql, parametros = self._consultar_base(consulta)
            resultado = self._classificar(registros)
        except ValueError as erro:
            registros, sql, parametros = [], None, []
            resultado = ResultadoVeracidade(
                classificacao=Classificacao.INCOMPLETA,
                justificativa=f"Não foi possível montar a consulta: {erro}.",
            )
        return resultado, {
            "campos_extraidos": consulta.model_dump(mode="json"),
            "sql": sql,
            "parametros": parametros,
            "registros": registros,
            "resultado": resultado.model_dump(mode="json"),
        }

    @staticmethod
    def _numero_jurisprudencia(consulta: ConsultaJurisprudencia) -> str | None:
        return (
            consulta.numero_processo_cnj
            or consulta.numero_registro_tribunal
            or consulta.numero_classe_tribunal
        )

    def _consultar_base(
        self,
        consulta: ConsultaJurisprudencia | ConsultaLegislacao,
    ) -> tuple[list[dict[str, Any]], str, list[Any]]:
        # 1. Validação mínima antes de abrir a base.
        if isinstance(consulta, ConsultaJurisprudencia):
            if (
                not self._numero_jurisprudencia(consulta)
                and consulta.numero_sumula is None
            ):
                raise ValueError("jurisprudência sem identificador ou súmula")
        elif not consulta.numero_artigo:
            raise ValueError("legislação sem número de artigo")

        uri = f"file:{self._database}?mode=ro"
        with sqlite3.connect(uri, uri=True) as conexao:
            conexao.row_factory = sqlite3.Row
            conexao.execute("PRAGMA query_only = ON")
            colunas = {
                linha[1] for linha in conexao.execute("PRAGMA table_info(documentos)")
            }

            # 2. Busca estruturada: identificadores, súmula ou dispositivo legal.
            consulta_estruturada = self._consulta_estruturada(consulta, colunas)
            if consulta_estruturada is not None:
                sql, parametros = consulta_estruturada
                linhas = conexao.execute(sql, parametros).fetchall()
                if (
                    len(linhas) > 1
                    and isinstance(consulta, ConsultaJurisprudencia)
                    and self._numero_jurisprudencia(consulta)
                ):
                    desambiguada = self._consulta_processo_desambiguada(
                        consulta, colunas
                    )
                    if desambiguada is not None:
                        sql_desambiguada, parametros_desambiguados = desambiguada
                        linhas_desambiguadas = conexao.execute(
                            sql_desambiguada, parametros_desambiguados
                        ).fetchall()
                        if linhas_desambiguadas:
                            return (
                                [dict(linha) for linha in linhas_desambiguadas],
                                sql_desambiguada,
                                parametros_desambiguados,
                            )
                return [dict(linha) for linha in linhas], sql, parametros

            raise ValueError("nenhuma consulta estruturada disponível")

    # Busca estruturada -----------------------------------------------------

    @staticmethod
    def _filtro_identificadores(
        consulta: ConsultaJurisprudencia,
        colunas: set[str],
    ) -> tuple[str, list[Any], bool] | None:
        # Prioridade 1: CNJ é um identificador exclusivo.
        if consulta.numero_processo_cnj and "numero_processo_cnj" in colunas:
            return (
                "numero_processo_cnj = ?",
                [re.sub(r"\D", "", consulta.numero_processo_cnj)],
                True,
            )

        # Prioridade 2: número de classe e de registro são consultados cruzadamente.
        valores = list(
            dict.fromkeys(
                re.sub(r"\D", "", valor)
                for valor in (
                    consulta.numero_classe_tribunal,
                    consulta.numero_registro_tribunal,
                )
                if valor
            )
        )
        colunas_identificadoras = [
            coluna
            for coluna in ("numero_classe_tribunal", "numero_registro_tribunal")
            if coluna in colunas
        ]
        if not valores or not colunas_identificadoras:
            return None

        marcadores = ", ".join("?" for _ in valores)
        filtro = (
            "("
            + " OR ".join(
                f"{coluna} IN ({marcadores})" for coluna in colunas_identificadoras
            )
            + ")"
        )
        return filtro, valores * len(colunas_identificadoras), False

    @classmethod
    def _consulta_estruturada(
        cls,
        consulta: ConsultaJurisprudencia | ConsultaLegislacao,
        colunas: set[str],
    ) -> ConsultaSql | None:
        """Escolhe a primeira rota estruturada disponível para a consulta."""
        if isinstance(consulta, ConsultaJurisprudencia):
            return cls._consulta_jurisprudencia_estruturada(consulta, colunas)
        return cls._consulta_legislacao_estruturada(consulta, colunas)

    @classmethod
    def _consulta_jurisprudencia_estruturada(
        cls, consulta: ConsultaJurisprudencia, colunas: set[str]
    ) -> ConsultaSql | None:
        # 1. CNJ: chave exclusiva, sem filtro adicional.
        if consulta.numero_processo_cnj and "numero_processo_cnj" in colunas:
            return cls._montar_consulta_documentos(
                ["natureza = 'acordao'", "numero_processo_cnj = ?"],
                [re.sub(r"\D", "", consulta.numero_processo_cnj)],
            )

        # 2. Número de classe ou de registro: busca cruzada nas duas colunas.
        identificador = cls._filtro_identificadores(consulta, colunas)
        if identificador is not None:
            filtro, parametros, _ = identificador
            filtros = ["natureza = 'acordao'", filtro]
            if consulta.tribunal:
                filtros.append("tribunal = ? COLLATE NOCASE")
                parametros.append(consulta.tribunal)
            return cls._montar_consulta_documentos(filtros, parametros)

        # 3. Súmula: número e tribunal, quando disponível.
        if consulta.numero_sumula is not None and "numero_sumula" in colunas:
            filtros = ["natureza = 'sumula'", "numero_sumula = ?"]
            parametros = [consulta.numero_sumula]
            if consulta.tribunal:
                filtros.append("tribunal = ? COLLATE NOCASE")
                parametros.append(consulta.tribunal)
            return cls._montar_consulta_documentos(filtros, parametros)
        return None

    @classmethod
    def _consulta_legislacao_estruturada(
        cls, consulta: ConsultaLegislacao, colunas: set[str]
    ) -> ConsultaSql | None:
        # Artigo: aceita as três grafias armazenáveis do ordinal.
        if not consulta.numero_artigo or "numero_artigo" not in colunas:
            return None
        artigo = consulta.numero_artigo

        artigo_sem_ordinal = artigo.rstrip("º°")
        variantes = list(
            dict.fromkeys([artigo, artigo_sem_ordinal, artigo_sem_ordinal + "º"])
        )
        filtros = [
            "natureza = 'dispositivo'",
            f"numero_artigo IN ({', '.join('?' for _ in variantes)})",
        ]
        parametros = variantes
        if consulta.numero_diploma:
            filtros.append("numero_diploma = ?")
            parametros.append(re.sub(r"\D", "", consulta.numero_diploma))
        elif consulta.diploma:
            filtros.append("diploma = ?")
            parametros.append(consulta.diploma)
        return cls._montar_consulta_documentos(filtros, parametros)

    @staticmethod
    def _montar_consulta_documentos(
        filtros: list[str], parametros: list[Any]
    ) -> ConsultaSql:
        sql = (
            "SELECT id, documento_id FROM documentos WHERE "
            + " AND ".join(filtros)
            + " LIMIT 10"
        )
        return sql, parametros

    # Desambiguação estruturada ---------------------------------------------

    @classmethod
    def _consulta_processo_desambiguada(
        cls,
        consulta: ConsultaJurisprudencia,
        colunas: set[str],
    ) -> tuple[str, list[Any]] | None:
        identificador = cls._filtro_identificadores(consulta, colunas)
        if identificador is None:
            return None
        filtro, parametros, _ = identificador
        filtros = ["natureza = 'acordao'", filtro]
        opcionais = (
            ("tribunal", consulta.tribunal),
            ("uf", consulta.uf),
            ("ano", consulta.ano),
            ("relator_norm", consulta.relator_norm),
        )
        for coluna, valor in opcionais:
            if valor is not None and coluna in colunas:
                filtros.append(f"{coluna} = ? COLLATE NOCASE")
                parametros.append(valor)
        if len(filtros) == 2:
            return None
        return (
            "SELECT id, documento_id FROM documentos WHERE "
            + " AND ".join(filtros)
            + " LIMIT 10",
            parametros,
        )

    @staticmethod
    def _classificar(registros: list[dict[str, Any]]) -> ResultadoVeracidade:
        if not registros:
            return ResultadoVeracidade(
                classificacao=Classificacao.INVENTADA,
                justificativa="A consulta estruturada não encontrou registro canônico.",
            )
        if len(registros) > 1:
            return ResultadoVeracidade(
                classificacao=Classificacao.INCOMPLETA,
                justificativa="A consulta estruturada encontrou mais de um registro canônico.",
            )
        return ResultadoVeracidade(
            classificacao=Classificacao.REAL,
            id_canonico=registros[0]["id"],
            justificativa="A consulta estruturada encontrou um único registro canônico.",
        )


def _resultado_incompleto() -> ResultadoVeracidade:
    return ResultadoVeracidade(
        classificacao=Classificacao.INCOMPLETA,
        justificativa="Citação sem identificador pesquisável.",
    )


def executar_veracidade(
    input_file: Path,
    output_file: Path,
    verificador: VerificadorConsulta,
) -> None:
    arquivos = listar_resultados(input_file)
    for arquivo in tqdm(arquivos, desc="Verificando citações", unit="documento"):
        documento = ler_documento(arquivo, DocumentoEntidades)
        candidatos = []
        verificacoes = []
        for item in documento.candidatos:
            if item.completude.completa and item.campos_extraidos is not None:
                resultado, verificacao = verificador.verificar(item.campos_extraidos)
                verificacoes.append(verificacao)
            else:
                resultado = _resultado_incompleto()
            candidatos.append(
                CandidatoClassificado(
                    candidato=item.candidato,
                    completude=item.completude,
                    veracidade=resultado,
                )
            )
        escrever_saida_documento(
            output_file,
            DocumentoClassificado(
                documento_id=documento.documento_id,
                candidatos=candidatos,
            ),
            verificacoes,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("./configs/pipeline.yaml"))
    args = parser.parse_args()
    load_dotenv()
    config = PipelineConfig.from_yaml(args.config)
    entrada = config.workdir / "03-entities"
    destino = config.workdir / "04-veracity"
    escrever_manifesto_etapa(destino, "veracity")
    executar_veracidade(
        entrada,
        destino,
        VerificadorVeracidade(config.database),
    )
    print(f"{destino}: veracidade concluída")
    materializar(destino, config.workdir / "predictions")


if __name__ == "__main__":
    main()
