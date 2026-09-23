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
    TipoCitacao,
    VerificadorConsulta,
)
from .utils import (
    escrever_manifesto_etapa,
    escrever_saida_documento,
    ler_documento,
    listar_resultados,
)


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

            expressao_fts = self._expressao_fts(consulta)
            filtros = ["documentos_fts MATCH ?", "d.tipo = ?", "d.natureza = ?"]
            if isinstance(consulta, ConsultaJurisprudencia):
                natureza = consulta.natureza or (
                    "sumula" if consulta.numero_sumula else "acordao"
                )
                parametros = [
                    expressao_fts,
                    TipoCitacao.JURISPRUDENCIA.value,
                    natureza,
                ]
                if consulta.numero_sumula and consulta.tribunal:
                    filtros.append("d.tribunal = ? COLLATE NOCASE")
                    parametros.append(consulta.tribunal)
            else:
                parametros = [expressao_fts, TipoCitacao.LEI.value, "dispositivo"]
            sql = (
                "SELECT d.id, d.documento_id FROM documentos_fts "
                "JOIN documentos AS d ON d.rowid = documentos_fts.rowid WHERE "
                + " AND ".join(filtros)
                + " LIMIT 10"
            )
            linhas = conexao.execute(sql, parametros).fetchall()
        return [dict(linha) for linha in linhas], sql, parametros

    @staticmethod
    def _filtro_identificadores(
        consulta: ConsultaJurisprudencia,
        colunas: set[str],
    ) -> tuple[str, list[Any], bool] | None:
        if consulta.numero_processo_cnj and "numero_processo_cnj" in colunas:
            return (
                "numero_processo_cnj = ?",
                [re.sub(r"\D", "", consulta.numero_processo_cnj)],
                True,
            )

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
    ) -> tuple[str, list[Any]] | None:
        filtros: list[str]
        parametros: list[Any]
        if isinstance(consulta, ConsultaJurisprudencia):
            identificador = cls._filtro_identificadores(consulta, colunas)
            if identificador is not None:
                filtro, parametros, eh_cnj = identificador
                filtros = ["natureza = 'acordao'", filtro]
                if not eh_cnj and consulta.tribunal:
                    filtros.append("tribunal = ? COLLATE NOCASE")
                    parametros.append(consulta.tribunal)
            elif consulta.numero_sumula is not None and "numero_sumula" in colunas:
                filtros = ["natureza = 'sumula'", "numero_sumula = ?"]
                parametros = [consulta.numero_sumula]
                if consulta.tribunal:
                    filtros.append("tribunal = ? COLLATE NOCASE")
                    parametros.append(consulta.tribunal)
            else:
                return None
        elif consulta.numero_artigo and "numero_artigo" in colunas:
            artigo = re.sub(
                r"^art(?:igo)?\.?\s*", "", consulta.numero_artigo, flags=re.IGNORECASE
            )
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
            if consulta.ano_diploma is not None:
                filtros.append("ano_diploma = ?")
                parametros.append(consulta.ano_diploma)
        else:
            return None
        sql = (
            "SELECT id, documento_id FROM documentos WHERE "
            + " AND ".join(filtros)
            + " LIMIT 10"
        )
        return sql, parametros

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

    @classmethod
    def _expressao_fts(
        cls,
        consulta: ConsultaJurisprudencia | ConsultaLegislacao,
    ) -> str:
        valores = cls._valores_fts(consulta)
        if (
            isinstance(consulta, ConsultaJurisprudencia)
            and cls._numero_jurisprudencia(consulta)
            and consulta.cadeia_recursal
            and len(consulta.cadeia_recursal) > 1
        ):
            classes = [
                cls._normalizar_classe(item) for item in consulta.cadeia_recursal
            ]
            termos = " ".join(f'"{valor}"' for valor in [*classes, valores[0]])
            return f"NEAR({termos}, 20)"
        return " AND ".join(f'"{valor}"' for valor in valores)

    @classmethod
    def _valores_fts(
        cls,
        consulta: ConsultaJurisprudencia | ConsultaLegislacao,
    ) -> list[str]:
        if isinstance(consulta, ConsultaJurisprudencia):
            numero = cls._numero_jurisprudencia(consulta)
            if numero:
                return [cls._normalizar_numero(numero)]
            if consulta.numero_sumula is not None:
                valores = [
                    "sumula",
                    cls._normalizar_numero(str(consulta.numero_sumula)),
                ]
                if consulta.sumula_vinculante:
                    valores.insert(1, "vinculante")
                return valores
            raise ValueError("jurisprudência sem processo ou súmula")
        if not consulta.numero_artigo:
            raise ValueError("legislação sem número de artigo")
        valores = [f"Artigo {cls._normalizar_numero(consulta.numero_artigo)}"]
        diploma = cls._normalizar_diploma(consulta)
        if diploma:
            valores.append(diploma)
        return valores

    @staticmethod
    def _normalizar_numero(valor: str) -> str:
        trecho_numerico = re.search(r"\d.*", valor)
        if not trecho_numerico:
            raise ValueError(f"número inválido: {valor}")
        valor = re.sub(
            r"[/\-\s(]+[A-Za-z]{2}\)?\s*$",
            "",
            trecho_numerico.group(),
        ).translate(
            str.maketrans(
                {
                    "O": "0",
                    "o": "0",
                    "I": "1",
                    "l": "1",
                    "S": "5",
                    "g": "9",
                    "G": "6",
                }
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
            + [
                digitos[indice : indice + 3]
                for indice in range(primeira, len(digitos), 3)
            ]
        )

    @staticmethod
    def _normalizar_classe(valor: str) -> str:
        classes = {
            "ARESP": "AGRAVO EM RECURSO ESPECIAL",
            "RESP": "RECURSO ESPECIAL",
            "RHC": "RECURSO EM HABEAS CORPUS",
            "RMS": "RECURSO EM MANDADO DE SEGURANCA",
            "RCL": "RECLAMACAO",
            "RR": "RECURSO DE REVISTA",
        }
        return classes.get(valor.upper(), valor)

    @classmethod
    def _normalizar_diploma(cls, consulta: ConsultaLegislacao) -> str | None:
        if consulta.numero_diploma:
            numero = cls._normalizar_numero(consulta.numero_diploma)
            return " ".join(
                [numero, str(consulta.ano_diploma)]
                if consulta.ano_diploma is not None
                else [numero]
            )
        if not consulta.diploma:
            return None
        diploma = " ".join(re.findall(r"\w+", consulta.diploma)).casefold()
        if "constituição" in diploma or "constituicao" in diploma:
            return "Constituicao Federal"
        return None

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
    from .orchestrator import materializar

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("pipeline.yaml"))
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
