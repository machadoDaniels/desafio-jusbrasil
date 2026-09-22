"""Etapa 3: extrai chaves de busca e verifica citações na base SQLite."""

from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import AsyncOpenAI, OpenAI, omit
from openai.types.chat import ChatCompletionMessageParam
from tqdm import tqdm

from .contracts import (
    CandidatoAnalisado,
    CandidatoCitacao,
    CandidatoClassificado,
    Classificacao,
    ClassificadorVeracidade,
    ClassificadorVeracidadeAsync,
    ConsultaJurisprudencia,
    ConsultaLegislacao,
    DocumentoClassificado,
    DocumentoCompletude,
    ModelConfig,
    PipelineConfig,
    ResultadoVeracidade,
    TipoCitacao,
)
from .utils import (
    criar_auditoria,
    escrever_manifesto_etapa,
    escrever_saida_documento,
    ler_documento,
    listar_resultados,
)

_PROMPT_JURISPRUDENCIA = """Extraia de uma citação de jurisprudência brasileira os dados necessários
para uma única consulta SQLite FTS5. Não escreva SQL e não avalie se a citação é
verdadeira.

Em valores_fts, retorne partes distintivas que devem aparecer simultaneamente no
documento, já normalizadas sem pontuação problemática para FTS5. Preserve como um
único valor os números compostos, com seus grupos separados por espaços. Inclua a
UF quando ela fizer parte do identificador. Para súmulas e temas, inclua sua
denominação e número. Não inclua palavras genéricas nem crie variações alternativas.

Defina natureza como acordao para processos e recursos ou sumula para súmulas.
Extraia tribunal, ano do julgamento e relator somente quando estiverem explícitos.
O ano dentro de um número processual não é o ano do julgamento."""

_PROMPT_LEI = """Extraia de uma citação de legislação brasileira os dados necessários para
uma única consulta SQLite FTS5. Não escreva SQL e não avalie se a citação é
verdadeira.

Em valores_fts, retorne partes distintivas que devem aparecer simultaneamente no
dispositivo, já normalizadas sem pontuação problemática para FTS5. Inclua o número
do artigo ou dispositivo e o diploma legal identificável. Inclua inciso, parágrafo
ou alínea apenas quando ajudarem a individualizar o dispositivo. Para diplomas
abreviados como CPC, CF e CLT, não inclua a sigla nem uma expansão incerta; use o
número da lei somente se ele estiver explícito no trecho. Não crie variações
alternativas."""


def _prompt_veracidade(tipo: TipoCitacao) -> str:
    return _PROMPT_JURISPRUDENCIA if tipo == TipoCitacao.JURISPRUDENCIA else _PROMPT_LEI


def _contrato_consulta(
    tipo: TipoCitacao,
) -> type[ConsultaJurisprudencia | ConsultaLegislacao]:
    return (
        ConsultaJurisprudencia
        if tipo == TipoCitacao.JURISPRUDENCIA
        else ConsultaLegislacao
    )


class VerificadorVeracidade:
    """Extrai uma consulta estruturada e a executa deterministicamente."""

    def __init__(
        self,
        cliente: OpenAI | AsyncOpenAI,
        config: ModelConfig,
        database: Path,
    ) -> None:
        self._cliente = cliente
        self._config = config
        self._database = database

    def classificar_auditada(
        self,
        candidato: CandidatoCitacao,
    ) -> tuple[ResultadoVeracidade, dict[str, Any]]:
        if not isinstance(self._cliente, OpenAI):
            raise TypeError("classificar_auditada() exige cliente síncrono")
        requisicao = self._requisicao(candidato)
        resposta = self._cliente.chat.completions.parse(**requisicao)
        return self._finalizar(requisicao, resposta, candidato.tipo)

    async def classificar_auditada_async(
        self,
        candidato: CandidatoCitacao,
    ) -> tuple[ResultadoVeracidade, dict[str, Any]]:
        if not isinstance(self._cliente, AsyncOpenAI):
            raise TypeError("classificar_auditada_async() exige cliente assíncrono")
        requisicao = self._requisicao(candidato)
        resposta = await self._cliente.chat.completions.parse(**requisicao)
        return self._finalizar(requisicao, resposta, candidato.tipo)

    @staticmethod
    def _mensagens(candidato: CandidatoCitacao) -> list[ChatCompletionMessageParam]:
        return [
            {"role": "system", "content": _prompt_veracidade(candidato.tipo)},
            {"role": "user", "content": f"Trecho original:\n{candidato.trecho}"},
        ]

    def _requisicao(self, candidato: CandidatoCitacao) -> dict[str, Any]:
        requisicao: dict[str, Any] = {
            "model": self._config.model,
            "temperature": self._config.temperature
            if self._config.temperature is not None
            else omit,
            "top_p": self._config.top_p if self._config.top_p is not None else omit,
            "messages": self._mensagens(candidato),
            "response_format": _contrato_consulta(candidato.tipo),
            "reasoning_effort": self._config.reasoning_effort
            if self._config.reasoning_effort is not None
            else omit,
        }
        if self._config.top_k is not None:
            requisicao["extra_body"] = {"top_k": self._config.top_k}
        return requisicao

    def _finalizar(
        self,
        requisicao: dict[str, Any],
        resposta: Any,
        tipo: TipoCitacao,
    ) -> tuple[ResultadoVeracidade, dict[str, Any]]:
        consulta = resposta.choices[0].message.parsed
        if consulta is None:
            raise RuntimeError("o modelo não retornou uma consulta estruturada")
        registros, sql, parametros = self._consultar_base(consulta)
        resultado = self._classificar(registros)
        auditoria = criar_auditoria(
            requisicao,
            resposta,
            _contrato_consulta(tipo),
            consulta.model_dump(mode="json"),
            sql=sql,
            parametros=parametros,
            registros=registros,
            resultado=resultado.model_dump(mode="json"),
        )
        return resultado, auditoria

    def _consultar_base(
        self,
        consulta: ConsultaJurisprudencia | ConsultaLegislacao,
    ) -> tuple[list[dict[str, Any]], str, list[Any]]:
        valores = [valor.strip() for valor in consulta.valores_fts if valor.strip()]
        if not valores:
            raise ValueError("valores_fts deve conter ao menos um valor não vazio")
        expressao_fts = " AND ".join(
            f'"{valor.replace(chr(34), chr(34) * 2)}"' for valor in valores
        )
        filtros = ["documentos_fts MATCH ?", "d.tipo = ?", "d.natureza = ?"]
        if isinstance(consulta, ConsultaJurisprudencia):
            parametros: list[Any] = [
                expressao_fts,
                TipoCitacao.JURISPRUDENCIA.value,
                consulta.natureza,
            ]
            if consulta.tribunal is not None:
                filtros.append("d.tribunal = ? COLLATE NOCASE")
                parametros.append(consulta.tribunal)
            if consulta.relator is not None:
                filtros.append("d.relator LIKE ? COLLATE NOCASE")
                parametros.append(f"%{consulta.relator}%")
            if consulta.ano is not None:
                filtros.append("d.ano = ?")
                parametros.append(consulta.ano)
        else:
            parametros = [expressao_fts, TipoCitacao.LEI.value, "dispositivo"]
        sql = (
            "SELECT d.id, d.documento_id FROM documentos_fts "
            "JOIN documentos AS d ON d.rowid = documentos_fts.rowid WHERE "
            + " AND ".join(filtros)
            + " LIMIT 2"
        )
        uri = f"file:{self._database}?mode=ro"
        with sqlite3.connect(uri, uri=True) as conexao:
            conexao.row_factory = sqlite3.Row
            conexao.execute("PRAGMA query_only = ON")
            linhas = conexao.execute(sql, parametros).fetchall()
        return [dict(linha) for linha in linhas], sql, parametros

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
    classificador: ClassificadorVeracidade,
) -> None:
    arquivos = listar_resultados(input_file)
    with tqdm(
        total=len(arquivos), desc="Verificando citações", unit="documento"
    ) as progresso:
        for arquivo in arquivos:
            documento = ler_documento(arquivo, DocumentoCompletude)
            candidatos = []
            chamadas = []
            for analisado in documento.candidatos:
                if analisado.completude.completa:
                    resultado, chamada = classificador.classificar_auditada(
                        analisado.candidato
                    )
                    chamadas.append(chamada)
                else:
                    resultado = _resultado_incompleto()
                candidatos.append(
                    CandidatoClassificado(
                        candidato=analisado.candidato,
                        completude=analisado.completude,
                        veracidade=resultado,
                    )
                )
            escrever_saida_documento(
                output_file,
                DocumentoClassificado(
                    documento_id=documento.documento_id,
                    candidatos=candidatos,
                ),
                chamadas,
            )
            progresso.update()


async def executar_veracidade_async(
    input_file: Path,
    output_file: Path,
    classificador: ClassificadorVeracidadeAsync,
    max_concurrency: int,
) -> None:
    arquivos = listar_resultados(input_file)
    semaforo = asyncio.Semaphore(max_concurrency)
    progresso = tqdm(total=len(arquivos), desc="Verificando citações", unit="documento")

    async def processar(arquivo: Path) -> None:
        documento = ler_documento(arquivo, DocumentoCompletude)

        async def classificar(
            analisado: CandidatoAnalisado,
        ) -> tuple[CandidatoClassificado, dict[str, Any] | None]:
            if analisado.completude.completa:
                async with semaforo:
                    (
                        resultado,
                        auditoria,
                    ) = await classificador.classificar_auditada_async(
                        analisado.candidato
                    )
            else:
                auditoria = None
                resultado = _resultado_incompleto()
            return (
                CandidatoClassificado(
                    candidato=analisado.candidato,
                    completude=analisado.completude,
                    veracidade=resultado,
                ),
                auditoria,
            )

        resultados = await asyncio.gather(
            *(classificar(analisado) for analisado in documento.candidatos)
        )
        escrever_saida_documento(
            output_file,
            DocumentoClassificado(
                documento_id=documento.documento_id,
                candidatos=[item[0] for item in resultados],
            ),
            [item[1] for item in resultados if item[1] is not None],
        )
        progresso.update()

    try:
        await asyncio.gather(*(processar(arquivo) for arquivo in arquivos))
    finally:
        progresso.close()


async def _executar_veracidade_async(
    config: PipelineConfig,
    entrada: Path,
    destino: Path,
) -> None:
    etapa = config.veracity
    async with AsyncOpenAI(base_url=etapa.base_url) as cliente:
        await executar_veracidade_async(
            entrada,
            destino,
            VerificadorVeracidade(cliente, etapa, config.database),
            etapa.max_concurrency,
        )


def main() -> None:
    from .orchestrator import materializar

    load_dotenv()
    config = PipelineConfig.from_yaml(Path("pipeline.yaml"))
    entrada = config.workdir / "02-completeness"
    destino = config.workdir / "03-veracity"
    etapa = config.veracity
    escrever_manifesto_etapa(destino, "veracity", etapa)
    if etapa.async_requests:
        asyncio.run(_executar_veracidade_async(config, entrada, destino))
    else:
        executar_veracidade(
            entrada,
            destino,
            VerificadorVeracidade(
                OpenAI(base_url=etapa.base_url),
                etapa,
                config.database,
            ),
        )
    print(f"{destino}: veracidade concluída")

    materializar(destino, config.workdir / "predictions")


if __name__ == "__main__":
    main()
