"""Etapa 3: extrai entidades usadas para consultar a base canônica."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import AsyncOpenAI, OpenAIError, omit
from openai.types.chat import ChatCompletionMessageParam
from pydantic import ValidationError
from tqdm import tqdm

from .contracts import (
    CandidatoAnalisado,
    CandidatoCitacao,
    CandidatoEntidades,
    ConsultaJurisprudencia,
    ConsultaJurisprudenciaAgente,
    ConsultaLegislacao,
    Contract,
    DocumentoCompletude,
    DocumentoEntidades,
    ExtratorEntidadesAsync,
    ModelConfig,
    PipelineConfig,
    TipoCitacao,
)
from .utils import (
    criar_auditoria,
    escrever_manifesto_etapa,
    escrever_saida_documento,
    ler_documento,
    listar_resultados,
    normalizar_numero_cnj,
    normalizar_relator,
)

_PROMPT_JURISPRUDENCIA = """Extraia os campos de identificação desta citação de
jurisprudência brasileira.

Preencha somente dados explícitos ou decorrentes de abreviações
jurídicas inequívocas."""

_LOG = logging.getLogger(__name__)

_PROMPT_LEI = """Extraia os campos de identificação desta citação de legislação
brasileira.

Preencha somente dados explícitos ou decorrentes de abreviações
jurídicas inequívocas."""


def _prompt_veracidade(tipo: TipoCitacao) -> str:
    return _PROMPT_JURISPRUDENCIA if tipo == TipoCitacao.JURISPRUDENCIA else _PROMPT_LEI


def _contrato_consulta(tipo: TipoCitacao) -> type[Contract]:
    return (
        ConsultaJurisprudenciaAgente
        if tipo == TipoCitacao.JURISPRUDENCIA
        else ConsultaLegislacao
    )


class AgenteExtratorEntidades:
    """Extrai do trecho os campos usados na consulta canônica."""

    def __init__(
        self,
        cliente: AsyncOpenAI,
        config: ModelConfig,
        relatores: Mapping[str, str] | None = None,
    ) -> None:
        self._cliente = cliente
        self._config = config
        self._relatores = relatores or {}

    async def extrair_auditada_async(
        self,
        candidato: CandidatoCitacao,
    ) -> tuple[ConsultaJurisprudencia | ConsultaLegislacao, dict[str, Any]]:
        if not isinstance(self._cliente, AsyncOpenAI):
            raise TypeError("extrair_auditada_async() exige cliente assíncrono")
        requisicao = self._requisicao(candidato)
        ultimo_erro: Exception | None = None
        for tentativa in range(1, self._config.max_retries + 1):
            try:
                async with asyncio.timeout(self._config.request_timeout_seconds):
                    resposta = await self._cliente.chat.completions.parse(**requisicao)
                return self._finalizar(requisicao, resposta, candidato, tentativa)
            except (
                TimeoutError,
                OpenAIError,
                ValidationError,
                ValueError,
                IndexError,
                AttributeError,
            ) as erro:
                ultimo_erro = erro
                _LOG.warning(
                    "entidades: tentativa %d/%d falhou para %r: %s",
                    tentativa,
                    self._config.max_retries,
                    candidato.trecho[:80],
                    erro,
                )
                if tentativa < self._config.max_retries:
                    await asyncio.sleep(tentativa)
        raise RuntimeError(
            f"retries esgotados para {candidato.trecho[:80]!r}: {ultimo_erro}"
        ) from ultimo_erro

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
        candidato: CandidatoCitacao,
        tentativa: int,
    ) -> tuple[ConsultaJurisprudencia | ConsultaLegislacao, dict[str, Any]]:
        consulta = resposta.choices[0].message.parsed
        if consulta is None:
            raise RuntimeError("o modelo não retornou entidades estruturadas")
        if candidato.tipo == TipoCitacao.JURISPRUDENCIA:
            numero_cnj = normalizar_numero_cnj(
                consulta.numero_processo_cnj, candidato.trecho
            )
            numero_sumula = consulta.numero_sumula
            natureza = (
                "sumula"
                if numero_sumula is not None
                else consulta.natureza or "acordao"
            )
            vinculante = consulta.sumula_vinculante
            if natureza == "sumula" and vinculante is None:
                vinculante = bool(
                    re.search(
                        r"\bs[uú]mula\s+vinculante\b", candidato.trecho, re.IGNORECASE
                    )
                )
            dados = consulta.model_dump()
            dados.update(
                natureza=natureza,
                numero_processo_cnj=numero_cnj,
                relator_norm=normalizar_relator(consulta.relator, self._relatores),
                sumula_vinculante=vinculante,
            )
            consulta = ConsultaJurisprudencia.model_validate(dados)
        else:
            consulta = ConsultaLegislacao.model_validate(consulta)
        auditoria = criar_auditoria(
            requisicao,
            resposta,
            _contrato_consulta(candidato.tipo),
            tentativa=tentativa,
            campos_extraidos=consulta.model_dump(mode="json"),
        )
        return consulta, auditoria


async def executar_entities_async(
    input_file: Path,
    output_file: Path,
    extrator: ExtratorEntidadesAsync,
    max_concurrency: int,
) -> None:
    arquivos = listar_resultados(input_file)
    semaforo = asyncio.Semaphore(max_concurrency)
    progresso = tqdm(total=len(arquivos), desc="Extraindo entidades", unit="documento")

    async def processar(arquivo: Path) -> None:
        documento = ler_documento(arquivo, DocumentoCompletude)

        async def extrair(
            analisado: CandidatoAnalisado,
        ) -> tuple[CandidatoEntidades, dict[str, Any]]:
            async with semaforo:
                campos, chamada = await extrator.extrair_auditada_async(
                    analisado.candidato
                )
            return CandidatoEntidades(
                candidato=analisado.candidato,
                completude=analisado.completude,
                campos_extraidos=campos,
            ), chamada

        resultados = await asyncio.gather(
            *(extrair(analisado) for analisado in documento.candidatos)
        )
        escrever_saida_documento(
            output_file,
            DocumentoEntidades(
                documento_id=documento.documento_id,
                candidatos=[item[0] for item in resultados],
            ),
            [item[1] for item in resultados],
        )
        progresso.update()

    try:
        await asyncio.gather(*(processar(arquivo) for arquivo in arquivos))
    finally:
        progresso.close()


async def _executar_entities_async(config: PipelineConfig, destino: Path) -> None:
    etapa = config.entities
    caminho_relatores = config.database.parent / "relatores_padronizacao.json"
    relatores = json.loads(caminho_relatores.read_text(encoding="utf-8"))
    async with AsyncOpenAI(base_url=etapa.base_url) as cliente:
        await executar_entities_async(
            config.workdir / "02-completeness",
            destino,
            AgenteExtratorEntidades(cliente, etapa, relatores),
            etapa.max_concurrency,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("pipeline.yaml"))
    args = parser.parse_args()
    load_dotenv()
    config = PipelineConfig.from_yaml(args.config)
    destino = config.workdir / "03-entities"
    etapa = config.entities
    escrever_manifesto_etapa(destino, "entities", etapa)
    asyncio.run(_executar_entities_async(config, destino))
    print(f"{destino}: entidades concluídas")


if __name__ == "__main__":
    main()
