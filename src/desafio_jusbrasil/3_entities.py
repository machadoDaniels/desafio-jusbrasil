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
    ConsultaLegislacaoAgente,
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
    criar_auditoria_erro,
    escrever_manifesto_etapa,
    escrever_saida_documento,
    ler_documento,
    listar_resultados,
    normalizar_numero_cnj,
    normalizar_relator,
)

_PROMPT_JURISPRUDENCIA = """Extraia os campos de identificação desta citação de
jurisprudência brasileira.

Preencha somente dados explícitos ou decorrentes de abreviações jurídicas inequívocas.

Exemplo completo — a classe principal é o recurso-base, não o recurso incidental mais externo:

Trecho: EDcl no AgInt no Agravo em Recurso Especial nº 1904603/TO

Resposta:
{
  "natureza": "acordao",
  "numero_processo_cnj": null,
  "numero_classe_tribunal": "1904603",
  "numero_registro_tribunal": null,
  "classe_processual": "AREsp — Agravo em Recurso Especial",
  "cadeia_recursal": [
    "EDcl — Embargos de Declaração",
    "AgInt — Agravo Interno",
    "AREsp — Agravo em Recurso Especial"
  ],
  "tribunal": null,
  "uf": "TO",
  "ano": null,
  "relator": null,
  "numero_sumula": null,
  "sumula_vinculante": null
}"""

_LOG = logging.getLogger(__name__)


class ErroExtracaoEntidades(RuntimeError):
    """Falha final de uma citação com as tentativas auditadas."""

    def __init__(self, mensagem: str, auditorias: list[dict[str, Any]]) -> None:
        super().__init__(mensagem)
        self.auditorias = auditorias


_PROMPT_LEI = """Extraia os campos de identificação desta citação de legislação
brasileira.

Preencha somente dados explícitos ou decorrentes de abreviações jurídicas inequívocas.

Exemplo completo — não use `Lei` como fallback quando o trecho identifica um diploma específico:

Trecho: art. 1.134 da Lei nº 13.105/2015

Resposta:
{
  "numero_artigo": "1134",
  "diploma": "CPC — Código de Processo Civil",
  "numero_diploma": "13105"
}"""


def _prompt_veracidade(tipo: TipoCitacao) -> str:
    return _PROMPT_JURISPRUDENCIA if tipo == TipoCitacao.JURISPRUDENCIA else _PROMPT_LEI


def _contrato_consulta(tipo: TipoCitacao) -> type[Contract]:
    return (
        ConsultaJurisprudenciaAgente
        if tipo == TipoCitacao.JURISPRUDENCIA
        else ConsultaLegislacaoAgente
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
    ) -> tuple[ConsultaJurisprudencia | ConsultaLegislacao, list[dict[str, Any]]]:
        if not isinstance(self._cliente, AsyncOpenAI):
            raise TypeError("extrair_auditada_async() exige cliente assíncrono")
        requisicao = self._requisicao(candidato)
        auditorias: list[dict[str, Any]] = []
        ultimo_erro: Exception | None = None
        for tentativa in range(1, self._config.max_retries + 1):
            resposta = None
            try:
                async with asyncio.timeout(self._config.request_timeout_seconds):
                    resposta = await self._cliente.chat.completions.parse(**requisicao)
                consulta, auditoria = self._finalizar(
                    requisicao, resposta, candidato, tentativa
                )
                return consulta, [*auditorias, auditoria]
            except (
                TimeoutError,
                OpenAIError,
                ValidationError,
                ValueError,
                IndexError,
                AttributeError,
            ) as erro:
                ultimo_erro = erro
                auditorias.append(
                    criar_auditoria_erro(
                        requisicao,
                        _contrato_consulta(candidato.tipo),
                        erro,
                        tentativa,
                        resposta,
                    )
                )
                _LOG.warning(
                    "entidades: tentativa %d/%d falhou para %r: %s",
                    tentativa,
                    self._config.max_retries,
                    candidato.trecho[:80],
                    erro,
                )
                if tentativa < self._config.max_retries:
                    await asyncio.sleep(tentativa)
        raise ErroExtracaoEntidades(
            f"retries esgotados para {candidato.trecho[:80]!r}: {ultimo_erro}",
            auditorias,
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
            consulta = ConsultaLegislacao.model_validate(consulta.model_dump())
        auditoria = criar_auditoria(
            requisicao,
            resposta,
            _contrato_consulta(candidato.tipo),
            tentativa=tentativa,
            campos_extraidos=consulta.model_dump(mode="json"),
        )
        return consulta, auditoria


async def _extrair_entidade(
    analisado: CandidatoAnalisado,
    extrator: ExtratorEntidadesAsync,
    semaforo: asyncio.Semaphore,
) -> tuple[CandidatoEntidades, list[dict[str, Any]]]:
    async with semaforo:
        campos, chamadas = await extrator.extrair_auditada_async(analisado.candidato)
    return CandidatoEntidades(
        candidato=analisado.candidato,
        completude=analisado.completude,
        campos_extraidos=campos,
    ), chamadas


async def _processar_documento_entities(
    arquivo: Path,
    output_file: Path,
    extrator: ExtratorEntidadesAsync,
    semaforo: asyncio.Semaphore,
    progresso: Any,
) -> None:
    documento = ler_documento(arquivo, DocumentoCompletude)
    candidatos: list[CandidatoEntidades] = []
    chamadas: list[dict[str, Any]] = []
    if not documento.candidatos:
        escrever_saida_documento(
            output_file,
            DocumentoEntidades(documento_id=documento.documento_id, candidatos=[]),
            [],
        )
    for analisado in documento.candidatos:
        try:
            candidato, chamadas_citacao = await _extrair_entidade(
                analisado, extrator, semaforo
            )
        except ErroExtracaoEntidades as erro:
            chamadas.extend(erro.auditorias)
            escrever_saida_documento(
                output_file,
                DocumentoEntidades(
                    documento_id=documento.documento_id,
                    candidatos=candidatos,
                ),
                chamadas,
            )
            raise
        candidatos.append(candidato)
        chamadas.extend(chamadas_citacao)
        escrever_saida_documento(
            output_file,
            DocumentoEntidades(
                documento_id=documento.documento_id,
                candidatos=candidatos,
            ),
            chamadas,
        )
        progresso.update()


async def executar_entities_async(
    input_file: Path,
    output_file: Path,
    extrator: ExtratorEntidadesAsync,
    max_concurrency: int,
) -> None:
    arquivos = listar_resultados(input_file)
    total_citacoes = sum(
        len(ler_documento(arquivo, DocumentoCompletude).candidatos)
        for arquivo in arquivos
    )
    semaforo = asyncio.Semaphore(max_concurrency)
    progresso = tqdm(
        total=total_citacoes,
        desc="Extraindo entidades",
        unit="citação",
    )
    try:
        await asyncio.gather(
            *(
                _processar_documento_entities(
                    arquivo, output_file, extrator, semaforo, progresso
                )
                for arquivo in arquivos
            )
        )
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
