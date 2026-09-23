"""Chamada auditável ao modelo para enriquecimento dos documentos canônicos."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from collections.abc import Mapping
from typing import Any

from openai import AsyncOpenAI, OpenAIError, omit
from pydantic import ValidationError

from ..utils import normalizar_numero_processo, normalizar_relator
from .contracts import (
    DocumentoEnriquecido,
    MetadadosAcordao,
    MetadadosDocumento,
    MetadadosSumula,
    contrato_para_natureza,
)

_LOG = logging.getLogger(__name__)

_PROMPTS = {
    "acordao": """Extraia somente metadados jurídicos explicitamente sustentados pelo documento.
Não use conhecimento externo e não invente valores. Retorne número do processo somente com
algarismos; classifique-o como cnj, classico ou sem_numero. Extraia a classe principal, todas as
classes da cadeia recursal e a UF. A ordem da cadeia recursal não tem significado. Campos ausentes
devem ser nulos.""",
    "sumula": """Extraia somente os metadados explicitamente sustentados pela súmula. Não use
conhecimento externo e não invente valores. Informe o número da súmula e se ela é vinculante;
use nulo quando o documento não sustentar o campo.""",
    "dispositivo": """Extraia somente os metadados legislativos explicitamente sustentados pelo
dispositivo. Não use conhecimento externo e não invente valores. Normalize o diploma para o
vocabulário permitido e retorne números de diploma somente com algarismos. Preserve sufixos
alfanuméricos do artigo. Campos ausentes devem ser nulos.""",
}


def _json(valor: Any) -> Any:
    if hasattr(valor, "model_dump"):
        return valor.model_dump(mode="json")
    if isinstance(valor, Mapping):
        return {str(chave): _json(item) for chave, item in valor.items()}
    if isinstance(valor, (list, tuple)):
        return [_json(item) for item in valor]
    return valor


def _entrada_agente(documento: Any, header_char_limit: int) -> dict[str, Any]:
    entrada = {
        "documento_id": documento.documento_id,
        "id": documento.id,
        "tribunal": documento.tribunal,
    }
    if documento.natureza == "acordao":
        entrada.update(ano=documento.ano, texto=documento.texto[:header_char_limit])
    else:
        entrada["texto"] = documento.texto
    return entrada


def _requisicao(
    documento: Any, contrato: type[Any], config: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "model": config["model"],
        "temperature": (
            config["temperature"] if config["temperature"] is not None else omit
        ),
        "top_p": config["top_p"] if config["top_p"] is not None else omit,
        "reasoning_effort": (
            config["reasoning_effort"]
            if config["reasoning_effort"] is not None
            else omit
        ),
        "messages": [
            {"role": "system", "content": _PROMPTS[documento.natureza]},
            {
                "role": "user",
                "content": json.dumps(
                    _entrada_agente(documento, config["header_char_limit"]),
                    ensure_ascii=False,
                ),
            },
        ],
        "response_format": contrato,
        **(
            {"extra_body": {"top_k": config["top_k"]}}
            if config["top_k"] is not None
            else {}
        ),
    }


class LimitadorTaxa:
    """Serializa o início das requisições para respeitar o intervalo configurado."""

    def __init__(self, intervalo: float) -> None:
        self._intervalo = intervalo
        self._proxima = 0.0
        self._lock = asyncio.Lock()

    async def aguardar(self) -> None:
        async with self._lock:
            agora = time.monotonic()
            espera = max(0.0, self._proxima - agora)
            if espera:
                await asyncio.sleep(espera)
            self._proxima = time.monotonic() + self._intervalo


def normalizar_campos(
    documento: Any,
    campos: MetadadosDocumento,
    relatores: Mapping[str, str],
) -> MetadadosDocumento:
    """Aplica as mesmas regras determinísticas a respostas novas e checkpoints."""
    if documento.natureza == "acordao":
        dados = campos.model_dump(exclude={"relator_norm"})
        numero, formato = normalizar_numero_processo(
            campos.numero_processo, documento.texto
        )
        dados.update(numero_processo=numero, formato_numero=formato)
        return MetadadosAcordao(
            **dados,
            relator_norm=normalizar_relator(documento.relator, relatores),
        )
    if documento.natureza == "sumula":
        dados = campos.model_dump()
        dados["sumula_vinculante"] = bool(
            re.search(r"\bs[uú]mula\s+vinculante\b", documento.texto, re.IGNORECASE)
        )
        return MetadadosSumula.model_validate(dados)
    return campos


def _input_auditoria(
    requisicao: Mapping[str, Any], contrato: type[Any]
) -> dict[str, Any]:
    entrada = {
        chave: valor
        for chave, valor in requisicao.items()
        if chave != "response_format" and valor is not omit
    }
    entrada["response_format"] = contrato.model_json_schema()
    return entrada


async def enriquecer_documento(
    documento: Any,
    *,
    cliente: AsyncOpenAI,
    config: Mapping[str, Any],
    semaforo: asyncio.Semaphore,
    limitador: LimitadorTaxa,
    relatores: Mapping[str, str],
) -> tuple[DocumentoEnriquecido, dict[str, Any]]:
    """Extrai, normaliza e audita os metadados de um documento."""
    contrato = contrato_para_natureza(documento.natureza)
    requisicao = _requisicao(documento, contrato, config)
    ultimo_erro: Exception | None = None

    for tentativa in range(1, config["max_retries"] + 1):
        try:
            await limitador.aguardar()
            async with semaforo:
                resposta = await cliente.chat.completions.parse(**requisicao)
            campos = resposta.choices[0].message.parsed
            if campos is None:
                raise ValueError("o modelo não retornou uma resposta estruturada")
            resultado = DocumentoEnriquecido(
                documento_id=documento.documento_id,
                id=documento.id,
                natureza=documento.natureza,
                campos=normalizar_campos(documento, campos, relatores),
            )
            usage = getattr(resposta, "usage", None)
            return resultado, {
                "tentativa": tentativa,
                "modelo": config["model"],
                "parametros": {
                    chave: valor
                    for chave, valor in config.items()
                    if not chave.startswith("_")
                    and chave
                    not in {
                        "model",
                        "max_concurrency",
                        "max_retries",
                        "retry_delay_seconds",
                        "header_char_limit",
                    }
                    and valor is not None
                },
                "uso": _json(usage) if usage is not None else None,
                "input": _input_auditoria(requisicao, contrato),
                "output": resposta.model_dump(
                    mode="json",
                    exclude={"choices": {"__all__": {"message": {"parsed"}}}},
                ),
            }
        except (ValidationError, ValueError, IndexError, AttributeError) as erro:
            ultimo_erro = erro
            requisicao["messages"].append(
                {
                    "role": "user",
                    "content": (
                        "A resposta anterior foi inválida. Corrija sem inventar dados: "
                        f"{str(erro)[:2000]}"
                    ),
                }
            )
        except OpenAIError as erro:
            ultimo_erro = erro

        _LOG.info(
            "documento %s: tentativa %d/%d falhou: %s",
            documento.documento_id,
            tentativa,
            config["max_retries"],
            ultimo_erro,
        )
        if tentativa < config["max_retries"]:
            await asyncio.sleep(config["retry_delay_seconds"] * tentativa)

    assert ultimo_erro is not None
    raise RuntimeError(
        f"{documento.documento_id}: retries esgotados: {ultimo_erro}"
    ) from ultimo_erro
