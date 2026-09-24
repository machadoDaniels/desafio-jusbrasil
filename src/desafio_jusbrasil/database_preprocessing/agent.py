"""Chamada auditável ao modelo para enriquecimento dos documentos canônicos."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from collections.abc import Mapping
from typing import Any

from openai import AsyncOpenAI, OpenAIError, omit
from pydantic import ValidationError

from ..utils import normalizar_numero_cnj
from .contracts import (
    Contract,
    DocumentoEnriquecido,
    DocumentoFonte,
    MetadadosAcordao,
    MetadadosDispositivo,
    MetadadosDocumento,
    MetadadosSumula,
    _MetadadosAcordaoAgente,
)

_LOG = logging.getLogger(__name__)


def _entrada_agente(
    documento: DocumentoFonte,
    text_start_char_limit: int | None,
    text_end_char_limit: int | None,
) -> dict[str, Any]:
    entrada = {"tribunal": documento.tribunal}
    if text_start_char_limit is None and text_end_char_limit is None:
        texto = documento.texto
    elif text_start_char_limit is None:
        texto = documento.texto[-text_end_char_limit:] if text_end_char_limit else ""
    elif text_end_char_limit is None:
        texto = documento.texto[:text_start_char_limit]
    elif len(documento.texto) <= text_start_char_limit + text_end_char_limit:
        texto = documento.texto
    else:
        inicio = documento.texto[:text_start_char_limit]
        fim = documento.texto[-text_end_char_limit:] if text_end_char_limit else ""
        texto = inicio + fim
    if documento.natureza == "acordao":
        entrada.update(ano=documento.ano, texto=texto)
    else:
        entrada["texto"] = texto
    return entrada


def _requisicao(
    documento: DocumentoFonte,
    contrato: type[Contract],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    extra_body = {
        chave: config[chave]
        for chave in ("top_k", "chat_template_kwargs")
        if config.get(chave) is not None
    }
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
            {"role": "system", "content": config["prompts"][documento.natureza]},
            *(
                mensagem
                for exemplo in config.get("_few_shot", {}).get(documento.natureza, [])
                for mensagem in (
                    {
                        "role": "user",
                        "content": json.dumps(exemplo["entrada"], ensure_ascii=False),
                    },
                    {
                        "role": "assistant",
                        "content": json.dumps(exemplo["saida"], ensure_ascii=False),
                    },
                )
            ),
            {
                "role": "user",
                "content": json.dumps(
                    _entrada_agente(
                        documento,
                        config["text_start_char_limit"],
                        config["text_end_char_limit"],
                    ),
                    ensure_ascii=False,
                ),
            },
        ],
        "response_format": contrato,
        **({"extra_body": extra_body} if extra_body else {}),
    }


def normalizar_campos(
    documento: DocumentoFonte,
    campos: MetadadosDocumento,
) -> MetadadosDocumento:
    """Aplica as mesmas regras determinísticas a respostas novas e checkpoints."""
    if documento.natureza == "acordao":
        dados = campos.model_dump(exclude={"relator_norm"})
        dados["numero_processo_cnj"] = normalizar_numero_cnj(
            campos.numero_processo_cnj, documento.texto
        )
        return MetadadosAcordao(**dados)
    if documento.natureza == "sumula":
        dados = campos.model_dump()
        dados["sumula_vinculante"] = bool(
            re.search(r"\bs[uú]mula\s+vinculante\b", documento.texto, re.IGNORECASE)
        )
        return MetadadosSumula.model_validate(dados)
    return campos


def _input_auditoria(
    requisicao: Mapping[str, Any], contrato: type[Contract]
) -> dict[str, Any]:
    entrada = {
        chave: valor
        for chave, valor in requisicao.items()
        if chave != "response_format" and valor is not omit
    }
    entrada["response_format"] = contrato.model_json_schema()
    return entrada


async def enriquecer_documento(
    documento: DocumentoFonte,
    *,
    cliente: AsyncOpenAI,
    config: Mapping[str, Any],
    semaforo: asyncio.Semaphore,
) -> tuple[DocumentoEnriquecido, dict[str, Any]]:
    """Extrai, normaliza e audita os metadados de um documento."""
    contrato = {
        "acordao": _MetadadosAcordaoAgente,
        "sumula": MetadadosSumula,
        "dispositivo": MetadadosDispositivo,
    }[documento.natureza]
    requisicao = _requisicao(documento, contrato, config)
    ultimo_erro: Exception | None = None

    for tentativa in range(1, config["max_retries"] + 1):
        try:
            async with semaforo:
                resposta = await cliente.chat.completions.parse(**requisicao)
            campos = resposta.choices[0].message.parsed
            if campos is None:
                raise ValueError("o modelo não retornou uma resposta estruturada")
            resultado = DocumentoEnriquecido(
                documento_id=documento.documento_id,
                id=documento.id,
                natureza=documento.natureza,
                campos=normalizar_campos(documento, campos),
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
                    }
                    and valor is not None
                },
                "uso": usage.model_dump(mode="json") if usage is not None else None,
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
