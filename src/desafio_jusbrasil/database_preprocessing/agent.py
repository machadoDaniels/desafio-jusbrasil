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
from . import cabecalho
from .contracts import (
    CONTRATOS_COM_TRECHOS,
    Contract,
    DocumentoEnriquecido,
    DocumentoFonte,
    MetadadosAcordao,
    MetadadosDispositivo,
    MetadadosDocumento,
    MetadadosSumula,
    _MetadadosAcordaoAgente,
    _MetadadosAcordaoSemantico,
    _MetadadosDispositivoAgente,
)

_LOG = logging.getLogger(__name__)


def _entrada_agente(
    documento: DocumentoFonte,
    text_start_char_limit: int | None,
    text_end_char_limit: int | None,
    *,
    janela: str | None = None,
    chaves: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    entrada = {"tribunal": documento.tribunal}
    if chaves is not None:
        entrada["chaves_do_cabecalho"] = dict(chaves)
    if janela is not None:
        texto = janela
    elif text_start_char_limit is None and text_end_char_limit is None:
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
    *,
    janela: str | None = None,
    chaves: Mapping[str, Any] | None = None,
    prompt: str | None = None,
) -> dict[str, Any]:
    extra_body = {
        chave: config[chave]
        for chave in ("top_k", "chat_template_kwargs")
        if config.get(chave) is not None
    }
    entrada = json.dumps(
        _entrada_agente(
            documento,
            config["text_start_char_limit"],
            config["text_end_char_limit"],
            janela=janela,
            chaves=chaves,
        ),
        ensure_ascii=False,
    )
    prompt_sistema = prompt if prompt is not None else config["prompts"][documento.natureza]
    exemplos = config.get("_few_shot", {}).get(documento.natureza, [])
    if config.get("nuextract_templates"):
        # NuExtract: template, instruções e exemplos vão pelo chat template.
        extra_body["chat_template_kwargs"] = {
            **extra_body.get("chat_template_kwargs", {}),
            "template": json.dumps(
                config["nuextract_templates"][documento.natureza], ensure_ascii=False
            ),
            "instructions": config["prompts"][documento.natureza],
        }
        mensagens = [
            *(
                {
                    "role": "developer",
                    "content": [
                        {"type": "text", "text": json.dumps(exemplo[chave], ensure_ascii=False)}
                        for chave in ("entrada", "saida")
                    ],
                }
                for exemplo in exemplos
            ),
            {"role": "user", "content": entrada},
        ]
    else:
        mensagens = [
            {"role": "system", "content": prompt_sistema},
            *(
                mensagem
                for exemplo in exemplos
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
            {"role": "user", "content": entrada},
        ]
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
        "messages": mensagens,
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
    return MetadadosDispositivo.model_validate(campos.model_dump())


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
    if config.get("modo_chaves", "modelo") == "hibrido":
        return await _enriquecer_hibrido(
            documento, cliente=cliente, config=config, semaforo=semaforo
        )
    contrato = {
        "acordao": _MetadadosAcordaoAgente,
        "sumula": MetadadosSumula,
        "dispositivo": _MetadadosDispositivoAgente,
    }[documento.natureza]
    contrato_requisicao = (
        CONTRATOS_COM_TRECHOS[documento.natureza]
        if config.get("extrair_verbatim")
        else contrato
    )
    requisicao = _requisicao(documento, contrato_requisicao, config)
    ultimo_erro: Exception | None = None

    for tentativa in range(1, config["max_retries"] + 1):
        try:
            async with semaforo:
                resposta = await cliente.chat.completions.parse(**requisicao)
            parsed = resposta.choices[0].message.parsed
            if parsed is None:
                raise ValueError("o modelo não retornou uma resposta estruturada")
            campos = (
                contrato.model_validate(parsed.model_dump(exclude={"trechos"}))
                if config.get("extrair_verbatim")
                else parsed
            )
            resultado = DocumentoEnriquecido(
                documento_id=documento.documento_id,
                id=documento.id,
                natureza=documento.natureza,
                campos=normalizar_campos(documento, campos),
                trechos=(
                    parsed.trechos.model_dump() if config.get("extrair_verbatim") else None
                ),
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
                "input": _input_auditoria(requisicao, contrato_requisicao),
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


def _auditoria_sem_modelo(config: Mapping[str, Any], chaves: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "tentativa": 0,
        "modelo": config["model"],
        "fonte": "cabecalho",
        "parametros": _parametros_auditoria(config),
        "uso": None,
        "input": {"chaves_do_cabecalho": dict(chaves)},
        "output": None,
    }


def _parametros_auditoria(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        chave: valor
        for chave, valor in config.items()
        if not chave.startswith("_")
        and chave not in {"model", "max_concurrency", "max_retries", "retry_delay_seconds"}
        and valor is not None
    }


async def _enriquecer_hibrido(
    documento: DocumentoFonte,
    *,
    cliente: AsyncOpenAI,
    config: Mapping[str, Any],
    semaforo: asyncio.Semaphore,
) -> tuple[DocumentoEnriquecido, dict[str, Any]]:
    """Chaves de busca por cabeçalho; modelo somente para classe, cadeia e UF residual.

    Súmulas e dispositivos com cabeçalho regular nem chamam o modelo. Um acórdão sempre
    chama, porque classe e cadeia recursal exigem leitura semântica; as chaves extraídas
    por regra prevalecem sobre qualquer valor devolvido pelo modelo.
    """
    if documento.natureza == "sumula":
        campos = cabecalho.campos_sumula(documento)
        if campos is not None:
            resultado = DocumentoEnriquecido(
                documento_id=documento.documento_id,
                id=documento.id,
                natureza=documento.natureza,
                campos=normalizar_campos(documento, MetadadosSumula(**campos)),
            )
            return resultado, _auditoria_sem_modelo(config, campos)
    if documento.natureza == "dispositivo":
        campos = cabecalho.campos_dispositivo(documento)
        if campos is not None:
            resultado = DocumentoEnriquecido(
                documento_id=documento.documento_id,
                id=documento.id,
                natureza=documento.natureza,
                campos=MetadadosDispositivo(**campos),
            )
            return resultado, _auditoria_sem_modelo(config, campos)
    if documento.natureza != "acordao":
        # Cabeçalho irregular: cai no fluxo com modelo, sem as chaves de cabeçalho.
        sem_hibrido = {**config, "modo_chaves": "modelo"}
        return await enriquecer_documento(
            documento, cliente=cliente, config=sem_hibrido, semaforo=semaforo
        )

    extraido = cabecalho.chaves_acordao(
        documento,
        janela_chars=config.get("janela_cabecalho_chars") or cabecalho.JANELA_CABECALHO_PADRAO,
        janela_tst_chars=config.get("janela_tst_chars") or cabecalho.JANELA_TST_PADRAO,
    )
    chaves = extraido["campos"]
    prompt = config["prompts"].get("acordao_semantico") or config["prompts"]["acordao"]
    requisicao = _requisicao(
        documento,
        _MetadadosAcordaoSemantico,
        config,
        janela=extraido["janela"],
        chaves=chaves,
        prompt=prompt,
    )
    ultimo_erro: Exception | None = None
    for tentativa in range(1, config["max_retries"] + 1):
        try:
            async with semaforo:
                resposta = await cliente.chat.completions.parse(**requisicao)
            parsed = resposta.choices[0].message.parsed
            if parsed is None:
                raise ValueError("o modelo não retornou uma resposta estruturada")
            dados = {
                **parsed.model_dump(),
                **{chave: valor for chave, valor in chaves.items() if valor is not None},
            }
            if dados.get("uf") is None:
                dados["uf"] = parsed.uf
            campos = _MetadadosAcordaoAgente.model_validate(dados)
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
                "fonte": "cabecalho+modelo",
                "parametros": _parametros_auditoria(config),
                "uso": usage.model_dump(mode="json") if usage is not None else None,
                "input": _input_auditoria(requisicao, _MetadadosAcordaoSemantico),
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
