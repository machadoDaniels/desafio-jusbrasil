"""Orquestra o enriquecimento auditável dos documentos canônicos."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI
from pydantic import BaseModel
from tqdm.auto import tqdm

from .agent import LimitadorTaxa, enriquecer_documento, normalizar_campos
from .contracts import DocumentoEnriquecido, DocumentoFonte
from .database import listar_documentos, materializar_banco

_PROMPT_VERSION = "2"
_SCHEMA_VERSION = "4"
_CONFIG_CHECKPOINT = (
    "model",
    "base_url",
    "temperature",
    "top_p",
    "top_k",
    "reasoning_effort",
    "header_char_limit",
    "api_key_env",
)


def _escrever_json(caminho: Path, dados: BaseModel | dict[str, Any]) -> None:
    caminho.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(dados, BaseModel):
        dados = dados.model_dump(mode="json")
    temporario = caminho.with_suffix(caminho.suffix + ".tmp")
    temporario.write_text(
        json.dumps(dados, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporario.replace(caminho)


def _hash_arquivo(caminho: Path) -> str:
    digest = hashlib.sha256()
    with caminho.open("rb") as arquivo:
        for bloco in iter(lambda: arquivo.read(1024 * 1024), b""):
            digest.update(bloco)
    return digest.hexdigest()


def _validar_configuracao(configuracao: Mapping[str, Any]) -> dict[str, Any]:
    if not configuracao.get("model"):
        raise ValueError("database_preprocessing.model é obrigatório")
    referencia_api_key = configuracao.get("api_key")
    api_key = None
    api_key_env = None
    if referencia_api_key:
        if not isinstance(referencia_api_key, str) or not referencia_api_key.startswith(
            "$"
        ):
            raise ValueError(
                "database_preprocessing.api_key deve referenciar $VARIAVEL"
            )
        api_key_env = referencia_api_key[1:]
        api_key = os.getenv(api_key_env)
        if not api_key:
            raise ValueError(f"variável de ambiente ausente: {api_key_env}")
    valores = {
        "_api_key": api_key,
        "api_key_env": api_key_env,
        "model": configuracao["model"],
        "base_url": configuracao.get("base_url"),
        "temperature": configuracao.get("temperature"),
        "top_p": configuracao.get("top_p"),
        "top_k": configuracao.get("top_k"),
        "reasoning_effort": configuracao.get("reasoning_effort"),
        "max_concurrency": configuracao.get("max_concurrency", 4),
        "max_retries": configuracao.get("max_retries", 3),
        "retry_delay_seconds": configuracao.get("retry_delay_seconds", 1),
        "request_interval_seconds": configuracao.get("request_interval_seconds", 0.75),
        "header_char_limit": configuracao.get("header_char_limit", 6000),
    }
    if valores["max_concurrency"] < 1 or valores["max_retries"] < 1:
        raise ValueError("max_concurrency e max_retries devem ser maiores que zero")
    if (
        valores["header_char_limit"] < 1
        or valores["retry_delay_seconds"] < 0
        or valores["request_interval_seconds"] < 0
    ):
        raise ValueError(
            "header_char_limit deve ser positivo e intervalos não podem ser negativos"
        )
    return valores


def _assinatura_checkpoint(configuracao: Mapping[str, Any]) -> tuple[Any, ...]:
    return tuple(configuracao.get(chave) for chave in _CONFIG_CHECKPOINT)


def _checkpoint_reutilizavel(
    caminho: Path,
    documento: DocumentoFonte,
    configuracao: Mapping[str, Any],
) -> DocumentoEnriquecido | None:
    caminho_auditoria = caminho.with_name("0001.json")
    if not caminho.is_file() or not caminho_auditoria.is_file():
        return None
    try:
        resultado = DocumentoEnriquecido.model_validate_json(
            caminho.read_text(encoding="utf-8")
        )
        auditoria = json.loads(caminho_auditoria.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if auditoria.get("modelo") != configuracao["model"] or auditoria.get(
        "parametros", {}
    ).get("base_url") != configuracao.get("base_url"):
        return None
    if (
        resultado.documento_id != documento.documento_id
        or resultado.id != documento.id
        or resultado.natureza != documento.natureza
    ):
        return None
    return resultado


def _carregar_relatores(origem: Path, destino: Path) -> dict[str, str]:
    candidatos = (
        origem.parent / "relatores_padronizacao.json",
        destino.parent / "relatores_padronizacao.json",
    )
    caminho = next((item for item in candidatos if item.is_file()), candidatos[0])
    with caminho.open(encoding="utf-8") as arquivo:
        relatores = json.load(arquivo)
    if not isinstance(relatores, dict) or not all(
        isinstance(chave, str) and isinstance(valor, str)
        for chave, valor in relatores.items()
    ):
        raise ValueError(f"dicionário de relatores inválido: {caminho}")
    return relatores


def _cobertura(
    resultados: Iterable[DocumentoEnriquecido],
) -> dict[str, dict[str, int]]:
    totais: dict[str, Counter[str]] = defaultdict(Counter)
    for resultado in resultados:
        for campo, valor in resultado.campos.model_dump(mode="json").items():
            totais[resultado.natureza][campo] += valor is not None
            totais[resultado.natureza][f"{campo}_nulo"] += valor is None
    return {natureza: dict(contagens) for natureza, contagens in totais.items()}


async def executar_async(
    *,
    origem: Path,
    destino: Path,
    diretorio_auditoria: Path,
    configuracao: Mapping[str, Any],
    force: bool = False,
) -> dict[str, Any]:
    """Enriquece checkpoints e materializa a cópia SQLite somente após validação."""
    config = _validar_configuracao(configuracao)
    diretorio_auditoria.mkdir(parents=True, exist_ok=True)
    caminho_manifesto = diretorio_auditoria / "manifest.json"
    manifesto_anterior = (
        json.loads(caminho_manifesto.read_text(encoding="utf-8"))
        if caminho_manifesto.is_file()
        else {}
    )
    manifesto_atual = {
        "schema_version": _SCHEMA_VERSION,
        "prompt_version": _PROMPT_VERSION,
        "criado_em": datetime.now(UTC).isoformat(),
        "origem": str(origem),
        "hash_origem": _hash_arquivo(origem),
        "configuracao_modelo": {
            chave: valor for chave, valor in config.items() if not chave.startswith("_")
        },
    }
    mesmo_checkpoint = all(
        manifesto_anterior.get(chave) == manifesto_atual[chave]
        for chave in ("prompt_version", "hash_origem")
    ) and _assinatura_checkpoint(
        manifesto_anterior.get("configuracao_modelo", {})
    ) == _assinatura_checkpoint(manifesto_atual["configuracao_modelo"])
    _escrever_json(caminho_manifesto, manifesto_atual)

    relatores = _carregar_relatores(origem, destino)
    documentos = listar_documentos(origem)
    resultados: list[DocumentoEnriquecido] = []
    pendentes: list[DocumentoFonte] = []
    reutilizados = 0
    for documento in documentos:
        checkpoint = (
            diretorio_auditoria
            / "documentos"
            / documento.documento_id
            / "resultado.json"
        )
        resultado = (
            _checkpoint_reutilizavel(checkpoint, documento, config)
            if mesmo_checkpoint
            else None
        )
        if resultado is None:
            pendentes.append(documento)
        else:
            resultado = resultado.model_copy(
                update={
                    "campos": normalizar_campos(documento, resultado.campos, relatores)
                }
            )
            _escrever_json(checkpoint, resultado)
            resultados.append(resultado)
            reutilizados += 1

    revisoes: list[dict[str, Any]] = []
    semaforo = asyncio.Semaphore(config["max_concurrency"])
    limitador = LimitadorTaxa(config["request_interval_seconds"])

    async def processar(
        documento: DocumentoFonte, cliente: AsyncOpenAI
    ) -> tuple[
        DocumentoFonte,
        tuple[DocumentoEnriquecido, dict[str, Any]] | RuntimeError,
    ]:
        try:
            resultado = await enriquecer_documento(
                documento,
                cliente=cliente,
                config=config,
                semaforo=semaforo,
                limitador=limitador,
                relatores=relatores,
            )
        except RuntimeError as erro:
            return documento, erro
        return documento, resultado

    async with AsyncOpenAI(
        base_url=config["base_url"], api_key=config["_api_key"], max_retries=0
    ) as cliente:
        tarefas = [
            asyncio.create_task(processar(documento, cliente))
            for documento in pendentes
        ]
        with tqdm(
            total=len(documentos),
            initial=reutilizados,
            desc="Pré-processando banco",
            unit="documento",
        ) as progresso:
            for tarefa in asyncio.as_completed(tarefas):
                documento, resposta = await tarefa
                if isinstance(resposta, RuntimeError):
                    revisoes.append(
                        {
                            "documento_id": documento.documento_id,
                            "id": documento.id,
                            "natureza": documento.natureza,
                            "motivo": str(resposta),
                        }
                    )
                else:
                    resultado, auditoria = resposta
                    pasta = diretorio_auditoria / "documentos" / documento.documento_id
                    _escrever_json(pasta / "resultado.json", resultado)
                    _escrever_json(pasta / "0001.json", auditoria)
                    resultados.append(resultado)
                progresso.update()

    with (diretorio_auditoria / "revisao.jsonl").open("w", encoding="utf-8") as arquivo:
        for revisao in revisoes:
            arquivo.write(json.dumps(revisao, ensure_ascii=False) + "\n")

    resultados.sort(key=lambda resultado: resultado.id)
    relatorio: dict[str, Any] = {
        "gerado_em": datetime.now(UTC).isoformat(),
        "origem": {"caminho": str(origem), "sha256": _hash_arquivo(origem)},
        "destino": {"caminho": str(destino)},
        "modelo": config["model"],
        "prompt_version": _PROMPT_VERSION,
        "schema_version": _SCHEMA_VERSION,
        "total_documentos": len(documentos),
        "processados": len(pendentes) - len(revisoes),
        "reutilizados": reutilizados,
        "invalidos": len(revisoes),
        "enviados_para_revisao": len(revisoes),
        "cobertura_por_natureza": _cobertura(resultados),
        "materializacao": None,
    }
    if revisoes:
        _escrever_json(diretorio_auditoria / "relatorio.json", relatorio)
        raise RuntimeError(
            f"{len(revisoes)} documentos exigem revisão; banco não materializado"
        )

    diagnostico = materializar_banco(origem, destino, resultados, force=force)
    relatorio["materializacao"] = diagnostico
    if destino.is_file():
        relatorio["destino"]["sha256"] = _hash_arquivo(destino)
    _escrever_json(diretorio_auditoria / "relatorio.json", relatorio)
    return relatorio


def executar(
    *,
    origem: Path,
    destino: Path,
    diretorio_auditoria: Path,
    configuracao: Mapping[str, Any],
    force: bool = False,
) -> dict[str, Any]:
    """Executa o pipeline assíncrono a partir de uma CLI síncrona."""
    return asyncio.run(
        executar_async(
            origem=origem,
            destino=destino,
            diretorio_auditoria=diretorio_auditoria,
            configuracao=configuracao,
            force=force,
        )
    )
