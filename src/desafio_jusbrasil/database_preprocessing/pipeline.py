"""Orquestra o enriquecimento auditável dos documentos canônicos."""

from __future__ import annotations

import asyncio
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

from ..utils import normalizar_relator
from .agent import enriquecer_documento
from .contracts import DocumentoEnriquecido, DocumentoFonte, MetadadosAcordao
from .database import listar_documentos, materializar_banco


def _escrever_json(caminho: Path, dados: BaseModel | dict[str, Any]) -> None:
    caminho.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(dados, BaseModel):
        dados = dados.model_dump(mode="json")
    temporario = caminho.with_suffix(caminho.suffix + ".tmp")
    temporario.write_text(
        json.dumps(dados, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporario.replace(caminho)


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
    parametros = auditoria.get("parametros", {})
    if (
        auditoria.get("modelo") != configuracao["model"]
        or parametros.get("base_url") != configuracao.get("base_url")
        or parametros.get("text_start_char_limit")
        != configuracao.get("text_start_char_limit")
        or parametros.get("text_end_char_limit")
        != configuracao.get("text_end_char_limit")
        or parametros.get("few_shot_path") != configuracao.get("few_shot_path")
    ):
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


async def _processar_documento(
    documento: DocumentoFonte,
    cliente: AsyncOpenAI,
    config: Mapping[str, Any],
    semaforo: asyncio.Semaphore,
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
        )
    except RuntimeError as erro:
        return documento, erro
    return documento, resultado


async def executar_async(
    *,
    origem: Path,
    destino: Path,
    diretorio_auditoria: Path,
    configuracao: Mapping[str, Any],
) -> dict[str, Any]:
    """Enriquece checkpoints e materializa a cópia SQLite somente após validação."""
    referencia_api_key = configuracao.get("api_key")
    api_key_env = referencia_api_key.removeprefix("$") if referencia_api_key else None
    config = {
        "_api_key": os.getenv(api_key_env) if api_key_env else None,
        "api_key_env": api_key_env,
        "text_start_char_limit": configuracao.get("text_start_char_limit", 10_000),
        "text_end_char_limit": configuracao.get("text_end_char_limit", 10_000),
        "chat_template_kwargs": configuracao.get("chat_template_kwargs"),
        "few_shot_path": configuracao.get("few_shot_path"),
        "_few_shot": (
            json.loads(Path(configuracao["few_shot_path"]).read_text(encoding="utf-8"))
            if configuracao.get("few_shot_path")
            else {}
        ),
        **{
            chave: configuracao[chave]
            for chave in (
                "model",
                "base_url",
                "temperature",
                "top_p",
                "top_k",
                "reasoning_effort",
                "max_concurrency",
                "max_retries",
                "retry_delay_seconds",
            )
        },
    }
    diretorio_auditoria.mkdir(parents=True, exist_ok=True)
    _escrever_json(
        diretorio_auditoria / "manifest.json",
        {
            "criado_em": datetime.now(UTC).isoformat(),
            "origem": str(origem),
            "configuracao_modelo": {
                chave: valor
                for chave, valor in config.items()
                if not chave.startswith("_")
            },
        },
    )

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
        resultado = _checkpoint_reutilizavel(checkpoint, documento, config)
        if resultado is None:
            pendentes.append(documento)
        else:
            resultados.append(resultado)
            reutilizados += 1

    revisoes: list[dict[str, Any]] = []
    semaforo = asyncio.Semaphore(config["max_concurrency"])

    async with AsyncOpenAI(
        base_url=config["base_url"], api_key=config["_api_key"], max_retries=0
    ) as cliente:
        tarefas = [
            asyncio.create_task(
                _processar_documento(documento, cliente, config, semaforo)
            )
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

    relatores = _carregar_relatores(origem, destino)
    documentos_por_id = {documento.documento_id: documento for documento in documentos}
    for indice, resultado in enumerate(resultados):
        documento = documentos_por_id[resultado.documento_id]
        if isinstance(resultado.campos, MetadadosAcordao):
            resultado = resultado.model_copy(
                update={
                    "campos": resultado.campos.model_copy(
                        update={
                            "relator_norm": normalizar_relator(
                                documento.relator, relatores
                            )
                        }
                    )
                }
            )
            resultados[indice] = resultado
        _escrever_json(
            diretorio_auditoria
            / "documentos"
            / documento.documento_id
            / "resultado.json",
            resultado,
        )

    with (diretorio_auditoria / "revisao.jsonl").open("w", encoding="utf-8") as arquivo:
        for revisao in revisoes:
            arquivo.write(json.dumps(revisao, ensure_ascii=False) + "\n")

    resultados.sort(key=lambda resultado: resultado.id)
    relatorio: dict[str, Any] = {
        "gerado_em": datetime.now(UTC).isoformat(),
        "origem": str(origem),
        "destino": str(destino),
        "modelo": config["model"],
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

    diagnostico = materializar_banco(origem, destino, resultados)
    relatorio["materializacao"] = diagnostico
    _escrever_json(diretorio_auditoria / "relatorio.json", relatorio)
    return relatorio


def executar(
    *,
    origem: Path,
    destino: Path,
    diretorio_auditoria: Path,
    configuracao: Mapping[str, Any],
) -> dict[str, Any]:
    """Executa o pipeline assíncrono a partir de uma CLI síncrona."""
    return asyncio.run(
        executar_async(
            origem=origem,
            destino=destino,
            diretorio_auditoria=diretorio_auditoria,
            configuracao=configuracao,
        )
    )
