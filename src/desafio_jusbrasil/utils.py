"""Utilitários compartilhados pelas etapas do pipeline."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from openai import omit

from .contracts import Contract, StageConfig


def ler_documento[Documento: Contract](
    arquivo: Path, contrato: type[Documento]
) -> Documento:
    """Lê e valida um checkpoint, preservando o caminho no erro."""
    try:
        return contrato.model_validate_json(arquivo.read_text(encoding="utf-8"))
    except ValueError as erro:
        raise ValueError(f"{arquivo}: {erro}") from erro


def listar_resultados(caminho: Path) -> list[Path]:
    """Lista checkpoints no layout atual e, por compatibilidade, no layout antigo."""
    arquivos = sorted(caminho.glob("*/resultado.json"))
    if not arquivos:
        arquivos = sorted(
            arquivo
            for arquivo in caminho.glob("*.json")
            if arquivo.name != "manifest.json"
        )
    if not arquivos:
        raise ValueError(f"nenhum arquivo JSON encontrado em {caminho}")
    return arquivos


def escrever_saida_documento(
    destino: Path,
    documento: Contract,
    chamadas: list[dict[str, Any]],
) -> None:
    """Grava o checkpoint agrupado e as chamadas ao modelo."""
    documento_id = documento.documento_id  # type: ignore[attr-defined]
    pasta = destino / documento_id
    pasta.mkdir(parents=True, exist_ok=True)
    for arquivo in pasta.glob("[0-9][0-9][0-9][0-9].json"):
        arquivo.unlink()
    arquivos = [(pasta / "resultado.json", documento.model_dump_json(indent=2))]
    arquivos.extend(
        (
            pasta / f"{indice:04d}.json",
            json.dumps(chamada, ensure_ascii=False, indent=2),
        )
        for indice, chamada in enumerate(chamadas, start=1)
    )
    for caminho, conteudo in arquivos:
        temporario = caminho.with_suffix(".json.tmp")
        temporario.write_text(conteudo + "\n", encoding="utf-8")
        temporario.replace(caminho)


def escrever_manifesto_etapa(
    destino: Path,
    etapa: str,
    config: StageConfig | None = None,
) -> None:
    """Registra a configuração efetiva da etapa junto aos checkpoints."""
    destino.mkdir(parents=True, exist_ok=True)
    manifesto = {
        "etapa": etapa,
        "gerado_em": datetime.now(UTC).isoformat(),
        "formato_checkpoint": "um JSON por documento",
        "configuracao_modelo": config.model_dump(mode="json") if config else None,
    }
    (destino / "manifest.json").write_text(
        json.dumps(manifesto, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def complementar_auditoria(
    auditoria: dict[str, Any],
    **detalhes: Any,
) -> dict[str, Any]:
    """Adiciona dados derivados mantendo input e output ao final."""
    return {
        **{
            chave: valor
            for chave, valor in auditoria.items()
            if chave not in {"input", "output"}
        },
        **detalhes,
        "input": auditoria["input"],
        "output": auditoria["output"],
    }


def criar_auditoria(
    requisicao: dict[str, Any],
    resposta: Any,
    response_format: type[Contract],
    **detalhes: Any,
) -> dict[str, Any]:
    """Monta a auditoria serializável de uma chamada estruturada."""
    entrada = {nome: valor for nome, valor in requisicao.items() if valor is not omit}
    entrada["response_format"] = response_format.model_json_schema()
    return {
        **detalhes,
        "input": entrada,
        "output": resposta.model_dump(
            mode="json",
            exclude={"choices": {"__all__": {"message": {"parsed"}}}},
        ),
    }
