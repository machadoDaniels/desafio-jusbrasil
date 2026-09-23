"""Utilitários compartilhados pelas etapas do pipeline."""

from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from openai import omit

from .contracts import Contract, StageConfig

_PADRAO_CNJ = re.compile(
    r"(?<!\d)(\d{1,7})\s*-\s*(\d{2})\s*\.\s*(\d{4})\s*\.\s*"
    r"(\d)\s*\.\s*(\d{2})\s*\.\s*(\d{4})(?!\d)"
)


def normalizar_numero_cnj(valor: str | None, trecho: str = "") -> str | None:
    """Normaliza CNJ e recupera zeros à esquerda do padrão publicado."""
    digitos = re.sub(r"\D", "", valor or "")
    candidatos = {
        correspondencia.group(1).zfill(7) + "".join(correspondencia.groups()[1:])
        for correspondencia in _PADRAO_CNJ.finditer(trecho)
    }
    compativeis = {
        candidato
        for candidato in candidatos
        if not digitos or candidato.lstrip("0") == digitos.lstrip("0")
    }
    if len(compativeis) == 1:
        digitos = compativeis.pop()
    return digitos if len(digitos) == 20 else None


def _chave_relator(valor: str) -> str:
    sem_acentos = "".join(
        caractere
        for caractere in unicodedata.normalize("NFKD", valor.casefold())
        if not unicodedata.combining(caractere)
    )
    return " ".join(re.sub(r"[^a-z0-9]+", " ", sem_acentos).split())


def normalizar_relator(relator: str | None, relatores: Mapping[str, str]) -> str | None:
    """Resolve uma variante de relator para um único nome canônico conhecido."""
    if not relator:
        return None
    chave = _chave_relator(relator)
    candidatos = {chave}
    for prefixo in ("relator ", "relatora ", "ministro ", "ministra ", "min "):
        if chave.startswith(prefixo):
            candidatos.add(chave.removeprefix(prefixo))
    correspondencias = {
        canonico
        for variante, canonico in relatores.items()
        if _chave_relator(variante) in candidatos
        or _chave_relator(canonico) in candidatos
    }
    return correspondencias.pop() if len(correspondencias) == 1 else None


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


def criar_auditoria_erro(
    requisicao: dict[str, Any],
    response_format: type[Contract],
    erro: Exception,
    tentativa: int,
    resposta: Any = None,
) -> dict[str, Any]:
    """Monta a auditoria de uma tentativa de modelo que terminou em erro."""
    entrada = {nome: valor for nome, valor in requisicao.items() if valor is not omit}
    entrada["response_format"] = response_format.model_json_schema()
    saida = (
        resposta.model_dump(
            mode="json",
            exclude={"choices": {"__all__": {"message": {"parsed"}}}},
        )
        if resposta is not None
        else None
    )
    return {
        "tentativa": tentativa,
        "erro": {"tipo": type(erro).__name__, "mensagem": str(erro)},
        "input": entrada,
        "output": saida,
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
