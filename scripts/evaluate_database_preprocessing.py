# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "pydantic>=2.0",
# ]
# ///

"""Compara uma run do pré-processamento com a run gold de referência."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from desafio_jusbrasil.database_preprocessing import DocumentoEnriquecido


def _ler_resultados(pasta: Path) -> dict[str, DocumentoEnriquecido]:
    documentos = pasta / "documentos"
    arquivos = sorted(documentos.glob("*/resultado.json"))
    if not arquivos:
        raise ValueError(f"nenhum resultado encontrado em {documentos}")
    resultados = {}
    for arquivo in arquivos:
        resultado = DocumentoEnriquecido.model_validate_json(
            arquivo.read_text(encoding="utf-8")
        )
        if resultado.documento_id in resultados:
            raise ValueError(f"documento duplicado: {resultado.documento_id}")
        resultados[resultado.documento_id] = resultado
    return resultados


def _valor_comparavel(campo: str, valor: Any) -> Any:
    if campo == "cadeia_recursal" and valor is not None:
        return frozenset(valor)
    return valor


def _metricas_campos(
    contagens: dict[str, Counter[str]],
    documentos_com_erro: dict[str, set[str]],
) -> dict[str, dict[str, Any]]:
    metricas = {}
    for campo, valores in sorted(contagens.items()):
        total = valores["total"]
        esperados = valores["valores_esperados"]
        preditos = valores["valores_preditos"]
        corretos = valores["valores_corretos"]
        metricas[campo] = {
            "corretos": valores["corretos"],
            "total": total,
            "acuracia": valores["corretos"] / total if total else 0.0,
            "valores_corretos": corretos,
            "valores_esperados": esperados,
            "valores_preditos": preditos,
            "precisao_valores": corretos / preditos if preditos else 0.0,
            "recall_valores": corretos / esperados if esperados else 0.0,
            "documentos_com_erro": sorted(documentos_com_erro.get(campo, set())),
        }
    return metricas


def avaliar(predicoes: Path, gold: Path) -> dict[str, Any]:
    esperados = _ler_resultados(gold)
    obtidos = _ler_resultados(predicoes)
    ids_esperados = set(esperados)
    ids_obtidos = set(obtidos)
    ids_comuns = sorted(ids_esperados & ids_obtidos)
    documentos_exatos = 0
    campos: dict[str, Counter[str]] = {}
    documentos_com_erro: dict[str, set[str]] = {}

    for documento_id in ids_comuns:
        esperado = esperados[documento_id]
        obtido = obtidos[documento_id]
        identidade_correta = (esperado.id, esperado.natureza) == (
            obtido.id,
            obtido.natureza,
        )
        esperado_campos = esperado.campos.model_dump(mode="json")
        obtido_campos = obtido.campos.model_dump(mode="json")
        campos_documento_corretos = identidade_correta
        for campo in sorted(set(esperado_campos) | set(obtido_campos)):
            chave = f"{esperado.natureza}.{campo}"
            contador = campos.setdefault(chave, Counter())
            valor_esperado = esperado_campos.get(campo)
            valor_obtido = obtido_campos.get(campo)
            igual = _valor_comparavel(campo, valor_esperado) == _valor_comparavel(
                campo, valor_obtido
            )
            contador["total"] += 1
            contador["corretos"] += igual
            contador["valores_esperados"] += valor_esperado is not None
            contador["valores_preditos"] += valor_obtido is not None
            contador["valores_corretos"] += (
                valor_esperado is not None and valor_obtido is not None and igual
            )
            if not igual:
                campos_documento_corretos = False
                documentos_com_erro.setdefault(chave, set()).add(documento_id)
        documentos_exatos += campos_documento_corretos

    total_gold = len(esperados)
    total_predito = len(obtidos)
    precisao = documentos_exatos / total_predito if total_predito else 0.0
    recall = documentos_exatos / total_gold if total_gold else 0.0
    return {
        "documentos_gold": total_gold,
        "documentos_preditos": total_predito,
        "documentos_comuns": len(ids_comuns),
        "documentos_exatos": documentos_exatos,
        "precisao_documentos_exatos": precisao,
        "recall_documentos_exatos": recall,
        "f1_documentos_exatos": (
            2 * precisao * recall / (precisao + recall) if precisao + recall else 0.0
        ),
        "documentos_ausentes": sorted(ids_esperados - ids_obtidos),
        "documentos_extras": sorted(ids_obtidos - ids_esperados),
        "metricas_por_campo": _metricas_campos(campos, documentos_com_erro),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "predicoes",
        nargs="?",
        type=Path,
        default=Path("outputs/database-preprocessing/run-005"),
    )
    parser.add_argument(
        "--gold",
        type=Path,
        default=Path("outputs/database-preprocessing/run-gemini-gold"),
    )
    args = parser.parse_args()
    relatorio = avaliar(args.predicoes, args.gold)
    args.predicoes.mkdir(parents=True, exist_ok=True)
    (args.predicoes / "avaliacao.json").write_text(
        json.dumps(relatorio, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    principais = {
        chave: relatorio[chave]
        for chave in (
            "documentos_gold",
            "documentos_preditos",
            "documentos_comuns",
            "documentos_exatos",
            "precisao_documentos_exatos",
            "recall_documentos_exatos",
            "f1_documentos_exatos",
        )
    }
    principais["acuracia_por_campo"] = {
        campo: metricas["acuracia"]
        for campo, metricas in relatorio["metricas_por_campo"].items()
    }
    print(json.dumps(principais, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
