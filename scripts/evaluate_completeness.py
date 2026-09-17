# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "numpy>=2.0",
#   "pandas>=2.0",
#   "pydantic>=2.0",
#   "pyyaml>=6.0",
# ]
# ///

"""Avalia a decisão de completude para as citações extraídas."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "desafio-jusbrasil-bracis-2026"), str(ROOT / "src")]

from kaggle_metric import _casar

from desafio_jusbrasil.contracts import DocumentoCompletude


def _ler(pasta: Path) -> dict[str, DocumentoCompletude]:
    arquivos = sorted(
        arquivo for arquivo in pasta.glob("*.json") if arquivo.name != "manifest.json"
    )
    if not arquivos:
        raise ValueError(f"nenhum JSON encontrado em {pasta}")
    return {
        arquivo.stem: DocumentoCompletude.model_validate_json(
            arquivo.read_text(encoding="utf-8")
        )
        for arquivo in arquivos
    }


def _dividir(a: int, b: int) -> float:
    return a / b if b else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "predicoes",
        nargs="?",
        type=Path,
        default=Path("outputs/run-001/02-completeness"),
    )
    parser.add_argument(
        "--gold",
        type=Path,
        default=Path("outputs/gold/02-completeness"),
    )
    args = parser.parse_args()

    golds = _ler(args.gold)
    preditos = _ler(args.predicoes)
    casados = corretos = total_gold = total_predito = 0

    for documento_id, gold in golds.items():
        pred = preditos.get(documento_id)
        itens = pred.candidatos if pred else []
        localizados = [
            item
            for item in itens
            if item.candidato.inicio is not None and item.candidato.fim is not None
        ]
        pares, _, _ = _casar(
            [item.candidato.model_dump() for item in gold.candidatos],
            [item.candidato.model_dump() for item in localizados],
        )
        casados += len(pares)
        total_gold += len(gold.candidatos)
        total_predito += len(localizados)
        corretos += sum(
            gold.candidatos[gi].completude.completa
            == localizados[pi].completude.completa
            for gi, pi in pares
        )

    precisao = _dividir(corretos, total_predito)
    recall = _dividir(corretos, total_gold)
    f1 = 2 * precisao * recall / (precisao + recall) if precisao + recall else 0.0
    print(
        json.dumps(
            {
                "spans_casados": casados,
                "decisoes_corretas": corretos,
                "preditos": total_predito,
                "golds": total_gold,
                "acuracia_nos_pares": _dividir(corretos, casados),
                "precisao_end_to_end": precisao,
                "recall_end_to_end": recall,
                "f1_end_to_end": f1,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
