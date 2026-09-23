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
from desafio_jusbrasil.utils import listar_resultados


def _pasta_etapa(pasta: Path) -> Path:
    etapa = pasta / "02-completeness"
    return etapa if etapa.is_dir() else pasta


def _ler(pasta: Path) -> dict[str, DocumentoCompletude]:
    pasta = _pasta_etapa(pasta)
    return {
        documento.documento_id: documento
        for arquivo in listar_resultados(pasta)
        for documento in [
            DocumentoCompletude.model_validate_json(arquivo.read_text(encoding="utf-8"))
        ]
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
    pasta_predicoes = _pasta_etapa(args.predicoes)
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
    relatorio = {
        "spans_casados": casados,
        "decisoes_corretas": corretos,
        "preditos": total_predito,
        "golds": total_gold,
        "acuracia_nos_pares": _dividir(corretos, casados),
        "precisao_end_to_end": precisao,
        "recall_end_to_end": recall,
        "f1_end_to_end": f1,
    }
    pasta_predicoes.mkdir(parents=True, exist_ok=True)
    (pasta_predicoes / "avaliacao.json").write_text(
        json.dumps(relatorio, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
