# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "numpy>=2.0",
#   "pandas>=2.0",
#   "pydantic>=2.0",
#   "pyyaml>=6.0",
# ]
# ///

"""Avalia classes e IDs canônicos produzidos pela etapa de veracidade."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "desafio-jusbrasil-bracis-2026"), str(ROOT / "src")]

from kaggle_metric import _casar

from desafio_jusbrasil.contracts import Classificacao, DocumentoClassificado


def _ler(pasta: Path) -> dict[str, DocumentoClassificado]:
    arquivos = sorted(pasta.glob("*.json"))
    if not arquivos:
        raise ValueError(f"nenhum JSON encontrado em {pasta}")
    return {
        arquivo.stem: DocumentoClassificado.model_validate_json(
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
        default=Path("outputs/run-001/03-veracity"),
    )
    parser.add_argument(
        "--gold",
        type=Path,
        default=Path("outputs/gold/03-veracity"),
    )
    args = parser.parse_args()

    golds = _ler(args.gold)
    preditos = _ler(args.predicoes)
    casados = classes_corretas = corretos = ids_corretos = reais_preditos_real = 0
    total_gold = total_predito = 0

    for documento_id, gold in golds.items():
        pred = preditos.get(documento_id)
        itens = pred.candidatos if pred else []
        pares, _, _ = _casar(
            [item.candidato.model_dump() for item in gold.candidatos],
            [item.candidato.model_dump() for item in itens],
        )
        casados += len(pares)
        total_gold += len(gold.candidatos)
        total_predito += len(itens)
        for gi, pi in pares:
            esperado = gold.candidatos[gi].veracidade
            obtido = itens[pi].veracidade
            classe_correta = esperado.classificacao == obtido.classificacao
            classes_corretas += classe_correta
            if classe_correta and esperado.classificacao == Classificacao.REAL:
                reais_preditos_real += 1
                ids_corretos += esperado.id_canonico == obtido.id_canonico
            corretos += classe_correta and (
                esperado.classificacao != Classificacao.REAL
                or esperado.id_canonico == obtido.id_canonico
            )

    precisao = _dividir(corretos, total_predito)
    recall = _dividir(corretos, total_gold)
    f1 = 2 * precisao * recall / (precisao + recall) if precisao + recall else 0.0
    print(
        json.dumps(
            {
                "spans_casados": casados,
                "predicoes_totalmente_corretas": corretos,
                "preditos": total_predito,
                "golds": total_gold,
                "acuracia_classe_nos_pares": _dividir(classes_corretas, casados),
                "acuracia_id_reais": _dividir(ids_corretos, reais_preditos_real),
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
