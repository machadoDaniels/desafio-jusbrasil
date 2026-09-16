# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "numpy>=2.0",
#   "pandas>=2.0",
# ]
# ///

"""Avalia um submission.csv com a métrica oficial."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from kaggle_metric import avaliar

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "desafio-jusbrasil-bracis-2026"


def _agregar_goldenset() -> pd.DataFrame:
    gold = pd.read_csv(
        DATASET / "goldenset_offsets.csv",
        dtype=str,
        keep_default_na=False,
    )

    def encode(grupo: pd.DataFrame) -> str:
        citacoes = []
        for linha in grupo.sort_values(["inicio", "fim"]).itertuples():
            id_canonico = linha.id_canonico if linha.classificacao == "real" else "-"
            citacoes.append(
                f"{linha.inicio},{linha.fim},{linha.classificacao},{id_canonico}"
            )
        return "|".join(citacoes)

    return (
        gold.groupby(["documento_id", "nivel"], sort=False)
        .apply(encode, include_groups=False)
        .rename("citacoes")
        .reset_index()
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Executa localmente a métrica oficial da competição."
    )
    parser.add_argument(
        "submission",
        nargs="?",
        type=Path,
        default=Path("outputs/run-001/submission.csv"),
        help="arquivo submission.csv",
    )
    args = parser.parse_args()

    submission = pd.read_csv(args.submission)
    resultado = avaliar(_agregar_goldenset(), submission)
    print(json.dumps(resultado, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
