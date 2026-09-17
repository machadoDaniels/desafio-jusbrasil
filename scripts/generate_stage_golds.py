# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "pandas>=2.0",
#   "pydantic>=2.0",
#   "pyyaml>=6.0",
# ]
# ///

"""Gera os checkpoints gold de cada etapa a partir do goldenset oficial."""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from desafio_jusbrasil.contracts import (
    CandidatoAnalisado,
    CandidatoCitacao,
    CandidatoClassificado,
    Classificacao,
    DocumentoClassificado,
    DocumentoCompletude,
    DocumentoExtraido,
    ResultadoCompletude,
    ResultadoVeracidade,
    TipoCitacao,
)

ETAPAS = ("01-extraction", "02-completeness", "03-veracity")


def _salvar(
    destino: Path,
    documento: DocumentoExtraido | DocumentoCompletude | DocumentoClassificado,
) -> None:
    destino.mkdir(parents=True, exist_ok=True)
    (destino / f"{documento.documento_id}.json").write_text(
        documento.model_dump_json(indent=2, exclude_none=True) + "\n",
        encoding="utf-8",
    )


def gerar(dataset: Path, destino: Path) -> None:
    gold = pd.read_csv(
        dataset / "goldenset_offsets.csv",
        dtype=str,
        keep_default_na=False,
    )

    for etapa in ETAPAS:
        shutil.rmtree(destino / etapa, ignore_errors=True)

    submissao = []
    for documento_id, linhas in gold.groupby("documento_id", sort=True):
        texto = (dataset / "txt" / f"{documento_id}.txt").read_text(encoding="utf-8")
        linhas = linhas.sort_values(["inicio", "fim"])
        candidatos = []
        for linha in linhas.itertuples():
            inicio, fim = int(linha.inicio), int(linha.fim)
            candidatos.append(
                CandidatoCitacao(
                    trecho=texto[inicio:fim],
                    tipo=TipoCitacao(linha.tipo),
                    inicio=inicio,
                    fim=fim,
                )
            )

        extracao = DocumentoExtraido(
            documento_id=documento_id,
            texto=texto,
            candidatos=candidatos,
        )
        analisados = []
        classificados = []
        for linha, candidato in zip(linhas.itertuples(), candidatos, strict=True):
            classificacao = Classificacao(linha.classificacao)
            completa = classificacao != Classificacao.INCOMPLETA
            completude = ResultadoCompletude(completa=completa)
            analisados.append(
                CandidatoAnalisado(candidato=candidato, completude=completude)
            )
            classificados.append(
                CandidatoClassificado(
                    candidato=candidato,
                    completude=completude,
                    veracidade=ResultadoVeracidade(
                        classificacao=classificacao,
                        id_canonico=(
                            int(linha.id_canonico)
                            if classificacao == Classificacao.REAL
                            else None
                        ),
                        justificativa="Derivado do goldenset.",
                    ),
                )
            )

        _salvar(destino / "01-extraction", extracao)
        _salvar(
            destino / "02-completeness",
            DocumentoCompletude(
                documento_id=documento_id,
                texto=texto,
                candidatos=analisados,
            ),
        )
        _salvar(
            destino / "03-veracity",
            DocumentoClassificado(
                documento_id=documento_id,
                texto=texto,
                candidatos=classificados,
            ),
        )
        citacoes = []
        for linha in linhas.itertuples():
            id_canonico = linha.id_canonico if linha.classificacao == "real" else "-"
            citacoes.append(
                f"{linha.inicio},{linha.fim},{linha.classificacao},{id_canonico},1.0000"
            )
        submissao.append((documento_id, "|".join(citacoes)))

    destino.mkdir(parents=True, exist_ok=True)
    with (destino / "submission_gold.csv").open(
        "w", newline="", encoding="utf-8"
    ) as arquivo:
        escritor = csv.writer(arquivo)
        escritor.writerow(["documento_id", "citacoes"])
        escritor.writerows(submissao)

    for etapa in ETAPAS:
        arquivos = list((destino / etapa).glob("*.json"))
        print(f"{destino / etapa}: {len(arquivos)} documentos")
    print(f"citações: {len(gold)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("desafio-jusbrasil-bracis-2026"),
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/gold"))
    args = parser.parse_args()
    gerar(args.dataset, args.output)


if __name__ == "__main__":
    main()
