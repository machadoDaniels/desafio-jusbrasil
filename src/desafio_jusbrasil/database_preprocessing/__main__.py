"""CLI do pré-processamento do banco canônico."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import yaml
from dotenv import load_dotenv

from .pipeline import executar


def _argumentos(argumentos: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Enriquece uma cópia auditável do banco canônico Jusbrasil."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("desafio-jusbrasil-bracis-2026/desafio1_bracis.db"),
        help=(
            "Banco SQLite de origem "
            "(padrão: desafio-jusbrasil-bracis-2026/desafio1_bracis.db)"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/desafio1_bracis_enriched.db"),
        help="Banco SQLite enriquecido (padrão: data/desafio1_bracis_enriched.db)",
    )
    parser.add_argument(
        "--audit-dir",
        type=Path,
        default=Path("outputs/database-preprocessing/run-001"),
        help="Diretório para manifest, checkpoints, relatório e revisão",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Permite substituir apenas o banco de saída existente",
    )
    return parser.parse_args(argumentos)


def _carregar_configuracao(caminho: Path = Path("pipeline.yaml")) -> dict:
    with caminho.open(encoding="utf-8") as arquivo:
        dados = yaml.safe_load(arquivo) or {}
    configuracao = dados.get("database_preprocessing")
    if not isinstance(configuracao, dict):
        raise TypeError("pipeline.yaml deve conter a seção database_preprocessing")
    return configuracao


def main(argumentos: Sequence[str] | None = None) -> None:
    args = _argumentos(argumentos)
    origem = args.input.resolve()
    destino = args.output.resolve()
    if origem == destino:
        raise ValueError("--input e --output não podem apontar para o mesmo banco")
    if not origem.is_file():
        raise FileNotFoundError(f"banco de entrada não encontrado: {origem}")
    if destino.exists() and not args.force:
        raise FileExistsError("banco de saída já existe; use --force para substituí-lo")

    load_dotenv()
    relatorio = executar(
        origem=origem,
        destino=destino,
        diretorio_auditoria=args.audit_dir.resolve(),
        configuracao=_carregar_configuracao(),
        force=args.force,
    )
    print(
        f"{destino}: {relatorio['processados']} processados, "
        f"{relatorio['reutilizados']} reutilizados"
    )


if __name__ == "__main__":
    main()
