"""CLI do pré-processamento do banco canônico."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from dotenv import load_dotenv

from .pipeline import executar


def _carregar_configuracao(caminho: Path = Path("pipeline.yaml")) -> dict:
    with caminho.open(encoding="utf-8") as arquivo:
        dados = yaml.safe_load(arquivo) or {}
    configuracao = dados.get("database_preprocessing")
    if not isinstance(configuracao, dict):
        raise TypeError(f"{caminho} deve conter a seção database_preprocessing")
    return configuracao


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("pipeline.yaml"))
    args = parser.parse_args()
    load_dotenv()
    configuracao = _carregar_configuracao(args.config)
    origem = Path(configuracao["input"]).resolve()
    destino = Path(configuracao["output"]).resolve()
    diretorio_auditoria = Path(configuracao["audit_dir"]).resolve()
    relatorio = executar(
        origem=origem,
        destino=destino,
        diretorio_auditoria=diretorio_auditoria,
        configuracao=configuracao,
    )
    print(
        f"{destino}: {relatorio['processados']} processados, "
        f"{relatorio['reutilizados']} reutilizados"
    )


if __name__ == "__main__":
    main()
