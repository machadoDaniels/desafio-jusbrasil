"""Executa o pipeline completo e materializa as predições finais."""

from __future__ import annotations

import argparse
import asyncio
from importlib import import_module
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI, OpenAI
from tqdm import tqdm

from .contracts import (
    Classificacao,
    ClassificadorCompletude,
    DocumentoClassificado,
    DocumentoPredito,
    ExtratorCandidatos,
    ExtratorEntidadesAsync,
    PipelineConfig,
    Predicao,
    Resolucao,
    VerificadorConsulta,
)
from .utils import escrever_manifesto_etapa, listar_resultados, materializar

_extractor = import_module(".1_extractor", __package__)
_completeness = import_module(".2_completeness", __package__)
_entities = import_module(".3_entities", __package__)
_veracity = import_module(".4_veracity", __package__)
AgenteExtrator = _extractor.AgenteExtrator
executar_extracao = _extractor.executar_extracao
AgenteCompletude = _completeness.AgenteCompletude
executar_completude = _completeness.executar_completude
AgenteExtratorEntidades = _entities.AgenteExtratorEntidades
executar_entities_async = _entities.executar_entities_async
VerificadorVeracidade = _veracity.VerificadorVeracidade
executar_veracidade = _veracity.executar_veracidade


class Orquestrador:
    def __init__(
        self,
        extrator: ExtratorCandidatos,
        completude: ClassificadorCompletude,
        extrator_consulta: ExtratorEntidadesAsync,
        verificador: VerificadorConsulta,
        entities_max_concurrency: int = 10,
    ) -> None:
        self._extrator = extrator
        self._completude = completude
        self._extrator_consulta = extrator_consulta
        self._verificador = verificador
        self._entities_max_concurrency = entities_max_concurrency

    async def executar(self, pasta_txt: Path, workdir: Path) -> None:
        workdir.mkdir(parents=True, exist_ok=True)
        extracao = workdir / "01-extraction"
        completude = workdir / "02-completeness"
        entities = workdir / "03-entities"
        veracidade = workdir / "04-veracity"

        executar_extracao(pasta_txt, extracao, self._extrator)
        executar_completude(extracao, completude, self._completude, pasta_txt)
        await executar_entities_async(
            completude,
            entities,
            self._extrator_consulta,
            max_concurrency=self._entities_max_concurrency,
        )
        executar_veracidade(entities, veracidade, self._verificador)
        materializar(veracidade, workdir / "predictions")


async def _main(
    caminho_configuracao: Path, substituicoes: dict[str, Path] | None = None
) -> None:
    load_dotenv()
    config = PipelineConfig.from_yaml(caminho_configuracao).model_copy(
        update=substituicoes or {}
    )
    escrever_manifesto_etapa(
        config.workdir / "01-extraction", "extractor", config.extractor
    )
    escrever_manifesto_etapa(
        config.workdir / "02-completeness", "completeness", config.completeness
    )
    escrever_manifesto_etapa(
        config.workdir / "03-entities", "entities", config.entities
    )
    escrever_manifesto_etapa(config.workdir / "04-veracity", "veracity")
    async with AsyncOpenAI(base_url=config.entities.base_url, max_retries=0) as cliente_entities:
        orquestrador = Orquestrador(
            AgenteExtrator(
                OpenAI(base_url=config.extractor.base_url),
                config.extractor,
            ),
            AgenteCompletude(
                OpenAI(base_url=config.completeness.base_url),
                config.completeness,
            ),
            AgenteExtratorEntidades.from_config(cliente_entities, config),
            VerificadorVeracidade(config.database),
            config.entities.max_concurrency,
        )
        await orquestrador.executar(config.input_dir, config.workdir)
    print(f"{config.workdir}: pipeline concluído")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("pipeline.yaml"))
    parser.add_argument("--input-dir", type=Path, help="substitui input_dir do YAML")
    parser.add_argument("--workdir", type=Path, help="substitui workdir do YAML")
    parser.add_argument("--database", type=Path, help="substitui database do YAML")
    args = parser.parse_args()
    substituicoes = {
        chave: valor
        for chave, valor in (
            ("input_dir", args.input_dir),
            ("workdir", args.workdir),
            ("database", args.database),
        )
        if valor is not None
    }
    asyncio.run(_main(args.config, substituicoes))


if __name__ == "__main__":
    main()
