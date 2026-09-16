"""Executa o pipeline completo e materializa as predições finais."""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from .completeness import AgenteCompletude, executar_completude
from .contracts import (
    ClassificadorCompletude,
    ClassificadorVeracidade,
    ExtratorCandidatos,
    PipelineConfig,
)
from .extractor import AgenteExtrator, executar_extracao
from .veracity import (
    AgenteVeracidade,
    criar_modelo_veracidade,
    executar_veracidade,
    materializar,
)


class Orquestrador:
    def __init__(
        self,
        extrator: ExtratorCandidatos,
        completude: ClassificadorCompletude,
        veracidade: ClassificadorVeracidade,
    ) -> None:
        self._extrator = extrator
        self._completude = completude
        self._veracidade = veracidade

    def executar(self, pasta_txt: Path, workdir: Path) -> None:
        workdir.mkdir(parents=True, exist_ok=True)
        extracao = workdir / "01-extraction"
        completude = workdir / "02-completeness"
        veracidade = workdir / "03-veracity"

        executar_extracao(pasta_txt, extracao, self._extrator)
        executar_completude(extracao, completude, self._completude)
        executar_veracidade(completude, veracidade, self._veracidade)
        materializar(veracidade, workdir / "predictions")


def main() -> None:
    load_dotenv()
    config = PipelineConfig.from_yaml(Path("pipeline.yaml"))
    orquestrador = Orquestrador(
        AgenteExtrator(
            OpenAI(base_url=config.extractor.base_url),
            config.extractor,
        ),
        AgenteCompletude(
            OpenAI(base_url=config.completeness.base_url),
            config.completeness,
        ),
        AgenteVeracidade(
            criar_modelo_veracidade(config.veracity),
            config.database,
        ),
    )
    orquestrador.executar(config.input_dir, config.workdir)
    print(f"{config.workdir}: pipeline concluído")


if __name__ == "__main__":
    main()
