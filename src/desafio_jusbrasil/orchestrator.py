"""Executa o pipeline completo e materializa as predições finais."""

from __future__ import annotations

from importlib import import_module
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

from .contracts import (
    Classificacao,
    ClassificadorCompletude,
    ClassificadorVeracidade,
    DocumentoClassificado,
    DocumentoPredito,
    ExtratorCandidatos,
    PipelineConfig,
    Predicao,
    Resolucao,
)
from .utils import escrever_manifesto_etapa, listar_resultados

_extractor = import_module(".1_extractor", __package__)
_completeness = import_module(".2_completeness", __package__)
_veracity = import_module(".3_veracity", __package__)
AgenteExtrator = _extractor.AgenteExtrator
executar_extracao = _extractor.executar_extracao
AgenteCompletude = _completeness.AgenteCompletude
executar_completude = _completeness.executar_completude
VerificadorVeracidade = _veracity.VerificadorVeracidade
executar_veracidade = _veracity.executar_veracidade


def materializar(entrada: Path, pasta_saida: Path) -> None:
    arquivos = listar_resultados(entrada)
    pasta_saida.mkdir(parents=True, exist_ok=True)
    for arquivo in tqdm(arquivos, desc="Materializando predições"):
        documento = DocumentoClassificado.model_validate_json(
            arquivo.read_text(encoding="utf-8")
        )
        citacoes = []
        for item in documento.candidatos:
            candidato = item.candidato
            if candidato.inicio is None or candidato.fim is None:
                continue
            resultado = item.veracidade
            resolucao = None
            if resultado.classificacao == Classificacao.REAL:
                assert resultado.id_canonico is not None
                resolucao = Resolucao(id_canonico=resultado.id_canonico)
            citacoes.append(
                Predicao(
                    inicio=candidato.inicio,
                    fim=candidato.fim,
                    trecho=candidato.trecho,
                    tipo=candidato.tipo,
                    classificacao=resultado.classificacao,
                    resolucao=resolucao,
                    confianca=resultado.confianca,
                )
            )
        predicao = DocumentoPredito(
            documento_id=documento.documento_id,
            citacoes=citacoes,
        )
        (pasta_saida / f"{documento.documento_id}.json").write_text(
            predicao.model_dump_json(indent=2, exclude_none=True) + "\n",
            encoding="utf-8",
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
        executar_completude(extracao, completude, self._completude, pasta_txt)
        executar_veracidade(completude, veracidade, self._veracidade)
        materializar(veracidade, workdir / "predictions")


def main() -> None:
    load_dotenv()
    config = PipelineConfig.from_yaml(Path("pipeline.yaml"))
    escrever_manifesto_etapa(
        config.workdir / "01-extraction", "extractor", config.extractor
    )
    escrever_manifesto_etapa(
        config.workdir / "02-completeness", "completeness", config.completeness
    )
    escrever_manifesto_etapa(
        config.workdir / "03-veracity", "veracity", config.veracity
    )
    orquestrador = Orquestrador(
        AgenteExtrator(
            OpenAI(base_url=config.extractor.base_url),
            config.extractor,
        ),
        AgenteCompletude(
            OpenAI(base_url=config.completeness.base_url),
            config.completeness,
        ),
        VerificadorVeracidade(
            OpenAI(base_url=config.veracity.base_url),
            config.veracity,
            config.database,
        ),
    )
    orquestrador.executar(config.input_dir, config.workdir)
    print(f"{config.workdir}: pipeline concluído")


if __name__ == "__main__":
    main()
