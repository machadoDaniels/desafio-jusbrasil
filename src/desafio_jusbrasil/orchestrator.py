"""Executa o pipeline completo e materializa as predições finais."""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

from .completeness import AgenteCompletude, executar_completude
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
    escrever_manifesto_etapa,
)
from .extractor import AgenteExtrator, executar_extracao
from .veracity import VerificadorVeracidade, executar_veracidade


def materializar(entrada: Path, pasta_saida: Path) -> None:
    arquivos = sorted(
        arquivo for arquivo in entrada.glob("*.json") if arquivo.name != "manifest.json"
    )
    if not arquivos:
        raise ValueError(f"nenhum arquivo JSON encontrado em {entrada}")
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
        (pasta_saida / arquivo.name).write_text(
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
        executar_completude(extracao, completude, self._completude)
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
