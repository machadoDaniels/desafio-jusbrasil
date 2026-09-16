"""Executa o pipeline completo e materializa as predições finais."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from openai import OpenAI

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
)
from .extractor import AgenteExtrator, executar_extracao
from .veracity import (
    AgenteVeracidade,
    criar_modelo_veracidade,
    executar_veracidade,
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
        extracao = workdir / "01-extraction.jsonl"
        completude = workdir / "02-completeness.jsonl"
        veracidade = workdir / "03-veracity.jsonl"

        executar_extracao(pasta_txt, extracao, self._extrator)
        executar_completude(extracao, completude, self._completude)
        executar_veracidade(completude, veracidade, self._veracidade)
        self.materializar(veracidade, workdir / "predictions")

    def materializar(self, entrada: Path, pasta_saida: Path) -> None:
        pasta_saida.mkdir(parents=True, exist_ok=True)
        for documento in self._ler_classificados(entrada):
            predicao = self._montar_predicao(documento)
            destino = pasta_saida / f"{documento.documento_id}.json"
            destino.write_text(
                predicao.model_dump_json(indent=2, exclude_none=True) + "\n",
                encoding="utf-8",
            )

    def _ler_classificados(self, entrada: Path) -> Iterator[DocumentoClassificado]:
        with entrada.open(encoding="utf-8") as arquivo:
            for linha in arquivo:
                if linha.strip():
                    yield DocumentoClassificado.model_validate_json(linha)

    def _montar_predicao(
        self,
        documento: DocumentoClassificado,
    ) -> DocumentoPredito:
        citacoes = []
        for item in documento.candidatos:
            resultado = item.veracidade
            resolucao = None
            if resultado.classificacao == Classificacao.REAL:
                resolucao = Resolucao(id_canonico=resultado.id_canonico)
            citacoes.append(
                Predicao(
                    inicio=item.candidato.inicio,
                    fim=item.candidato.fim,
                    trecho=item.candidato.trecho,
                    tipo=item.candidato.tipo,
                    classificacao=resultado.classificacao,
                    resolucao=resolucao,
                    confianca=resultado.confianca,
                )
            )
        return DocumentoPredito(
            documento_id=documento.documento_id,
            citacoes=citacoes,
        )


def main() -> None:
    config = PipelineConfig.from_yaml(Path("pipeline.yaml"))
    cliente_openai = OpenAI(base_url=config.base_url)
    orquestrador = Orquestrador(
        AgenteExtrator(
            cliente_openai,
            config.model,
            config.temperature,
            config.reasoning_effort,
        ),
        AgenteCompletude(
            cliente_openai,
            config.model,
            config.temperature,
            config.reasoning_effort,
        ),
        AgenteVeracidade(
            criar_modelo_veracidade(config),
            config.database,
        ),
    )
    orquestrador.executar(config.input_dir, config.workdir)
    print(f"{config.workdir}: pipeline concluído")


if __name__ == "__main__":
    main()
