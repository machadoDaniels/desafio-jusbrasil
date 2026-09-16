"""Etapa 2: decide se cada citação permite uma consulta canônica."""

from __future__ import annotations

from pathlib import Path
from tempfile import NamedTemporaryFile

from openai import OpenAI

from .contracts import (
    CandidatoAnalisado,
    CandidatoCitacao,
    ClassificadorCompletude,
    DocumentoCompletude,
    DocumentoExtraido,
    PipelineConfig,
    ResultadoCompletude,
)

_SYSTEM_PROMPT = """Você avalia citações jurídicas brasileiras.
Decida se o trecho e seu contexto fornecem informação suficiente para formular
uma consulta específica a uma base canônica. Extraia somente dados sustentados
pelo texto. Não decida se a citação existe. Uma descrição vaga, sem identificador
suficiente, é incompleta. Se completa=false, consulta deve ser null."""


class AgenteCompletude:
    """Classifica completude por uma chamada estruturada ao modelo."""

    def __init__(
        self,
        cliente: OpenAI,
        modelo: str,
        temperature: float = 0,
        reasoning_effort: str | None = None,
    ) -> None:
        self._cliente = cliente
        self._modelo = modelo
        self._temperature = temperature
        self._reasoning_effort = reasoning_effort

    def classificar(
        self,
        candidato: CandidatoCitacao,
        contexto: str,
    ) -> ResultadoCompletude:
        return self._consultar_modelo(candidato, contexto)

    def _consultar_modelo(
        self,
        candidato: CandidatoCitacao,
        contexto: str,
    ) -> ResultadoCompletude:
        parametros = {}
        if self._reasoning_effort is not None:
            parametros["reasoning_effort"] = self._reasoning_effort
        resposta = self._cliente.chat.completions.parse(
            model=self._modelo,
            temperature=self._temperature,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        "Candidato:\n"
                        f"{candidato.model_dump_json(exclude_none=True)}\n\n"
                        f"Contexto:\n{contexto}"
                    ),
                },
            ],
            response_format=ResultadoCompletude,
            **parametros,
        )
        resultado = resposta.choices[0].message.parsed
        if resultado is None:
            raise RuntimeError("o modelo não retornou uma análise estruturada")
        return resultado


def _obter_contexto(
    texto: str,
    candidato: CandidatoCitacao,
    margem: int = 300,
) -> str:
    inicio = max(0, candidato.inicio - margem)
    fim = min(len(texto), candidato.fim + margem)
    return texto[inicio:fim]


def _ler_jsonl(caminho: Path) -> list[DocumentoExtraido]:
    documentos = []
    with caminho.open(encoding="utf-8") as arquivo:
        for numero, linha in enumerate(arquivo, start=1):
            if linha.strip():
                try:
                    documentos.append(DocumentoExtraido.model_validate_json(linha))
                except ValueError as erro:
                    raise ValueError(f"{caminho}:{numero}: {erro}") from erro
    return documentos


def _escrever_jsonl(documentos: list[DocumentoCompletude], destino: Path) -> None:
    destino.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        "w", encoding="utf-8", dir=destino.parent, delete=False
    ) as temporario:
        caminho_temporario = Path(temporario.name)
        for documento in documentos:
            temporario.write(documento.model_dump_json(exclude_none=True) + "\n")
    caminho_temporario.replace(destino)


def executar_completude(
    input_file: Path,
    output_file: Path,
    classificador: ClassificadorCompletude,
) -> None:
    saida = []
    for documento in _ler_jsonl(input_file):
        candidatos = [
            CandidatoAnalisado(
                candidato=candidato,
                completude=classificador.classificar(
                    candidato,
                    _obter_contexto(documento.texto, candidato),
                ),
            )
            for candidato in documento.candidatos
        ]
        saida.append(
            DocumentoCompletude(
                documento_id=documento.documento_id,
                texto=documento.texto,
                candidatos=candidatos,
            )
        )
    _escrever_jsonl(saida, output_file)


def main() -> None:
    config = PipelineConfig.from_yaml(Path("pipeline.yaml"))
    entrada = config.workdir / "01-extraction.jsonl"
    destino = config.workdir / "02-completeness.jsonl"
    executar_completude(
        entrada,
        destino,
        AgenteCompletude(
            OpenAI(base_url=config.base_url),
            config.model,
            config.temperature,
            config.reasoning_effort,
        ),
    )
    print(f"{destino}: completude concluída")


if __name__ == "__main__":
    main()
