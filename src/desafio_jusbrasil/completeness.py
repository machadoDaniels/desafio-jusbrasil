"""Etapa 2: decide se cada citação permite uma consulta canônica."""

from __future__ import annotations

import asyncio
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI, OpenAI
from openai.types.chat import ChatCompletionMessageParam
from tqdm import tqdm

from .contracts import (
    CandidatoAnalisado,
    CandidatoCitacao,
    ClassificadorCompletude,
    ClassificadorCompletudeAsync,
    DocumentoCompletude,
    DocumentoExtraido,
    ModelConfig,
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
        cliente: OpenAI | AsyncOpenAI,
        config: ModelConfig,
    ) -> None:
        self._cliente = cliente
        self._config = config

    def classificar(
        self,
        candidato: CandidatoCitacao,
        contexto: str,
    ) -> ResultadoCompletude:
        if not isinstance(self._cliente, OpenAI):
            raise TypeError("classificar() exige um cliente OpenAI síncrono")
        return self._consultar_modelo(candidato, contexto)

    async def classificar_async(
        self,
        candidato: CandidatoCitacao,
        contexto: str,
    ) -> ResultadoCompletude:
        if not isinstance(self._cliente, AsyncOpenAI):
            raise TypeError("classificar_async() exige um cliente AsyncOpenAI")
        return await self._consultar_modelo_async(candidato, contexto)

    def _mensagens(
        self,
        candidato: CandidatoCitacao,
        contexto: str,
    ) -> list[ChatCompletionMessageParam]:
        return [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "Candidato:\n"
                    f"{candidato.model_dump_json(exclude_none=True)}\n\n"
                    f"Contexto:\n{contexto}"
                ),
            },
        ]

    def _consultar_modelo(
        self,
        candidato: CandidatoCitacao,
        contexto: str,
    ) -> ResultadoCompletude:
        assert isinstance(self._cliente, OpenAI)
        resposta = self._cliente.chat.completions.parse(
            model=self._config.model,
            temperature=self._config.temperature,
            messages=self._mensagens(candidato, contexto),
            response_format=ResultadoCompletude,
            reasoning_effort=self._config.reasoning_effort,
        )
        return self._obter_resultado(resposta.choices[0].message.parsed)

    async def _consultar_modelo_async(
        self,
        candidato: CandidatoCitacao,
        contexto: str,
    ) -> ResultadoCompletude:
        assert isinstance(self._cliente, AsyncOpenAI)
        resposta = await self._cliente.chat.completions.parse(
            model=self._config.model,
            temperature=self._config.temperature,
            messages=self._mensagens(candidato, contexto),
            response_format=ResultadoCompletude,
            reasoning_effort=self._config.reasoning_effort,
        )
        return self._obter_resultado(resposta.choices[0].message.parsed)

    @staticmethod
    def _obter_resultado(
        resultado: ResultadoCompletude | None,
    ) -> ResultadoCompletude:
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


def _ler_jsons(caminho: Path) -> list[DocumentoExtraido]:
    arquivos = sorted(caminho.glob("*.json"))
    if not arquivos:
        raise ValueError(f"nenhum arquivo JSON encontrado em {caminho}")
    documentos = []
    for arquivo in arquivos:
        try:
            documentos.append(
                DocumentoExtraido.model_validate_json(
                    arquivo.read_text(encoding="utf-8")
                )
            )
        except ValueError as erro:
            raise ValueError(f"{arquivo}: {erro}") from erro
    return documentos


def _escrever_jsons(documentos: list[DocumentoCompletude], destino: Path) -> None:
    destino.mkdir(parents=True, exist_ok=True)
    for documento in documentos:
        caminho = destino / f"{documento.documento_id}.json"
        caminho.write_text(
            documento.model_dump_json(indent=2, exclude_none=True) + "\n",
            encoding="utf-8",
        )


def executar_completude(
    input_file: Path,
    output_file: Path,
    classificador: ClassificadorCompletude,
) -> None:
    documentos = _ler_jsons(input_file)
    total = sum(len(documento.candidatos) for documento in documentos)
    saida = []
    with tqdm(total=total, desc="Avaliando completude") as progresso:
        for documento in documentos:
            candidatos = []
            for candidato in documento.candidatos:
                candidatos.append(
                    CandidatoAnalisado(
                        candidato=candidato,
                        completude=classificador.classificar(
                            candidato,
                            _obter_contexto(documento.texto, candidato),
                        ),
                    )
                )
                progresso.update()
            saida.append(
                DocumentoCompletude(
                    documento_id=documento.documento_id,
                    texto=documento.texto,
                    candidatos=candidatos,
                )
            )
    _escrever_jsons(saida, output_file)


async def executar_completude_async(
    input_file: Path,
    output_file: Path,
    classificador: ClassificadorCompletudeAsync,
    max_concurrency: int,
) -> None:
    documentos = _ler_jsons(input_file)
    total = sum(len(documento.candidatos) for documento in documentos)
    semaforo = asyncio.Semaphore(max_concurrency)
    progresso = tqdm(total=total, desc="Avaliando completude")

    async def processar(documento: DocumentoExtraido) -> DocumentoCompletude:
        async def classificar(candidato: CandidatoCitacao) -> CandidatoAnalisado:
            async with semaforo:
                resultado = await classificador.classificar_async(
                    candidato,
                    _obter_contexto(documento.texto, candidato),
                )
            progresso.update()
            return CandidatoAnalisado(candidato=candidato, completude=resultado)

        candidatos = await asyncio.gather(
            *(classificar(candidato) for candidato in documento.candidatos)
        )
        return DocumentoCompletude(
            documento_id=documento.documento_id,
            texto=documento.texto,
            candidatos=candidatos,
        )

    try:
        saida = await asyncio.gather(
            *(processar(documento) for documento in documentos)
        )
    finally:
        progresso.close()
    _escrever_jsons(saida, output_file)


async def _executar_completude_async(
    config: PipelineConfig,
    entrada: Path,
    destino: Path,
) -> None:
    etapa = config.completeness
    async with AsyncOpenAI(base_url=etapa.base_url) as cliente:
        await executar_completude_async(
            entrada,
            destino,
            AgenteCompletude(cliente, etapa),
            etapa.max_concurrency,
        )


def main() -> None:
    load_dotenv()
    config = PipelineConfig.from_yaml(Path("pipeline.yaml"))
    entrada = config.workdir / "01-extraction"
    destino = config.workdir / "02-completeness"
    etapa = config.completeness
    if etapa.async_requests:
        asyncio.run(_executar_completude_async(config, entrada, destino))
    else:
        executar_completude(
            entrada,
            destino,
            AgenteCompletude(
                OpenAI(base_url=etapa.base_url),
                etapa,
            ),
        )
    print(f"{destino}: completude concluída")


if __name__ == "__main__":
    main()
