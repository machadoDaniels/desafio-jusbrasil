"""Etapa 1: extrai candidatos de documentos jurídicos."""

from __future__ import annotations

import asyncio
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI, OpenAI, omit
from openai.types.chat import ChatCompletionMessageParam
from tqdm import tqdm

from .contracts import (
    CandidatoCitacao,
    CandidatoCitacaoRequest,
    DocumentoExtraido,
    ExtratorCandidatos,
    ExtratorCandidatosAsync,
    LoteCandidatosRequest,
    ModelConfig,
    PipelineConfig,
)

_SYSTEM_PROMPT = """Você extrai citações de documentos jurídicos brasileiros.
Retorne todas as citações a jurisprudência, súmulas e dispositivos legais.
Retorne somente o trecho literal, o tipo e, opcionalmente, a confiança.
Não calcule nem retorne posições, índices, offsets, início ou fim.
Não extraia números do processo do próprio cabeçalho, números de OAB,
protocolos, valores monetários ou referências sem natureza jurídica.
Classifique como jurisprudencia ou lei. Não avalie veracidade nesta etapa."""


class AgenteExtrator:
    """Extrai candidatos com saída estruturada e corrige seus spans localmente."""

    def __init__(
        self,
        cliente: OpenAI | AsyncOpenAI,
        config: ModelConfig,
    ) -> None:
        self._cliente = cliente
        self._config = config

    def extrair(self, texto: str) -> list[CandidatoCitacao]:
        if not isinstance(self._cliente, OpenAI):
            raise TypeError("extrair() exige um cliente OpenAI síncrono")
        candidatos = self._consultar_modelo(texto)
        return self._processar_candidatos(texto, candidatos)

    async def extrair_async(self, texto: str) -> list[CandidatoCitacao]:
        if not isinstance(self._cliente, AsyncOpenAI):
            raise TypeError("extrair_async() exige um cliente AsyncOpenAI")
        candidatos = await self._consultar_modelo_async(texto)
        return self._processar_candidatos(texto, candidatos)

    def _mensagens(self, texto: str) -> list[ChatCompletionMessageParam]:
        return [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Extraia as citações deste documento:\n\n{texto}",
            },
        ]

    def _consultar_modelo(self, texto: str) -> list[CandidatoCitacaoRequest]:
        assert isinstance(self._cliente, OpenAI)
        resposta = self._cliente.chat.completions.parse(
            model=self._config.model,
            temperature=(
                self._config.temperature
                if self._config.temperature is not None
                else omit
            ),
            top_p=self._config.top_p if self._config.top_p is not None else omit,
            messages=self._mensagens(texto),
            response_format=LoteCandidatosRequest,
            reasoning_effort=(
                self._config.reasoning_effort
                if self._config.reasoning_effort is not None
                else omit
            ),
            extra_body=(
                {"top_k": self._config.top_k}
                if self._config.top_k is not None
                else None
            ),
        )
        return self._obter_candidatos(resposta.choices[0].message.parsed)

    async def _consultar_modelo_async(
        self,
        texto: str,
    ) -> list[CandidatoCitacaoRequest]:
        assert isinstance(self._cliente, AsyncOpenAI)
        resposta = await self._cliente.chat.completions.parse(
            model=self._config.model,
            temperature=(
                self._config.temperature
                if self._config.temperature is not None
                else omit
            ),
            top_p=self._config.top_p if self._config.top_p is not None else omit,
            messages=self._mensagens(texto),
            response_format=LoteCandidatosRequest,
            reasoning_effort=(
                self._config.reasoning_effort
                if self._config.reasoning_effort is not None
                else omit
            ),
            extra_body=(
                {"top_k": self._config.top_k}
                if self._config.top_k is not None
                else None
            ),
        )
        return self._obter_candidatos(resposta.choices[0].message.parsed)

    @staticmethod
    def _obter_candidatos(
        resultado: LoteCandidatosRequest | None,
    ) -> list[CandidatoCitacaoRequest]:
        if resultado is None:
            raise RuntimeError("o modelo não retornou uma extração estruturada")
        return resultado.candidatos

    def _processar_candidatos(
        self,
        texto: str,
        candidatos: list[CandidatoCitacaoRequest],
    ) -> list[CandidatoCitacao]:
        candidatos_com_spans = self._adicionar_spans(texto, candidatos)
        return self._remover_duplicatas(candidatos_com_spans)

    @staticmethod
    def _adicionar_spans(
        texto: str,
        candidatos: list[CandidatoCitacaoRequest],
    ) -> list[CandidatoCitacao]:
        encontrados = []
        for candidato in candidatos:
            inicio = texto.find(candidato.trecho)
            while inicio >= 0:
                encontrados.append(
                    CandidatoCitacao(
                        **candidato.model_dump(),
                        inicio=inicio,
                        fim=inicio + len(candidato.trecho),
                    )
                )
                inicio = texto.find(candidato.trecho, inicio + 1)
        return encontrados

    @staticmethod
    def _remover_duplicatas(
        candidatos: list[CandidatoCitacao],
    ) -> list[CandidatoCitacao]:
        priorizados = sorted(
            candidatos,
            key=lambda candidato: (
                -(candidato.confianca_extracao or 0.0),
                candidato.inicio,
                candidato.fim,
            ),
        )
        unicos: list[CandidatoCitacao] = []
        for candidato in priorizados:
            if not any(_iou(candidato, existente) >= 0.5 for existente in unicos):
                unicos.append(candidato)
        return sorted(unicos, key=lambda candidato: (candidato.inicio, candidato.fim))


def _iou(a: CandidatoCitacao, b: CandidatoCitacao) -> float:
    intersecao = max(0, min(a.fim, b.fim) - max(a.inicio, b.inicio))
    if intersecao == 0:
        return 0.0
    uniao = (a.fim - a.inicio) + (b.fim - b.inicio) - intersecao
    return intersecao / uniao


def _escrever_jsons(documentos: list[DocumentoExtraido], destino: Path) -> None:
    destino.mkdir(parents=True, exist_ok=True)
    for documento in documentos:
        caminho = destino / f"{documento.documento_id}.json"
        caminho.write_text(
            documento.model_dump_json(indent=2, exclude_none=True) + "\n",
            encoding="utf-8",
        )


def executar_extracao(
    input_dir: Path,
    output_file: Path,
    extrator: ExtratorCandidatos,
) -> None:
    arquivos = sorted(input_dir.glob("*.txt"))
    if not arquivos:
        raise ValueError(f"nenhum arquivo .txt encontrado em {input_dir}")

    documentos = []
    for arquivo in tqdm(arquivos, desc="Extraindo candidatos"):
        texto = arquivo.read_text(encoding="utf-8")
        documentos.append(
            DocumentoExtraido(
                documento_id=arquivo.stem,
                texto=texto,
                candidatos=extrator.extrair(texto),
            )
        )
    _escrever_jsons(documentos, output_file)


async def executar_extracao_async(
    input_dir: Path,
    output_file: Path,
    extrator: ExtratorCandidatosAsync,
    max_concurrency: int,
) -> None:
    arquivos = sorted(input_dir.glob("*.txt"))
    if not arquivos:
        raise ValueError(f"nenhum arquivo .txt encontrado em {input_dir}")

    semaforo = asyncio.Semaphore(max_concurrency)
    progresso = tqdm(total=len(arquivos), desc="Extraindo candidatos")

    async def processar(arquivo: Path) -> DocumentoExtraido:
        async with semaforo:
            texto = arquivo.read_text(encoding="utf-8")
            candidatos = await extrator.extrair_async(texto)
        progresso.update()
        return DocumentoExtraido(
            documento_id=arquivo.stem,
            texto=texto,
            candidatos=candidatos,
        )

    try:
        documentos = await asyncio.gather(*(processar(arquivo) for arquivo in arquivos))
    finally:
        progresso.close()
    _escrever_jsons(documentos, output_file)


async def _executar_extracao_async(config: PipelineConfig, destino: Path) -> None:
    async with AsyncOpenAI(base_url=config.extractor.base_url) as cliente:
        await executar_extracao_async(
            config.input_dir,
            destino,
            AgenteExtrator(cliente, config.extractor),
            config.extractor.max_concurrency,
        )


def main() -> None:
    load_dotenv()
    config = PipelineConfig.from_yaml(Path("pipeline.yaml"))
    destino = config.workdir / "01-extraction"
    if config.extractor.async_requests:
        asyncio.run(_executar_extracao_async(config, destino))
    else:
        executar_extracao(
            config.input_dir,
            destino,
            AgenteExtrator(
                OpenAI(base_url=config.extractor.base_url),
                config.extractor,
            ),
        )
    print(f"{destino}: extração concluída")


if __name__ == "__main__":
    main()
