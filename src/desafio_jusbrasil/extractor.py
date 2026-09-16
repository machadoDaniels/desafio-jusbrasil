"""Etapa 1: extrai candidatos de documentos jurídicos."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import AsyncOpenAI, OpenAI, omit
from openai.types.chat import ChatCompletionMessageParam
from tqdm import tqdm

from .contracts import (
    AuditoriaChamadaModelo,
    CandidatoCitacao,
    CandidatoCitacaoRequest,
    DocumentoExtraido,
    ExtratorCandidatos,
    ExtratorCandidatosAsync,
    LoteCandidatosRequest,
    ModelConfig,
    PipelineConfig,
    escrever_manifesto_etapa,
)

_SYSTEM_PROMPT = """Você extrai citações de documentos jurídicos brasileiros.
Retorne todas as citações a jurisprudência, súmulas e dispositivos legais.
Retorne somente o trecho literal verbatim(não remova nada, nem marcadores e formatação), o tipo e, opcionalmente, a confiança.
Classifique como jurisprudencia ou lei."""


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
        return self.extrair_auditada(texto)[0]

    def extrair_auditada(
        self,
        texto: str,
    ) -> tuple[list[CandidatoCitacao], AuditoriaChamadaModelo]:
        if not isinstance(self._cliente, OpenAI):
            raise TypeError("extrair() exige um cliente OpenAI síncrono")
        candidatos, auditoria = self._consultar_modelo(texto)
        return self._processar_candidatos(texto, candidatos), auditoria

    async def extrair_async(self, texto: str) -> list[CandidatoCitacao]:
        candidatos, _ = await self.extrair_auditada_async(texto)
        return candidatos

    async def extrair_auditada_async(
        self,
        texto: str,
    ) -> tuple[list[CandidatoCitacao], AuditoriaChamadaModelo]:
        if not isinstance(self._cliente, AsyncOpenAI):
            raise TypeError("extrair_async() exige um cliente AsyncOpenAI")
        candidatos, auditoria = await self._consultar_modelo_async(texto)
        return self._processar_candidatos(texto, candidatos), auditoria

    def _mensagens(self, texto: str) -> list[ChatCompletionMessageParam]:
        return [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Extraia as citações deste documento:\n\n{texto}",
            },
        ]

    def _consultar_modelo(
        self, texto: str
    ) -> tuple[list[CandidatoCitacaoRequest], AuditoriaChamadaModelo]:
        assert isinstance(self._cliente, OpenAI)
        requisicao = self._requisicao(texto)
        resposta = self._cliente.chat.completions.parse(**requisicao)
        resultado = self._obter_candidatos(resposta.choices[0].message.parsed)
        return resultado, self._auditoria(requisicao, resposta, resultado)

    def _requisicao(self, texto: str) -> dict[str, Any]:
        requisicao: dict[str, Any] = {
            "model": self._config.model,
            "temperature": self._config.temperature
            if self._config.temperature is not None
            else omit,
            "top_p": self._config.top_p if self._config.top_p is not None else omit,
            "messages": self._mensagens(texto),
            "response_format": LoteCandidatosRequest,
            "reasoning_effort": (
                self._config.reasoning_effort
                if self._config.reasoning_effort is not None
                else omit
            ),
        }
        if self._config.top_k is not None:
            requisicao["extra_body"] = {"top_k": self._config.top_k}
        return requisicao

    @staticmethod
    def _auditoria(
        requisicao: dict,
        resposta: Any,
        resultado: list[CandidatoCitacaoRequest],
    ) -> AuditoriaChamadaModelo:
        entrada = {
            nome: valor for nome, valor in requisicao.items() if valor is not omit
        }
        entrada["response_format"] = LoteCandidatosRequest.model_json_schema()
        return AuditoriaChamadaModelo(
            input=entrada,
            output={
                "bruta": resposta.model_dump(
                    mode="json",
                    exclude={"choices": {"__all__": {"message": {"parsed"}}}},
                ),
                "estruturada": [item.model_dump(mode="json") for item in resultado],
            },
        )

    async def _consultar_modelo_async(
        self,
        texto: str,
    ) -> tuple[list[CandidatoCitacaoRequest], AuditoriaChamadaModelo]:
        assert isinstance(self._cliente, AsyncOpenAI)
        requisicao = self._requisicao(texto)
        resposta = await self._cliente.chat.completions.parse(**requisicao)
        resultado = self._obter_candidatos(resposta.choices[0].message.parsed)
        return resultado, self._auditoria(requisicao, resposta, resultado)

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
        return self._adicionar_spans(texto, candidatos)

    @staticmethod
    def _adicionar_spans(
        texto: str,
        candidatos: list[CandidatoCitacaoRequest],
    ) -> list[CandidatoCitacao]:
        encontrados = []
        texto_normalizado, mapa = _normalizar_espacos_com_mapa(texto)
        for candidato in candidatos:
            spans = _localizar_todas_ocorrencias(texto, candidato.trecho)
            if not spans:
                trecho_normalizado = _normalizar_espacos(candidato.trecho)
                spans = _localizar_no_texto_normalizado(
                    texto_normalizado,
                    mapa,
                    trecho_normalizado,
                )
            if not spans:
                encontrados.append(CandidatoCitacao(**candidato.model_dump()))
                continue
            for inicio, fim in spans:
                encontrados.append(
                    CandidatoCitacao(
                        **candidato.model_dump(exclude={"trecho"}),
                        trecho=texto[inicio:fim],
                        inicio=inicio,
                        fim=fim,
                    )
                )
        return encontrados


def _localizar_todas_ocorrencias(texto: str, trecho: str) -> list[tuple[int, int]]:
    spans = []
    inicio = texto.find(trecho)
    while inicio >= 0:
        spans.append((inicio, inicio + len(trecho)))
        inicio = texto.find(trecho, inicio + 1)
    return spans


def _normalizar_espacos(texto: str) -> str:
    return " ".join(texto.split())


def _normalizar_espacos_com_mapa(
    texto: str,
) -> tuple[str, list[tuple[int, int]]]:
    caracteres = []
    mapa = []
    indice = 0
    while indice < len(texto):
        if texto[indice].isspace():
            inicio = indice
            while indice < len(texto) and texto[indice].isspace():
                indice += 1
            caracteres.append(" ")
            mapa.append((inicio, indice))
        else:
            caracteres.append(texto[indice])
            mapa.append((indice, indice + 1))
            indice += 1
    return "".join(caracteres), mapa


def _localizar_no_texto_normalizado(
    texto_normalizado: str,
    mapa: list[tuple[int, int]],
    trecho_normalizado: str,
) -> list[tuple[int, int]]:
    if not trecho_normalizado:
        return []
    spans = []
    inicio = texto_normalizado.find(trecho_normalizado)
    while inicio >= 0:
        fim_normalizado = inicio + len(trecho_normalizado)
        inicio_original = mapa[inicio][0]
        fim_original = mapa[fim_normalizado - 1][1]
        spans.append((inicio_original, fim_original))
        inicio = texto_normalizado.find(trecho_normalizado, inicio + 1)
    return spans


def _escrever_jsons(documentos: list[DocumentoExtraido], destino: Path) -> None:
    destino.mkdir(parents=True, exist_ok=True)
    for documento in documentos:
        caminho = destino / f"{documento.documento_id}.json"
        caminho.write_text(
            documento.model_dump_json(indent=2) + "\n",
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
        candidatos, chamada = extrator.extrair_auditada(texto)
        documentos.append(
            DocumentoExtraido(
                documento_id=arquivo.stem,
                texto=texto,
                candidatos=candidatos,
                chamadas_modelo=[chamada],
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
            candidatos, chamada = await extrator.extrair_auditada_async(texto)
        progresso.update()
        return DocumentoExtraido(
            documento_id=arquivo.stem,
            texto=texto,
            candidatos=candidatos,
            chamadas_modelo=[chamada],
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
    escrever_manifesto_etapa(destino, "extractor", config.extractor)
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
