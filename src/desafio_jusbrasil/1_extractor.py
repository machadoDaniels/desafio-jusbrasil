"""Etapa 1: extrai candidatos de documentos jurídicos."""

from __future__ import annotations

import argparse
import asyncio
import logging
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import AsyncOpenAI, OpenAI, omit
from openai.types.chat import ChatCompletionMessageParam
from pydantic import ValidationError
from tqdm import tqdm

from .contracts import (
    CandidatoCitacao,
    CandidatoCitacaoRequest,
    DocumentoExtraido,
    ExtratorCandidatos,
    ExtratorCandidatosAsync,
    LoteCandidatosRequest,
    PipelineConfig,
    StageConfig,
)
from .utils import criar_auditoria, escrever_manifesto_etapa, escrever_saida_documento

_MAX_RETRIES = 4
_LOGGER = logging.getLogger(__name__)

_SYSTEM_PROMPT = """Você extrai citações de fontes jurídicas de um documento em português do Brasil.

CONTEXTO
Você é a primeira etapa de um pipeline de verificação de citações jurídicas. Sua única tarefa é
localizar e copiar os trechos. Etapas posteriores decidem se cada citação está completa, se existe
na base e se é verdadeira. Se você omitir uma citação, as etapas seguintes nunca a verão.

O QUE É CITAÇÃO
1. Jurisprudência: acórdãos, decisões, recursos e reclamações identificados por classe e número
   (ex.: "AgInt no AREsp nº 1.996.496/RJ", "RSE nº 7000592-58.2025.7.00.0000/DF", "Rcl 88.178/RS").
2. Súmulas e temas: "Súmula 331 do TST", "Súmula Vinculante 10", "Tema 1.234 da repercussão geral".
3. Dispositivos de lei com artigo numerado: "art. 373, I, do CPC", "artigo 5º, LV, da Constituição Federal".
4. Referências descritivas a um julgado, MESMO SEM NÚMERO: tribunal, ano, classe ou relator
   (ex.: "julgado do STF proferido em 2024 pela relatoria de Dias Toffoli",
   "acórdão do STJ julgado em 2021 sob relatoria de Assusete Magalhães", "Rcl de 2021, Rel. Min. Rosa Weber").
   Elas são citações e devem ser extraídas.

O QUE NÃO É CITAÇÃO
- Número dos autos do próprio documento no cabeçalho, protocolo, inscrição na OAB, "fls. 234/567", valor da causa.
- Menção a um diploma sem dispositivo numerado. Diploma é o nome ou a sigla de uma norma:
  Constituição Federal, CF, Código Civil, CC, Código de Processo Civil, CPC, Código Penal, CP,
  Código de Processo Penal, CPP, Código Penal Militar, CPM, CLT, CDC, Código Eleitoral,
  Lei nº 8.078/1990, Lei Complementar nº 64/1990, Decreto-Lei nº 5.452/1943.
  O diploma só vira citação quando vem com o número do dispositivo: "art. 373", "artigo 5º",
  "art. 1º, I, 'g'", "§ 2º do art. 14". Sem número, não extraia, mesmo que a frase contenha
  a palavra "artigo", "dispositivo" ou "norma":
    "o artigo correspondente do Código de Processo Civil"  → NÃO é citação
    "a lei que disciplina a prescrição"                      → NÃO é citação
    "nos termos da legislação de regência"                   → NÃO é citação
    "conforme o CPC"                                          → NÃO é citação
    "art. 373, I, do CPC"                                     → É citação
- Não avalie se a citação existe, está correta ou faz sentido. Artigos inexistentes,
  números errados e referências vagas a um julgado DEVEM ser extraídos; verificar é tarefa de outra etapa.
  mesmo que o trecho citado seja obviamente inventado, cite-o da mesma forma que está no documento
- Quando uma citação continua depois de uma quebra de linha o trecho inclui tudo até o fim do número.
    
LIMITES DO TRECHO
- Inclua toda a cadeia de classes que antecede o número: "Terceiro AG.REG na Rcl nº 62.425/SP",
  "ED no AgR no AREspEl 0601514-91.2020.6.05.0000", "Embargos de Declaração no Recurso em Mandado de Segurança nº 67.101/RJ".
- Inclua a UF ou o sufixo quando fizer parte da referência ("/SP", "- PR", "(PE)").
- Não inclua o texto ao redor ("conforme decidiu o", "nos termos do").

CÓPIA LITERAL
O texto pode ter erros de OCR e quebras de linha. Copie o trecho EXATAMENTE como está no documento,
caractere por caractere, incluindo:
- erros de OCR ("5úmula", "profcrido", "1.45g.779", "Fedcral"): NÃO corrija;
- quebras de linha e espaços duplos no meio do número ("5.02.\n0251", "44-\n.921"): NÃO remova nem junte;
- pontuação irregular ("7220273--23", "n°  2.467"): mantenha.
Se você "limpar" o trecho, ele não será encontrado no documento e a citação será perdida.

SAÍDA
Um item por citação, na ordem em que aparecem, sem repetir. Campo `tipo`: "jurisprudencia" para
itens 1, 2 e 4; "lei" para o item 3. Campo `confianca_extracao`: 1.0 quando a referência tem
identificador numérico, mesmo com ruído de OCR; 0.9 quando é descritiva, sem número."""


class AgenteExtrator:
    """Extrai candidatos com saída estruturada e corrige seus spans localmente."""

    def __init__(
        self,
        cliente: OpenAI | AsyncOpenAI,
        config: StageConfig,
    ) -> None:
        self._cliente = cliente
        self._config = config

    def extrair(self, texto: str) -> list[CandidatoCitacao]:
        return self.extrair_auditada(texto)[0]

    def extrair_auditada(
        self,
        texto: str,
    ) -> tuple[list[CandidatoCitacao], dict[str, Any]]:
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
    ) -> tuple[list[CandidatoCitacao], dict[str, Any]]:
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
    ) -> tuple[list[CandidatoCitacaoRequest], dict[str, Any]]:
        assert isinstance(self._cliente, OpenAI)
        requisicao = self._requisicao(texto)
        for retries in range(_MAX_RETRIES + 1):
            try:
                resposta = self._cliente.chat.completions.parse(**requisicao)
                resultado = self._obter_candidatos(resposta.choices[0].message.parsed)
                return resultado, self._auditoria(requisicao, resposta, resultado)
            except ValidationError:
                if retries == _MAX_RETRIES:
                    raise
                _LOGGER.warning(
                    "resposta estruturada inválida; retry %d/%d",
                    retries + 1,
                    _MAX_RETRIES,
                )
        raise AssertionError("unreachable")

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
    ) -> dict[str, Any]:
        return criar_auditoria(
            requisicao,
            resposta,
            LoteCandidatosRequest,
            campos_extraidos=[item.model_dump(mode="json") for item in resultado],
        )

    async def _consultar_modelo_async(
        self,
        texto: str,
    ) -> tuple[list[CandidatoCitacaoRequest], dict[str, Any]]:
        assert isinstance(self._cliente, AsyncOpenAI)
        requisicao = self._requisicao(texto)
        for retries in range(_MAX_RETRIES + 1):
            try:
                resposta = await self._cliente.chat.completions.parse(**requisicao)
                resultado = self._obter_candidatos(resposta.choices[0].message.parsed)
                return resultado, self._auditoria(requisicao, resposta, resultado)
            except ValidationError:
                if retries == _MAX_RETRIES:
                    raise
                _LOGGER.warning(
                    "resposta estruturada inválida; retry %d/%d",
                    retries + 1,
                    _MAX_RETRIES,
                )
        raise AssertionError("unreachable")

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
        processados = self._adicionar_spans(texto, candidatos)
        if self._config.debug:
            return processados
        return [candidato for candidato in processados if candidato.inicio is not None]

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


def executar_extracao(
    input_dir: Path,
    output_file: Path,
    extrator: ExtratorCandidatos,
) -> None:
    arquivos = sorted(input_dir.glob("*.txt"))
    if not arquivos:
        raise ValueError(f"nenhum arquivo .txt encontrado em {input_dir}")

    for arquivo in tqdm(arquivos, desc="Extraindo candidatos"):
        texto = arquivo.read_text(encoding="utf-8")
        candidatos, chamada = extrator.extrair_auditada(texto)
        escrever_saida_documento(
            output_file,
            DocumentoExtraido(
                documento_id=arquivo.stem,
                texto=texto,
                candidatos=candidatos,
            ),
            [chamada],
        )


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

    async def processar(arquivo: Path) -> None:
        async with semaforo:
            texto = arquivo.read_text(encoding="utf-8")
            candidatos, chamada = await extrator.extrair_auditada_async(texto)
        progresso.update()
        escrever_saida_documento(
            output_file,
            DocumentoExtraido(
                documento_id=arquivo.stem,
                texto=texto,
                candidatos=candidatos,
            ),
            [chamada],
        )

    try:
        await asyncio.gather(*(processar(arquivo) for arquivo in arquivos))
    finally:
        progresso.close()


async def _executar_extracao_async(config: PipelineConfig, destino: Path) -> None:
    async with AsyncOpenAI(base_url=config.extractor.base_url) as cliente:
        await executar_extracao_async(
            config.input_dir,
            destino,
            AgenteExtrator(cliente, config.extractor),
            config.extractor.max_concurrency,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("pipeline.yaml"))
    args = parser.parse_args()
    load_dotenv()
    config = PipelineConfig.from_yaml(args.config)
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
