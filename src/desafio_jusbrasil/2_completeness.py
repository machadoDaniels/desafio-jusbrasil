"""Etapa 2: decide se cada citação permite uma consulta canônica."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import AsyncOpenAI, OpenAI, omit
from openai.types.chat import ChatCompletionMessageParam
from pydantic import ValidationError
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
    TipoCitacao,
)
from .utils import (
    criar_auditoria,
    escrever_manifesto_etapa,
    escrever_saida_documento,
    ler_documento,
    listar_resultados,
)

_MAX_TENTATIVAS_PARSE = 3

_PROMPT_JURISPRUDENCIA = """Você avalia a completude de uma citação de jurisprudência brasileira.
Retorne somente o campo booleano completa. Não formule consultas e não avalie
veracidade.

A citação é completa quando possui identificador numerado pesquisável: número de
processo ou recurso, número de súmula ou número de tema. Classe processual,
tribunal, ano e relator sem esse número não individualizam o precedente. O ano
isolado não é número de processo.

Tolere espaços, pontuação irregular e trocas reconhecíveis entre letras e dígitos
causadas por OCR. Use o contexto somente para unir partes da mesma referência;
nunca use o número de outra citação próxima. Uma referência numerada permanece
completa mesmo que seja inexistente ou juridicamente incorreta."""

_PROMPT_LEI = """Você avalia a completude de uma citação de legislação brasileira.
Retorne somente o campo booleano completa. Não formule consultas e não avalie
veracidade.

A citação é completa quando possui um dispositivo numerado e um diploma legal
identificável. Menção genérica à legislação ou artigo sem diploma identificável é
incompleta. Tolere espaços, pontuação irregular e trocas reconhecíveis entre
letras e dígitos causadas por OCR. Use o contexto somente para unir partes da
mesma referência; nunca use dados de outra citação próxima. Uma referência
pesquisável permanece completa mesmo que seja inexistente ou juridicamente
incorreta."""


class AgenteCompletude:
    """Classifica completude por uma chamada estruturada ao modelo."""

    def __init__(
        self,
        cliente: OpenAI | AsyncOpenAI,
        config: ModelConfig,
    ) -> None:
        self._cliente = cliente
        self._config = config

    def classificar_auditada(
        self,
        candidato: CandidatoCitacao,
        contexto: str,
    ) -> tuple[ResultadoCompletude, list[dict[str, Any]]]:
        if not isinstance(self._cliente, OpenAI):
            raise TypeError("classificar_auditada() exige cliente síncrono")
        requisicao = self._requisicao(candidato, contexto)
        auditorias = []
        for tentativa in range(_MAX_TENTATIVAS_PARSE):
            try:
                resposta = self._cliente.chat.completions.parse(**requisicao)
                resultado, auditoria = self._finalizar(requisicao, resposta)
                return resultado, [*auditorias, auditoria]
            except ValidationError as erro:
                auditorias.append(self._auditoria_erro(requisicao, erro))
                if tentativa == _MAX_TENTATIVAS_PARSE - 1:
                    raise
        raise AssertionError("tentativas de parse esgotadas")

    async def classificar_auditada_async(
        self,
        candidato: CandidatoCitacao,
        contexto: str,
    ) -> tuple[ResultadoCompletude, list[dict[str, Any]]]:
        if not isinstance(self._cliente, AsyncOpenAI):
            raise TypeError("classificar_auditada_async() exige cliente assíncrono")
        requisicao = self._requisicao(candidato, contexto)
        auditorias = []
        for tentativa in range(_MAX_TENTATIVAS_PARSE):
            try:
                resposta = await self._cliente.chat.completions.parse(**requisicao)
                resultado, auditoria = self._finalizar(requisicao, resposta)
                return resultado, [*auditorias, auditoria]
            except ValidationError as erro:
                auditorias.append(self._auditoria_erro(requisicao, erro))
                if tentativa == _MAX_TENTATIVAS_PARSE - 1:
                    raise
        raise AssertionError("tentativas de parse esgotadas")

    @staticmethod
    def _mensagens(
        candidato: CandidatoCitacao,
        contexto: str,
    ) -> list[ChatCompletionMessageParam]:
        prompt = (
            _PROMPT_JURISPRUDENCIA
            if candidato.tipo == TipoCitacao.JURISPRUDENCIA
            else _PROMPT_LEI
        )
        return [
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": f"Trecho:\n{candidato.trecho}\n\nContexto:\n{contexto}",
            },
        ]

    def _requisicao(
        self,
        candidato: CandidatoCitacao,
        contexto: str,
    ) -> dict[str, Any]:
        requisicao: dict[str, Any] = {
            "model": self._config.model,
            "temperature": self._config.temperature
            if self._config.temperature is not None
            else omit,
            "top_p": self._config.top_p if self._config.top_p is not None else omit,
            "messages": self._mensagens(candidato, contexto),
            "response_format": ResultadoCompletude,
            "reasoning_effort": self._config.reasoning_effort
            if self._config.reasoning_effort is not None
            else omit,
        }
        if self._config.top_k is not None:
            requisicao["extra_body"] = {"top_k": self._config.top_k}
        return requisicao

    @staticmethod
    def _auditoria(
        requisicao: dict,
        resposta: Any,
        resultado: ResultadoCompletude,
    ) -> dict[str, Any]:
        """Preserva a chamada ao modelo em formato JSON reproduzível.

        Substitui a classe Pydantic de ``response_format`` pelo schema enviado ao
        servidor e separa a resposta bruta do resultado estruturado. O campo
        interno ``parsed`` é removido da cópia bruta porque contém um objeto
        Pydantic não pertencente à resposta HTTP e já está em ``resultado``.
        """
        return criar_auditoria(
            requisicao,
            resposta,
            ResultadoCompletude,
            resultado=resultado.model_dump(mode="json"),
        )

    @staticmethod
    def _auditoria_erro(
        requisicao: dict,
        erro: ValidationError,
    ) -> dict[str, Any]:
        entrada = {
            nome: valor for nome, valor in requisicao.items() if valor is not omit
        }
        entrada["response_format"] = ResultadoCompletude.model_json_schema()
        return {
            "input": entrada,
            "output": {"erro_validacao": erro.errors(include_input=True)},
        }

    @classmethod
    def _finalizar(
        cls,
        requisicao: dict,
        resposta: Any,
    ) -> tuple[ResultadoCompletude, dict[str, Any]]:
        resultado = resposta.choices[0].message.parsed
        if resultado is None:
            raise RuntimeError("o modelo não retornou uma análise estruturada")
        return resultado, cls._auditoria(requisicao, resposta, resultado)


def _obter_contexto(
    texto: str,
    candidato: CandidatoCitacao,
    margem: int = 300,
) -> str:
    if candidato.inicio is None or candidato.fim is None:
        return candidato.trecho
    inicio = max(0, candidato.inicio - margem)
    fim = min(len(texto), candidato.fim + margem)
    return texto[inicio:fim]


def executar_completude(
    input_file: Path,
    output_file: Path,
    classificador: ClassificadorCompletude,
    input_dir: Path,
) -> None:
    arquivos = listar_resultados(input_file)
    with tqdm(
        total=len(arquivos), desc="Avaliando completude", unit="documento"
    ) as progresso:
        for arquivo in arquivos:
            documento = ler_documento(arquivo, DocumentoExtraido)
            texto = (input_dir / f"{documento.documento_id}.txt").read_text(
                encoding="utf-8"
            )
            candidatos = []
            chamadas = []
            for candidato in documento.candidatos:
                contexto = _obter_contexto(texto, candidato)
                resultado, auditorias = classificador.classificar_auditada(
                    candidato, contexto
                )
                chamadas.extend(auditorias)
                candidatos.append(
                    CandidatoAnalisado(candidato=candidato, completude=resultado)
                )
            escrever_saida_documento(
                output_file,
                DocumentoCompletude(
                    documento_id=documento.documento_id,
                    candidatos=candidatos,
                ),
                chamadas,
            )
            progresso.update()


async def executar_completude_async(
    input_file: Path,
    output_file: Path,
    classificador: ClassificadorCompletudeAsync,
    max_concurrency: int,
    input_dir: Path,
) -> None:
    arquivos = listar_resultados(input_file)
    semaforo = asyncio.Semaphore(max_concurrency)
    progresso = tqdm(total=len(arquivos), desc="Avaliando completude", unit="documento")

    async def processar(arquivo: Path) -> None:
        documento = ler_documento(arquivo, DocumentoExtraido)
        texto = (input_dir / f"{documento.documento_id}.txt").read_text(
            encoding="utf-8"
        )

        async def classificar(
            candidato: CandidatoCitacao,
        ) -> tuple[CandidatoAnalisado, list[dict[str, Any]]]:
            contexto = _obter_contexto(texto, candidato)
            async with semaforo:
                resultado, auditorias = await classificador.classificar_auditada_async(
                    candidato, contexto
                )
            return CandidatoAnalisado(
                candidato=candidato,
                completude=resultado,
            ), auditorias

        resultados = await asyncio.gather(
            *(classificar(candidato) for candidato in documento.candidatos)
        )
        escrever_saida_documento(
            output_file,
            DocumentoCompletude(
                documento_id=documento.documento_id,
                candidatos=[item[0] for item in resultados],
            ),
            [auditoria for item in resultados for auditoria in item[1]],
        )
        progresso.update()

    try:
        await asyncio.gather(*(processar(arquivo) for arquivo in arquivos))
    finally:
        progresso.close()


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
            config.input_dir,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("pipeline.yaml"))
    args = parser.parse_args()
    load_dotenv()
    config = PipelineConfig.from_yaml(args.config)
    entrada = config.workdir / "01-extraction"
    destino = config.workdir / "02-completeness"
    etapa = config.completeness
    escrever_manifesto_etapa(destino, "completeness", etapa)
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
            config.input_dir,
        )
    print(f"{destino}: completude concluída")


if __name__ == "__main__":
    main()
