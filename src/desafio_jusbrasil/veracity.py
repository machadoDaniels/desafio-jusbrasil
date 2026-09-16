"""Etapa 3: verifica citações com um agente LangChain e a base SQLite."""

from __future__ import annotations

import asyncio
import json
import os
import re
import sqlite3
from pathlib import Path

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import StructuredTool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from tqdm import tqdm

from .contracts import (
    CandidatoAnalisado,
    CandidatoClassificado,
    Classificacao,
    ClassificadorVeracidade,
    ClassificadorVeracidadeAsync,
    ConsultaCanonica,
    DocumentoClassificado,
    DocumentoCompletude,
    DocumentoPredito,
    ModelConfig,
    PipelineConfig,
    Predicao,
    RegistroCanonico,
    Resolucao,
    ResultadoVeracidade,
    TipoCitacao,
)

_SYSTEM_PROMPT = """Você verifica citações jurídicas contra uma base canônica fechada.
Sempre chame a ferramenta buscar_base_canonica antes de responder.
Classifique como real somente quando exatamente um registro retornado corresponder
claramente à consulta. Use o id_canonico desse registro. Se nenhum registro
corresponder, classifique como inventada. Se faltarem dados ou houver ambiguidade
entre registros, classifique como incompleta. Nunca invente um id_canonico.
A justificativa deve ser curta e baseada no resultado da ferramenta.
Nesta primeira versão, retorne null em confianca."""


def criar_modelo_veracidade(config: ModelConfig) -> BaseChatModel:
    if config.provider == "gemini":
        return ChatGoogleGenerativeAI(
            model=config.model,
            api_key=os.environ["OPENAI_API_KEY"],
            temperature=config.temperature,
            thinking_level=config.reasoning_effort,
        )
    return ChatOpenAI(
        model=config.model,
        temperature=config.temperature,
        reasoning_effort=config.reasoning_effort,
        base_url=config.base_url,
    )


class AgenteVeracidade:
    """Agente com uma única ferramenta de consulta à base canônica."""

    def __init__(self, modelo: BaseChatModel, database: Path) -> None:
        self._database = database
        ferramenta = StructuredTool.from_function(
            func=self._buscar_para_agente,
            name="buscar_base_canonica",
            description=(
                "Busca acórdãos, súmulas ou dispositivos na base canônica. "
                "Use os campos extraídos da citação e não invente valores."
            ),
            args_schema=ConsultaCanonica,
        )
        self._agent = create_agent(
            model=modelo,
            tools=[ferramenta],
            system_prompt=_SYSTEM_PROMPT,
            response_format=ResultadoVeracidade,
        )

    def classificar(self, consulta: ConsultaCanonica) -> ResultadoVeracidade:
        resposta = self._agent.invoke(self._entrada(consulta))
        return self._obter_resultado(resposta)

    async def classificar_async(
        self,
        consulta: ConsultaCanonica,
    ) -> ResultadoVeracidade:
        resposta = await self._agent.ainvoke(self._entrada(consulta))
        return self._obter_resultado(resposta)

    @staticmethod
    def _entrada(consulta: ConsultaCanonica) -> dict:
        return {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Verifique esta consulta canônica:\n"
                        + consulta.model_dump_json(exclude_none=True)
                    ),
                }
            ]
        }

    @staticmethod
    def _obter_resultado(resposta: dict) -> ResultadoVeracidade:
        if not any(
            getattr(mensagem, "type", None) == "tool"
            for mensagem in resposta.get("messages", [])
        ):
            raise RuntimeError("o agente de veracidade não consultou a base canônica")
        return ResultadoVeracidade.model_validate(resposta["structured_response"])

    def buscar(self, consulta: ConsultaCanonica) -> list[RegistroCanonico]:
        termos = self._montar_consulta_fts(consulta)
        if not termos:
            return []
        return self._executar_consulta(termos, consulta.tipo)

    def _buscar_para_agente(
        self,
        tipo: TipoCitacao,
        classe_processual: str | None = None,
        numero: str | None = None,
        tribunal: str | None = None,
        uf: str | None = None,
        ano: int | None = None,
        relator: str | None = None,
        dispositivo: str | None = None,
    ) -> str:
        consulta = ConsultaCanonica(
            tipo=tipo,
            classe_processual=classe_processual,
            numero=numero,
            tribunal=tribunal,
            uf=uf,
            ano=ano,
            relator=relator,
            dispositivo=dispositivo,
        )
        registros = self.buscar(consulta)
        dados = []
        for registro in registros:
            item = registro.model_dump(mode="json")
            item["texto"] = registro.texto[:3000]
            dados.append(item)
        return json.dumps(dados, ensure_ascii=False)

    def _montar_consulta_fts(self, consulta: ConsultaCanonica) -> str:
        valores = (
            consulta.classe_processual,
            consulta.numero,
            consulta.tribunal,
            consulta.uf,
            str(consulta.ano) if consulta.ano else None,
            consulta.relator,
            consulta.dispositivo,
        )
        tokens = []
        for valor in valores:
            if not valor:
                continue
            for token in re.findall(r"\w+", valor, flags=re.UNICODE):
                if len(token) > 1 and token.casefold() not in {
                    t.casefold() for t in tokens
                }:
                    tokens.append(token)
        return " OR ".join(f'"{token}"' for token in tokens)

    def _executar_consulta(
        self,
        termos: str,
        tipo: TipoCitacao,
    ) -> list[RegistroCanonico]:
        sql = """
            SELECT d.documento_id, d.id, d.tribunal, d.ano, d.relator,
                   d.natureza, d.tipo, d.texto
              FROM documentos_fts AS f
              JOIN documentos AS d ON d.rowid = f.rowid
             WHERE documentos_fts MATCH ? AND d.tipo = ?
             ORDER BY bm25(documentos_fts)
             LIMIT 10
        """
        with sqlite3.connect(self._database) as conexao:
            linhas = conexao.execute(sql, (termos, tipo.value)).fetchall()
        return [
            RegistroCanonico(
                documento_id=linha[0],
                id_canonico=linha[1],
                tribunal=linha[2],
                ano=linha[3],
                relator=linha[4],
                natureza=linha[5],
                tipo=linha[6],
                texto=linha[7],
            )
            for linha in linhas
        ]


def _ler_jsons(caminho: Path) -> list[DocumentoCompletude]:
    arquivos = sorted(caminho.glob("*.json"))
    if not arquivos:
        raise ValueError(f"nenhum arquivo JSON encontrado em {caminho}")
    documentos = []
    for arquivo in arquivos:
        try:
            documentos.append(
                DocumentoCompletude.model_validate_json(
                    arquivo.read_text(encoding="utf-8")
                )
            )
        except ValueError as erro:
            raise ValueError(f"{arquivo}: {erro}") from erro
    return documentos


def _escrever_jsons(documentos: list[DocumentoClassificado], destino: Path) -> None:
    destino.mkdir(parents=True, exist_ok=True)
    for documento in documentos:
        caminho = destino / f"{documento.documento_id}.json"
        caminho.write_text(
            documento.model_dump_json(indent=2, exclude_none=True) + "\n",
            encoding="utf-8",
        )


def executar_veracidade(
    input_file: Path,
    output_file: Path,
    classificador: ClassificadorVeracidade,
) -> None:
    documentos = _ler_jsons(input_file)
    total = sum(len(documento.candidatos) for documento in documentos)
    saida = []
    with tqdm(total=total, desc="Verificando citações") as progresso:
        for documento in documentos:
            candidatos = []
            for analisado in documento.candidatos:
                if analisado.completude.completa:
                    assert analisado.completude.consulta is not None
                    resultado = classificador.classificar(analisado.completude.consulta)
                else:
                    resultado = ResultadoVeracidade(
                        classificacao=Classificacao.INCOMPLETA,
                        justificativa=analisado.completude.justificativa,
                    )
                candidatos.append(
                    CandidatoClassificado(
                        candidato=analisado.candidato,
                        completude=analisado.completude,
                        veracidade=resultado,
                    )
                )
                progresso.update()
            saida.append(
                DocumentoClassificado(
                    documento_id=documento.documento_id,
                    texto=documento.texto,
                    candidatos=candidatos,
                )
            )
    _escrever_jsons(saida, output_file)


async def executar_veracidade_async(
    input_file: Path,
    output_file: Path,
    classificador: ClassificadorVeracidadeAsync,
    max_concurrency: int,
) -> None:
    documentos = _ler_jsons(input_file)
    total = sum(len(documento.candidatos) for documento in documentos)
    semaforo = asyncio.Semaphore(max_concurrency)
    progresso = tqdm(total=total, desc="Verificando citações")

    async def processar(documento: DocumentoCompletude) -> DocumentoClassificado:
        async def classificar(
            analisado: CandidatoAnalisado,
        ) -> CandidatoClassificado:
            if analisado.completude.completa:
                assert analisado.completude.consulta is not None
                async with semaforo:
                    resultado = await classificador.classificar_async(
                        analisado.completude.consulta
                    )
            else:
                resultado = ResultadoVeracidade(
                    classificacao=Classificacao.INCOMPLETA,
                    justificativa=analisado.completude.justificativa,
                )
            progresso.update()
            return CandidatoClassificado(
                candidato=analisado.candidato,
                completude=analisado.completude,
                veracidade=resultado,
            )

        candidatos = await asyncio.gather(
            *(classificar(analisado) for analisado in documento.candidatos)
        )
        return DocumentoClassificado(
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


def materializar(entrada: Path, pasta_saida: Path) -> None:
    arquivos = sorted(entrada.glob("*.json"))
    if not arquivos:
        raise ValueError(f"nenhum arquivo JSON encontrado em {entrada}")
    pasta_saida.mkdir(parents=True, exist_ok=True)
    for arquivo in tqdm(arquivos, desc="Materializando predições"):
        documento = DocumentoClassificado.model_validate_json(
            arquivo.read_text(encoding="utf-8")
        )
        citacoes = []
        for item in documento.candidatos:
            resultado = item.veracidade
            resolucao = None
            if resultado.classificacao == Classificacao.REAL:
                assert resultado.id_canonico is not None
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
        predicao = DocumentoPredito(
            documento_id=documento.documento_id,
            citacoes=citacoes,
        )
        (pasta_saida / arquivo.name).write_text(
            predicao.model_dump_json(indent=2, exclude_none=True) + "\n",
            encoding="utf-8",
        )


def main() -> None:
    load_dotenv()
    config = PipelineConfig.from_yaml(Path("pipeline.yaml"))
    entrada = config.workdir / "02-completeness"
    destino = config.workdir / "03-veracity"
    etapa = config.veracity
    agente = AgenteVeracidade(
        criar_modelo_veracidade(etapa),
        config.database,
    )
    if etapa.async_requests:
        asyncio.run(
            executar_veracidade_async(
                entrada,
                destino,
                agente,
                etapa.max_concurrency,
            )
        )
    else:
        executar_veracidade(entrada, destino, agente)
    print(f"{destino}: veracidade concluída")

    materializar(destino, config.workdir / "predictions")


if __name__ == "__main__":
    main()
