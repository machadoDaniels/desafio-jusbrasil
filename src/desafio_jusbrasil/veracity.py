"""Etapa 3: verifica citações com um agente LangChain e a base SQLite."""

from __future__ import annotations

import asyncio
import json
import sqlite3
from pathlib import Path

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI
from tqdm import tqdm

from .contracts import (
    AuditoriaChamadaModelo,
    CandidatoAnalisado,
    CandidatoCitacao,
    CandidatoClassificado,
    Classificacao,
    ClassificadorVeracidade,
    ClassificadorVeracidadeAsync,
    ConsultaSQL,
    DocumentoClassificado,
    DocumentoCompletude,
    DocumentoPredito,
    ModelConfig,
    PipelineConfig,
    Predicao,
    Resolucao,
    ResultadoVeracidade,
    TipoCitacao,
    escrever_manifesto_etapa,
)

_SCHEMA_SQL = """Schema disponível:
CREATE TABLE documentos (
  documento_id TEXT PRIMARY KEY,
  id INTEGER NOT NULL UNIQUE,
  tribunal TEXT,
  ano INTEGER,
  relator TEXT,
  natureza TEXT NOT NULL, -- acordao, sumula ou dispositivo
  tipo TEXT NOT NULL,     -- jurisprudencia ou lei
  texto TEXT NOT NULL,
  texto_len INTEGER NOT NULL
);
CREATE VIRTUAL TABLE documentos_fts USING fts5(texto, content='documentos',
content_rowid='rowid');"""

_REGRAS_RESULTADO = """A consulta deve ser somente leitura e pode ser executada várias vezes. Em MATCH,
remova ou coloque entre aspas caracteres especiais como barras e hífens. Se a
ferramenta retornar erro_sql, corrija o SQL e tente novamente antes de responder.
Não classifique como inventada após uma única consulta vazia. Use documentos.id
como id_canonico. Classifique como real somente quando exatamente um registro
corresponder claramente ao trecho. Classifique como inventada somente após esgotar
buscas alternativas. Se os resultados permanecerem ambíguos, classifique como
incompleta. Nunca invente id_canonico. Baseie a justificativa nos resultados SQL
e retorne null em confianca."""

_PROMPT_JURISPRUDENCIA = f"""Você verifica uma citação de jurisprudência contra uma base SQLite fechada.
Receba o trecho original, escreva o SQL que considerar adequado e sempre execute
a ferramenta consultar_base antes de responder.

{_SCHEMA_SQL}

Priorize natureza='acordao' para processos e recursos e natureza='sumula' para
súmulas. Tente o identificador literal, suas partes distintivas e FTS5. Tolere
abreviações, espaços, pontos, hífens, barras e ruído de OCR. Teste números com e
sem pontuação e separadores. Use tribunal e ano apenas como filtros auxiliares.

{_REGRAS_RESULTADO}"""

_PROMPT_LEI = f"""Você verifica uma citação de legislação contra uma base SQLite fechada.
Receba o trecho original, escreva o SQL que considerar adequado e sempre execute
a ferramenta consultar_base antes de responder.

{_SCHEMA_SQL}

Priorize exclusivamente registros com natureza='dispositivo'. Procure o número do
dispositivo junto ao diploma legal e tente variações de nome, abreviação,
pontuação, espaços e ruído de OCR. Acórdãos que apenas mencionam o dispositivo não
são o registro canônico da lei.

{_REGRAS_RESULTADO}"""


def _prompt_veracidade(tipo: TipoCitacao) -> str:
    return _PROMPT_JURISPRUDENCIA if tipo == TipoCitacao.JURISPRUDENCIA else _PROMPT_LEI


def criar_modelo_veracidade(config: ModelConfig) -> BaseChatModel:
    return ChatOpenAI(
        model=config.model,
        base_url=config.base_url,
        temperature=config.temperature,
        top_p=config.top_p,
        reasoning_effort=config.reasoning_effort,
        extra_body={"top_k": config.top_k} if config.top_k is not None else None,
    )


class AgenteVeracidade:
    """Agente com uma única ferramenta de consulta à base canônica."""

    def __init__(self, modelo: BaseChatModel, database: Path) -> None:
        self._database = database
        ferramenta = StructuredTool.from_function(
            func=self._consultar_base,
            name="consultar_base",
            description=(
                "Executa uma consulta SQL somente leitura na base canônica e "
                "retorna no máximo 10 registros."
            ),
            args_schema=ConsultaSQL,
        )
        self._agents = {
            tipo: create_agent(
                model=modelo,
                tools=[ferramenta],
                system_prompt=_prompt_veracidade(tipo),
                response_format=ResultadoVeracidade,
            )
            for tipo in TipoCitacao
        }

    def classificar_auditada(
        self,
        candidato: CandidatoCitacao,
    ) -> tuple[ResultadoVeracidade, AuditoriaChamadaModelo]:
        entrada = self._entrada(candidato)
        resposta = self._agents[candidato.tipo].invoke(entrada)
        resultado = self._obter_resultado(resposta)
        return resultado, self._auditoria(
            entrada,
            resposta,
            resultado,
            _prompt_veracidade(candidato.tipo),
        )

    async def classificar_auditada_async(
        self,
        candidato: CandidatoCitacao,
    ) -> tuple[ResultadoVeracidade, AuditoriaChamadaModelo]:
        entrada = self._entrada(candidato)
        resposta = await self._agents[candidato.tipo].ainvoke(entrada)
        resultado = self._obter_resultado(resposta)
        return resultado, self._auditoria(
            entrada,
            resposta,
            resultado,
            _prompt_veracidade(candidato.tipo),
        )

    @staticmethod
    def _entrada(candidato: CandidatoCitacao) -> dict:
        return {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        f"Tipo definido pelo extractor: {candidato.tipo.value}\n"
                        f"Trecho original:\n{candidato.trecho}"
                    ),
                }
            ]
        }

    @staticmethod
    def _auditoria(
        entrada: dict,
        resposta: dict,
        resultado: ResultadoVeracidade,
        system_prompt: str,
    ) -> AuditoriaChamadaModelo:
        mensagens = []
        for mensagem in resposta.get("messages", []):
            if hasattr(mensagem, "model_dump"):
                mensagens.append(mensagem.model_dump(mode="json"))
            else:
                mensagens.append(mensagem)
        input_completo = {
            "system_prompt": system_prompt,
            "messages": entrada["messages"],
            "tools": [
                {
                    "name": "consultar_base",
                    "parameters": ConsultaSQL.model_json_schema(),
                }
            ],
            "response_format": ResultadoVeracidade.model_json_schema(),
        }
        return AuditoriaChamadaModelo(
            input=input_completo,
            output={
                "mensagens_agente": mensagens,
                "estruturada": resultado.model_dump(mode="json"),
            },
        )

    @staticmethod
    def _obter_resultado(resposta: dict) -> ResultadoVeracidade:
        if not any(
            getattr(mensagem, "type", None) == "tool"
            for mensagem in resposta.get("messages", [])
        ):
            raise RuntimeError("o agente de veracidade não consultou a base canônica")
        return ResultadoVeracidade.model_validate(resposta["structured_response"])

    def _consultar_base(self, sql: str) -> str:
        consulta = sql.strip().removesuffix(";")
        if ";" in consulta or not consulta.casefold().startswith(("select", "with")):
            return _erro_sql("a ferramenta aceita uma única consulta SELECT", consulta)

        try:
            uri = f"file:{self._database}?mode=ro"
            with sqlite3.connect(uri, uri=True) as conexao:
                conexao.row_factory = sqlite3.Row
                conexao.execute("PRAGMA query_only = ON")
                cursor = conexao.execute(consulta)
                colunas = [item[0] for item in cursor.description or []]
                linhas = cursor.fetchmany(10)
        except sqlite3.Error as erro:
            return _erro_sql(str(erro), consulta)

        dados = []
        for linha in linhas:
            item = dict(zip(colunas, linha, strict=True))
            if isinstance(item.get("texto"), str):
                item["texto"] = item["texto"][:3000]
            dados.append(item)
        return json.dumps(dados, ensure_ascii=False, default=str)


def _erro_sql(mensagem: str, sql: str) -> str:
    return json.dumps(
        {"erro_sql": mensagem, "sql": sql, "instrucao": "corrija e tente novamente"},
        ensure_ascii=False,
    )


def _listar_jsons(caminho: Path) -> list[Path]:
    arquivos = sorted(
        arquivo for arquivo in caminho.glob("*.json") if arquivo.name != "manifest.json"
    )
    if not arquivos:
        raise ValueError(f"nenhum arquivo JSON encontrado em {caminho}")
    return arquivos


def _ler_documento(arquivo: Path) -> DocumentoCompletude:
    try:
        return DocumentoCompletude.model_validate_json(
            arquivo.read_text(encoding="utf-8")
        )
    except ValueError as erro:
        raise ValueError(f"{arquivo}: {erro}") from erro


def _escrever_json(documento: DocumentoClassificado, destino: Path) -> None:
    destino.mkdir(parents=True, exist_ok=True)
    caminho = destino / f"{documento.documento_id}.json"
    temporario = caminho.with_suffix(".json.tmp")
    temporario.write_text(
        documento.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    temporario.replace(caminho)


def _resultado_incompleto() -> ResultadoVeracidade:
    return ResultadoVeracidade(
        classificacao=Classificacao.INCOMPLETA,
        justificativa="Citação sem identificador pesquisável.",
    )


def executar_veracidade(
    input_file: Path,
    output_file: Path,
    classificador: ClassificadorVeracidade,
) -> None:
    arquivos = _listar_jsons(input_file)
    with tqdm(
        total=len(arquivos), desc="Verificando citações", unit="documento"
    ) as progresso:
        for arquivo in arquivos:
            documento = _ler_documento(arquivo)
            candidatos = []
            chamadas = []
            for analisado in documento.candidatos:
                if analisado.completude.completa:
                    resultado, chamada = classificador.classificar_auditada(
                        analisado.candidato
                    )
                    chamadas.append(chamada)
                else:
                    resultado = _resultado_incompleto()
                candidatos.append(
                    CandidatoClassificado(
                        candidato=analisado.candidato,
                        completude=analisado.completude,
                        veracidade=resultado,
                    )
                )
            _escrever_json(
                DocumentoClassificado(
                    documento_id=documento.documento_id,
                    texto=documento.texto,
                    candidatos=candidatos,
                    chamadas_modelo=chamadas,
                ),
                output_file,
            )
            progresso.update()


async def executar_veracidade_async(
    input_file: Path,
    output_file: Path,
    classificador: ClassificadorVeracidadeAsync,
    max_concurrency: int,
) -> None:
    arquivos = _listar_jsons(input_file)
    semaforo = asyncio.Semaphore(max_concurrency)
    progresso = tqdm(total=len(arquivos), desc="Verificando citações", unit="documento")

    async def processar(arquivo: Path) -> None:
        documento = _ler_documento(arquivo)

        async def classificar(
            analisado: CandidatoAnalisado,
        ) -> tuple[CandidatoClassificado, AuditoriaChamadaModelo | None]:
            if analisado.completude.completa:
                async with semaforo:
                    (
                        resultado,
                        auditoria,
                    ) = await classificador.classificar_auditada_async(
                        analisado.candidato
                    )
            else:
                auditoria = None
                resultado = _resultado_incompleto()
            return (
                CandidatoClassificado(
                    candidato=analisado.candidato,
                    completude=analisado.completude,
                    veracidade=resultado,
                ),
                auditoria,
            )

        resultados = await asyncio.gather(
            *(classificar(analisado) for analisado in documento.candidatos)
        )
        _escrever_json(
            DocumentoClassificado(
                documento_id=documento.documento_id,
                texto=documento.texto,
                candidatos=[item[0] for item in resultados],
                chamadas_modelo=[item[1] for item in resultados if item[1] is not None],
            ),
            output_file,
        )
        progresso.update()

    try:
        await asyncio.gather(*(processar(arquivo) for arquivo in arquivos))
    finally:
        progresso.close()


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
            if item.candidato.inicio is None or item.candidato.fim is None:
                continue
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
    escrever_manifesto_etapa(destino, "veracity", etapa)
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
