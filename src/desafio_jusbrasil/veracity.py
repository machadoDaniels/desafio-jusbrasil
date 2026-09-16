"""Etapa 3: verifica citações com um agente LangChain e a base SQLite."""

from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path
from tempfile import NamedTemporaryFile

from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI

from .contracts import (
    CandidatoClassificado,
    Classificacao,
    ClassificadorVeracidade,
    ConsultaCanonica,
    DocumentoClassificado,
    DocumentoCompletude,
    PipelineConfig,
    RegistroCanonico,
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
        resposta = self._agent.invoke(
            {
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
        )
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


def _ler_jsonl(caminho: Path) -> list[DocumentoCompletude]:
    documentos = []
    with caminho.open(encoding="utf-8") as arquivo:
        for numero, linha in enumerate(arquivo, start=1):
            if linha.strip():
                try:
                    documentos.append(DocumentoCompletude.model_validate_json(linha))
                except ValueError as erro:
                    raise ValueError(f"{caminho}:{numero}: {erro}") from erro
    return documentos


def _escrever_jsonl(documentos: list[DocumentoClassificado], destino: Path) -> None:
    destino.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        "w", encoding="utf-8", dir=destino.parent, delete=False
    ) as temporario:
        caminho_temporario = Path(temporario.name)
        for documento in documentos:
            temporario.write(documento.model_dump_json(exclude_none=True) + "\n")
    caminho_temporario.replace(destino)


def executar_veracidade(
    input_file: Path,
    output_file: Path,
    classificador: ClassificadorVeracidade,
) -> None:
    saida = []
    for documento in _ler_jsonl(input_file):
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
        saida.append(
            DocumentoClassificado(
                documento_id=documento.documento_id,
                texto=documento.texto,
                candidatos=candidatos,
            )
        )
    _escrever_jsonl(saida, output_file)


def main() -> None:
    config = PipelineConfig.from_yaml(Path("pipeline.yaml"))
    entrada = config.workdir / "02-completeness.jsonl"
    destino = config.workdir / "03-veracity.jsonl"
    modelo = ChatOpenAI(
        model=config.model,
        temperature=0,
        base_url=config.base_url,
    )
    agente = AgenteVeracidade(modelo, config.database)
    executar_veracidade(entrada, destino, agente)
    print(f"{destino}: veracidade concluída")


if __name__ == "__main__":
    main()
