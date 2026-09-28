"""Etapa 3: extrai entidades usadas para consultar a base canônica."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import AsyncOpenAI, OpenAIError, omit
from openai.types.chat import ChatCompletionMessageParam
from pydantic import ValidationError
from tqdm import tqdm

from .contracts import (
    CandidatoAnalisado,
    CandidatoCitacao,
    CandidatoEntidades,
    ConsultaJurisprudencia,
    ConsultaJurisprudenciaAgente,
    ConsultaLegislacao,
    ConsultaLegislacaoAgente,
    Contract,
    DocumentoCompletude,
    DocumentoEntidades,
    ExtratorEntidadesAsync,
    ModelConfig,
    PipelineConfig,
    TipoCitacao,
)
from .utils import (
    criar_auditoria,
    criar_auditoria_erro,
    escrever_manifesto_etapa,
    escrever_saida_documento,
    ler_documento,
    listar_resultados,
    normalizar_numero_cnj,
    normalizar_relator,
)

_PROMPT_JURISPRUDENCIA = """CONTEXTO
Você é a terceira etapa de um pipeline de verificação de citações jurídicas. As etapas
anteriores já localizaram o trecho no documento e decidiram que ele é uma citação de
jurisprudência. Sua tarefa é transformar o trecho em campos estruturados. A etapa seguinte
não usa modelo: ela monta uma consulta SQL exata com os campos que você devolver e compara
com uma base de acórdãos e súmulas. Se um número vier errado, quebrado ou em campo errado,
a consulta não encontra o registro e uma citação verdadeira é marcada como inventada.

TAREFA
Extraia os campos de identificação desta citação de jurisprudência brasileira.
Preencha somente dados explícitos no trecho ou decorrentes de siglas jurídicas inequívocas.
Campos ausentes ficam nulos. Não invente tribunal, UF ou ano. Não avalie se a citação
existe: um número inexistente deve ser extraído do mesmo jeito.

CAMPOS
- natureza: "sumula" quando o trecho menciona Súmula; "acordao" para processo ou recurso.
- numero_processo_cnj: o número único nacional, 20 dígitos em 6 blocos: 7 + 2 + 4 + 1 + 2 + 4.
  Pode vir com pontos ("7001184-15.2019.7.00.0000") ou sem ("7001184-1520197000000"):
  os dois são o mesmo número. Copie TODOS os dígitos na ordem em que aparecem, sem
  reagrupar, sem inserir nem remover nenhum; se o primeiro bloco tiver menos de 7 dígitos,
  complete com zeros à esquerda. Sempre que o trecho tiver esse formato, preencha ESTE campo
  e deixe numero_classe_tribunal e numero_registro_tribunal nulos.
    "0600216-46.2020.6.14.0022"      → "06002164620206140022"
    "7001184-1520197000000"          → "70011841520197000000"
    "ARR-471-22.2011.5.03.0044"      → "00004712220115030044"
    "TST-RR-79500-16.2009.5.15.0016" → "00795001620095150016"
- numero_classe_tribunal: o número sequencial que acompanha a sigla da classe nas numerações
  próprias de STF e STJ, somente dígitos. "REsp 1.741.784" → "1741784"; "Rcl 68.244" → "68244".
- numero_registro_tribunal: o registro interno no formato AAAA/NNNNNNN-D, somente dígitos.
  "2018/0116304-1" → "201801163041". Raro; só preencha se o trecho trouxer esse formato.
- classe_processual: a classe do processo-base, isto é, a última da cadeia. Em
  "AgInt no REsp", é REsp; em "EDcl no AgRg no AREsp", é AREsp. Use os valores da lista.
- cadeia_recursal: todas as classes citadas, sem ordem e sem repetição.
- tribunal: apenas se a sigla ou o nome do tribunal estiver escrito no trecho.
- uf: a sigla de duas letras do estado que acompanha o número. O separador varia e deve
  ser ignorado: "/PR", "- PR", "–CE", "(MA)" resultam em "PR", "CE", "MA". Sem sigla de
  estado no trecho, deixe nulo. Atenção: em "TST-RR-...", RR é a classe Recurso de
  Revista, não Roraima.
- ano: só o ano do julgamento quando escrito ("de 2021", "julgado em 2020"). Não use o ano
  que está dentro do CNJ.
- relator: o nome como aparece, sem "Rel.", "Min." ou "Ministro".
- numero_sumula e sumula_vinculante: preencha para súmulas; sumula_vinculante é true
  apenas para "Súmula Vinculante" do STF.
- Em súmulas, numero_processo_cnj, numero_classe_tribunal e numero_registro_tribunal ficam nulos.

NÚMEROS COM RUÍDO
O trecho pode vir de OCR, com quebras de linha e pontuação irregular no meio do número.
Junte os pedaços de um mesmo número separados por quebra de linha, hífen duplo ou ponto solto,
sem inventar dígitos:
    "0600216-46.2020-\n.6.14.0022"  → CNJ "06002164620206140022"
    "TST-AgRR-25823-78.2015.5.24.\n0091" → CNJ "00258237820155240091"
    "n. 2785 (SP)"                  → numero_classe_tribunal "2785", uf "SP"
Letras no lugar de dígitos são erro de OCR, converta: g→9, l e I→1, O→0, S→5, B→8, Z→2.
    "1.45g.779"  → "1459779"        "l.741.784" → "1741784"

SÚMULAS
    "Súmula Vinculante 10"  → natureza "sumula", numero_sumula 10, sumula_vinculante true, tribunal "STF"
    "Súmula 331 do TST"     → natureza "sumula", numero_sumula 331, sumula_vinculante false, tribunal "TST"
    "5úmula 211 do STJ"     → natureza "sumula", numero_sumula 211, sumula_vinculante false, tribunal "STJ"

REFERÊNCIAS SEM NÚMERO
Citações como "acórdão do STJ julgado em 2021 sob relatoria de Assusete Magalhães" ou
"Rcl de 2021, Rel. Min. Rosa Weber" não têm número: preencha tribunal, ano, relator e classe
quando escritos, e deixe todos os campos numéricos nulos.

Exemplo completo — a classe principal é o recurso-base, não o recurso incidental mais externo:

Trecho: EDcl no AgInt no Agravo em Recurso Especial nº 1904603/TO

Resposta:
{
  "natureza": "acordao",
  "numero_processo_cnj": null,
  "numero_classe_tribunal": "1904603",
  "numero_registro_tribunal": null,
  "classe_processual": "AREsp — Agravo em Recurso Especial",
  "cadeia_recursal": [
    "EDcl — Embargos de Declaração",
    "AgInt — Agravo Interno",
    "AREsp — Agravo em Recurso Especial"
  ],
  "tribunal": null,
  "uf": "TO",
  "ano": null,
  "relator": null,
  "numero_sumula": null,
  "sumula_vinculante": null
}"""

_LOG = logging.getLogger(__name__)


class ErroExtracaoEntidades(RuntimeError):
    """Falha final de uma citação com as tentativas auditadas."""

    def __init__(self, mensagem: str, auditorias: list[dict[str, Any]]) -> None:
        super().__init__(mensagem)
        self.auditorias = auditorias


_PROMPT_LEI = """CONTEXTO
Você é a terceira etapa de um pipeline de verificação de citações jurídicas. As etapas
anteriores já localizaram o trecho no documento e decidiram que ele é uma citação de
legislação. Sua tarefa é transformar o trecho em campos estruturados. A etapa seguinte
não usa modelo: ela monta uma consulta SQL exata com os campos que você devolver e
compara com uma base de dispositivos de lei. Se um campo vier errado, a consulta não
encontra o registro e uma citação verdadeira é marcada como inventada.

TAREFA
Extraia os campos de identificação desta citação de legislação brasileira.
Preencha somente dados explícitos no trecho ou decorrentes de siglas e nomes de
diplomas inequívocos. Campos ausentes ficam nulos. Não corrija nem avalie a citação:
um artigo inexistente deve ser extraído do mesmo jeito.

CAMPOS
- numero_artigo: só o número do artigo, sem "art.", sem ordinal e sem inciso, parágrafo
  ou alínea. "art. 93, IX" → "93". "artigo 5º, LV" → "5". "§ 2º do art. 14" → "14".
- diploma: o nome canônico da lista permitida. Reconheça as variantes:
  "Constituição da República", "Constituição Federal", "CF/88", "CRFB", "Carta Magna"
  → CF — Constituição Federal.
  "Código de Processo Civil", "CPC", "Lei nº 13.105/2015" → CPC — Código de Processo Civil.
  "Consolidação das Leis do Trabalho", "CLT", "Decreto-Lei nº 5.452/1943" → CLT.
  "Código Eleitoral", "Lei nº 4.737/1965" → CE — Código Eleitoral.
  "Lei Complementar nº 64/1990" → LC — Lei Complementar.
  Use "Lei" apenas quando o trecho traz um número de lei que não corresponde a nenhum
  dos códigos da lista.
- numero_diploma: o número da norma, somente dígitos, NUNCA o ano.
  "Lei nº 13.105/2015" → "13105". "Lei Complementar nº 64/1990" → "64".
  Quando o trecho cita um código pelo nome, informe o número da lei que o instituiu:
  CPC → "13105"; CC → "10406"; CDC → "8078"; CPP → "3689"; CPM → "1001";
  CE → "4737"; CLT → "5452".
  A Constituição Federal não tem número: numero_diploma nulo. Nunca escreva "1988".

Exemplo completo — não use `Lei` como fallback quando o trecho identifica um diploma específico:

Trecho: art. 1.134 da Lei nº 13.105/2015

Resposta:
{
  "numero_artigo": "1134",
  "diploma": "CPC — Código de Processo Civil",
  "numero_diploma": "13105"
}"""


def _prompt_veracidade(tipo: TipoCitacao) -> str:
    return _PROMPT_JURISPRUDENCIA if tipo == TipoCitacao.JURISPRUDENCIA else _PROMPT_LEI


def _contrato_consulta(tipo: TipoCitacao) -> type[Contract]:
    return (
        ConsultaJurisprudenciaAgente
        if tipo == TipoCitacao.JURISPRUDENCIA
        else ConsultaLegislacaoAgente
    )


class AgenteExtratorEntidades:
    """Extrai do trecho os campos usados na consulta canônica."""

    def __init__(
        self,
        cliente: AsyncOpenAI,
        config: ModelConfig,
        relatores: Mapping[str, str] | None = None,
    ) -> None:
        self._cliente = cliente
        self._config = config
        self._relatores = relatores or {}

    async def extrair_auditada_async(
        self,
        candidato: CandidatoCitacao,
    ) -> tuple[ConsultaJurisprudencia | ConsultaLegislacao, list[dict[str, Any]]]:
        if not isinstance(self._cliente, AsyncOpenAI):
            raise TypeError("extrair_auditada_async() exige cliente assíncrono")
        requisicao = self._requisicao(candidato)
        auditorias: list[dict[str, Any]] = []
        ultimo_erro: Exception | None = None
        for tentativa in range(1, self._config.max_retries + 1):
            resposta = None
            try:
                async with asyncio.timeout(self._config.request_timeout_seconds):
                    resposta = await self._cliente.chat.completions.parse(**requisicao)
                consulta, auditoria = self._finalizar(
                    requisicao, resposta, candidato, tentativa
                )
                return consulta, [*auditorias, auditoria]
            except (
                TimeoutError,
                OpenAIError,
                ValidationError,
                ValueError,
                IndexError,
                AttributeError,
            ) as erro:
                ultimo_erro = erro
                auditorias.append(
                    criar_auditoria_erro(
                        requisicao,
                        _contrato_consulta(candidato.tipo),
                        erro,
                        tentativa,
                        resposta,
                    )
                )
                _LOG.warning(
                    "entidades: tentativa %d/%d falhou para %r: %s",
                    tentativa,
                    self._config.max_retries,
                    candidato.trecho[:80],
                    erro,
                )
                if tentativa < self._config.max_retries:
                    await asyncio.sleep(tentativa)
        raise ErroExtracaoEntidades(
            f"retries esgotados para {candidato.trecho[:80]!r}: {ultimo_erro}",
            auditorias,
        ) from ultimo_erro

    @staticmethod
    def _mensagens(candidato: CandidatoCitacao) -> list[ChatCompletionMessageParam]:
        return [
            {"role": "system", "content": _prompt_veracidade(candidato.tipo)},
            {"role": "user", "content": f"Trecho original:\n{candidato.trecho}"},
        ]

    def _requisicao(self, candidato: CandidatoCitacao) -> dict[str, Any]:
        requisicao: dict[str, Any] = {
            "model": self._config.model,
            "temperature": self._config.temperature
            if self._config.temperature is not None
            else omit,
            "top_p": self._config.top_p if self._config.top_p is not None else omit,
            "messages": self._mensagens(candidato),
            "response_format": _contrato_consulta(candidato.tipo),
            "reasoning_effort": self._config.reasoning_effort
            if self._config.reasoning_effort is not None
            else omit,
        }
        if self._config.top_k is not None:
            requisicao["extra_body"] = {"top_k": self._config.top_k}
        return requisicao

    def _finalizar(
        self,
        requisicao: dict[str, Any],
        resposta: Any,
        candidato: CandidatoCitacao,
        tentativa: int,
    ) -> tuple[ConsultaJurisprudencia | ConsultaLegislacao, dict[str, Any]]:
        consulta = resposta.choices[0].message.parsed
        if consulta is None:
            raise RuntimeError("o modelo não retornou entidades estruturadas")
        if candidato.tipo == TipoCitacao.JURISPRUDENCIA:
            numero_cnj = normalizar_numero_cnj(
                consulta.numero_processo_cnj, candidato.trecho
            )
            numero_sumula = consulta.numero_sumula
            natureza = (
                "sumula"
                if numero_sumula is not None
                else consulta.natureza or "acordao"
            )
            vinculante = consulta.sumula_vinculante
            if natureza == "sumula" and vinculante is None:
                vinculante = bool(
                    re.search(
                        r"\bs[uú]mula\s+vinculante\b", candidato.trecho, re.IGNORECASE
                    )
                )
            dados = consulta.model_dump()
            dados.update(
                natureza=natureza,
                numero_processo_cnj=numero_cnj,
                relator_norm=normalizar_relator(consulta.relator, self._relatores),
                sumula_vinculante=vinculante,
            )
            consulta = ConsultaJurisprudencia.model_validate(dados)
        else:
            consulta = ConsultaLegislacao.model_validate(consulta.model_dump())
        auditoria = criar_auditoria(
            requisicao,
            resposta,
            _contrato_consulta(candidato.tipo),
            tentativa=tentativa,
            campos_extraidos=consulta.model_dump(mode="json"),
        )
        return consulta, auditoria


async def _extrair_entidade(
    analisado: CandidatoAnalisado,
    extrator: ExtratorEntidadesAsync,
    semaforo: asyncio.Semaphore,
) -> tuple[CandidatoEntidades, list[dict[str, Any]]]:
    async with semaforo:
        campos, chamadas = await extrator.extrair_auditada_async(analisado.candidato)
    return CandidatoEntidades(
        candidato=analisado.candidato,
        completude=analisado.completude,
        campos_extraidos=campos,
    ), chamadas


async def _processar_documento_entities(
    arquivo: Path,
    output_file: Path,
    extrator: ExtratorEntidadesAsync,
    semaforo: asyncio.Semaphore,
    progresso: Any,
) -> None:
    documento = ler_documento(arquivo, DocumentoCompletude)
    candidatos: list[CandidatoEntidades] = []
    chamadas: list[dict[str, Any]] = []
    if not documento.candidatos:
        escrever_saida_documento(
            output_file,
            DocumentoEntidades(documento_id=documento.documento_id, candidatos=[]),
            [],
        )
    for analisado in documento.candidatos:
        try:
            candidato, chamadas_citacao = await _extrair_entidade(
                analisado, extrator, semaforo
            )
        except ErroExtracaoEntidades as erro:
            chamadas.extend(erro.auditorias)
            escrever_saida_documento(
                output_file,
                DocumentoEntidades(
                    documento_id=documento.documento_id,
                    candidatos=candidatos,
                ),
                chamadas,
            )
            raise
        candidatos.append(candidato)
        chamadas.extend(chamadas_citacao)
        escrever_saida_documento(
            output_file,
            DocumentoEntidades(
                documento_id=documento.documento_id,
                candidatos=candidatos,
            ),
            chamadas,
        )
        progresso.update()


async def executar_entities_async(
    input_file: Path,
    output_file: Path,
    extrator: ExtratorEntidadesAsync,
    max_concurrency: int,
) -> None:
    arquivos = listar_resultados(input_file)
    total_citacoes = sum(
        len(ler_documento(arquivo, DocumentoCompletude).candidatos)
        for arquivo in arquivos
    )
    semaforo = asyncio.Semaphore(max_concurrency)
    progresso = tqdm(
        total=total_citacoes,
        desc="Extraindo entidades",
        unit="citação",
    )
    try:
        await asyncio.gather(
            *(
                _processar_documento_entities(
                    arquivo, output_file, extrator, semaforo, progresso
                )
                for arquivo in arquivos
            )
        )
    finally:
        progresso.close()


async def _executar_entities_async(config: PipelineConfig, destino: Path) -> None:
    etapa = config.entities
    caminho_relatores = config.database.parent / "relatores_padronizacao.json"
    relatores = json.loads(caminho_relatores.read_text(encoding="utf-8"))
    async with AsyncOpenAI(base_url=etapa.base_url) as cliente:
        await executar_entities_async(
            config.workdir / "02-completeness",
            destino,
            AgenteExtratorEntidades(cliente, etapa, relatores),
            etapa.max_concurrency,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("pipeline.yaml"))
    args = parser.parse_args()
    load_dotenv()
    config = PipelineConfig.from_yaml(args.config)
    destino = config.workdir / "03-entities"
    etapa = config.entities
    escrever_manifesto_etapa(destino, "entities", etapa)
    asyncio.run(_executar_entities_async(config, destino))
    print(f"{destino}: entidades concluídas")


if __name__ == "__main__":
    main()
