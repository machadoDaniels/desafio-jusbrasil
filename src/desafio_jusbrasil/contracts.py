"""Contratos compartilhados entre as etapas do pipeline."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Any, Literal, Protocol

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class Contract(BaseModel):
    """Base estrita para dados persistidos e respostas de agentes."""

    model_config = ConfigDict(extra="forbid")


class TipoCitacao(StrEnum):
    JURISPRUDENCIA = "jurisprudencia"
    LEI = "lei"


class Classificacao(StrEnum):
    REAL = "real"
    INVENTADA = "inventada"
    INCOMPLETA = "incompleta"


ReasoningEffort = Literal["none", "minimal", "low", "medium", "high", "xhigh", "max"]
NaturezaJurisprudencia = Literal["acordao", "sumula"]
Tribunal = Literal["STF", "STJ", "STM", "TSE", "TST"]
UF = Literal[
    "AC",
    "AL",
    "AP",
    "AM",
    "BA",
    "CE",
    "DF",
    "ES",
    "GO",
    "MA",
    "MT",
    "MS",
    "MG",
    "PA",
    "PB",
    "PR",
    "PE",
    "PI",
    "RJ",
    "RN",
    "RS",
    "RO",
    "RR",
    "SC",
    "SP",
    "SE",
    "TO",
]
ClasseProcessual = Literal[
    "AI",
    "APL",
    "AR",
    "AREsp",
    "AREspe",
    "ARR",
    "AgInt",
    "AgRg",
    "E",
    "EDcl",
    "HC",
    "RE",
    "REsp",
    "REspe",
    "RHC",
    "RMS",
    "RR",
    "RSE",
    "Rcl",
    "Rp",
    "SLS",
]
Diploma = Literal[
    "Constituição Federal",
    "Código Civil",
    "Código de Defesa do Consumidor",
    "Código de Processo Civil",
    "Código de Processo Penal",
    "Código Penal Militar",
    "Código Eleitoral",
    "Consolidação das Leis do Trabalho",
    "Lei Complementar",
    "Lei",
]


class ModelConfig(Contract):
    model: str
    base_url: str | None = None
    temperature: float | None = Field(default=None, ge=0, le=2)
    top_p: float | None = Field(default=None, ge=0, le=1)
    top_k: int | None = Field(default=None, ge=1)
    reasoning_effort: ReasoningEffort | None = None


class StageConfig(ModelConfig):
    async_requests: bool = False
    max_concurrency: int = Field(default=4, ge=1)
    max_retries: int = Field(default=3, ge=1)
    request_timeout_seconds: float = Field(default=120, gt=0)
    debug: bool = False


class PipelineConfig(Contract):
    input_dir: Path
    workdir: Path
    database: Path
    database_preprocessing: dict[str, Any] | None = None
    extractor: StageConfig
    completeness: StageConfig
    entities: StageConfig

    @classmethod
    def from_yaml(cls, caminho: Path) -> PipelineConfig:
        dados = yaml.safe_load(caminho.read_text(encoding="utf-8"))
        return cls.model_validate(dados)


class CandidatoCitacaoRequest(Contract):
    trecho: str = Field(min_length=1)
    tipo: TipoCitacao
    confianca_extracao: float | None = Field(default=None, ge=0, le=1)


class LoteCandidatosRequest(Contract):
    candidatos: list[CandidatoCitacaoRequest]


class CandidatoCitacao(CandidatoCitacaoRequest):
    inicio: int | None = Field(default=None, ge=0)
    fim: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def validar_intervalo(self) -> CandidatoCitacao:
        if (self.inicio is None) != (self.fim is None):
            raise ValueError("inicio e fim devem ser ambos preenchidos ou ambos nulos")
        if self.inicio is not None and self.fim is not None and self.fim <= self.inicio:
            raise ValueError("fim deve ser maior que inicio")
        return self


class DocumentoExtraido(Contract):
    documento_id: str = Field(min_length=1)
    candidatos: list[CandidatoCitacao]
    texto: str = Field(default="", exclude=True)
    chamadas_modelo: list[dict[str, Any]] = Field(default_factory=list, exclude=True)

    @model_validator(mode="after")
    def validar_spans(self) -> DocumentoExtraido:
        if not self.texto:
            return self
        for candidato in self.candidatos:
            if candidato.inicio is None or candidato.fim is None:
                continue
            if candidato.fim > len(self.texto):
                raise ValueError("span fora dos limites do documento")
            if self.texto[candidato.inicio : candidato.fim] != candidato.trecho:
                raise ValueError("trecho não corresponde ao span do documento")
        return self


class ResultadoCompletude(Contract):
    completa: bool


class CandidatoAnalisado(Contract):
    candidato: CandidatoCitacao
    completude: ResultadoCompletude


class DocumentoCompletude(Contract):
    documento_id: str = Field(min_length=1)
    candidatos: list[CandidatoAnalisado]
    texto: str = Field(default="", exclude=True)
    chamadas_modelo: list[dict[str, Any]] = Field(default_factory=list, exclude=True)


class ConsultaJurisprudenciaAgente(Contract):
    """Entidades de jurisprudência solicitadas ao modelo."""

    natureza: NaturezaJurisprudencia | None = Field(
        default=None,
        description="'sumula' de súmula; 'acordao' para processo ou recurso.",
    )
    numero_processo_cnj: str | None = None
    numero_classe_tribunal: str | None = None
    numero_registro_tribunal: str | None = None
    classe_processual: ClasseProcessual | None = Field(
        default=None,
        description="Classe do processo ou recurso principal, independentemente da cadeia.",
    )
    cadeia_recursal: list[ClasseProcessual] | None = Field(
        default=None,
        description="Todas as classes citadas, sem ordem nem repetições.",
    )

    @field_validator("cadeia_recursal")
    @classmethod
    def remover_repeticoes_da_cadeia(
        cls, valor: list[ClasseProcessual] | None
    ) -> list[ClasseProcessual] | None:
        return list(dict.fromkeys(valor)) if valor else valor

    tribunal: Tribunal | None = Field(
        default=None, description="Sigla canônica do tribunal explicitamente citado."
    )
    uf: UF | None = Field(
        default=None,
        description="UF do processo; não confunda a classe RR com o estado de Roraima.",
    )
    ano: int | None = Field(
        default=None,
        ge=1,
        le=9999,
        description="Ano do julgamento, não o ano contido no número CNJ.",
    )
    relator: str | None = Field(
        default=None, description="Nome do relator como aparece na citação."
    )
    numero_sumula: int | None = Field(
        default=None, ge=1, description="Número inteiro da súmula, sem prefixo."
    )
    sumula_vinculante: bool | None = Field(
        default=None,
        description=(
            "true apenas para Súmula Vinculante do STF; false para súmula comum."
        ),
    )

    @field_validator(
        "numero_processo_cnj", "numero_classe_tribunal", "numero_registro_tribunal"
    )
    @classmethod
    def validar_numeros(cls, valor: str | None, info: Any) -> str | None:
        if valor is None:
            return None
        if not valor.isascii() or not valor.isdigit():
            raise ValueError(f"{info.field_name} deve conter somente dígitos ASCII")
        return valor


class ConsultaJurisprudencia(ConsultaJurisprudenciaAgente):
    """Consulta de jurisprudência normalizada para busca na base canônica."""

    relator_norm: str | None = None

    @field_validator("numero_processo_cnj")
    @classmethod
    def validar_cnj_normalizado(cls, valor: str | None) -> str | None:
        if valor is not None and len(valor) != 20:
            raise ValueError("numero_processo_cnj deve conter 20 dígitos")
        return valor


class ConsultaLegislacao(Contract):
    numero_artigo: str | None = Field(
        default=None, description="Número do artigo sem o prefixo 'Art.'."
    )
    diploma: Diploma | None = Field(
        default=None, description="Nome canônico do diploma entre as opções permitidas."
    )
    numero_diploma: str | None = Field(
        default=None, description="Número do diploma somente com dígitos."
    )
    ano_diploma: int | None = Field(
        default=None, ge=1, le=9999, description="Ano do diploma com quatro dígitos."
    )


class CandidatoEntidades(Contract):
    candidato: CandidatoCitacao
    completude: ResultadoCompletude
    campos_extraidos: ConsultaJurisprudencia | ConsultaLegislacao | None = None


class DocumentoEntidades(Contract):
    documento_id: str = Field(min_length=1)
    candidatos: list[CandidatoEntidades]


class ResultadoVeracidade(Contract):
    classificacao: Classificacao
    id_canonico: int | None = None
    justificativa: str = Field(min_length=1)
    confianca: float | None = Field(default=None, ge=0, le=1)

    @model_validator(mode="after")
    def validar_resolucao(self) -> ResultadoVeracidade:
        if self.classificacao == Classificacao.REAL and self.id_canonico is None:
            raise ValueError("classificação real exige id_canonico")
        if self.classificacao != Classificacao.REAL and self.id_canonico is not None:
            raise ValueError("somente classificação real aceita id_canonico")
        return self


class CandidatoClassificado(Contract):
    candidato: CandidatoCitacao
    completude: ResultadoCompletude
    veracidade: ResultadoVeracidade


class DocumentoClassificado(Contract):
    documento_id: str = Field(min_length=1)
    candidatos: list[CandidatoClassificado]
    texto: str = Field(default="", exclude=True)
    chamadas_modelo: list[dict[str, Any]] = Field(default_factory=list, exclude=True)


class Resolucao(Contract):
    id_canonico: int


class Predicao(Contract):
    inicio: int = Field(ge=0)
    fim: int = Field(gt=0)
    trecho: str = Field(min_length=1)
    tipo: TipoCitacao
    classificacao: Classificacao
    resolucao: Resolucao | None = None
    confianca: float | None = Field(default=None, ge=0, le=1)

    @model_validator(mode="after")
    def validar_resolucao(self) -> Predicao:
        if self.fim <= self.inicio:
            raise ValueError("fim deve ser maior que inicio")
        if self.classificacao == Classificacao.REAL and self.resolucao is None:
            raise ValueError("classificação real exige resolução")
        if self.classificacao != Classificacao.REAL and self.resolucao is not None:
            raise ValueError("somente classificação real aceita resolução")
        return self


class DocumentoPredito(Contract):
    documento_id: str = Field(min_length=1)
    citacoes: list[Predicao]


class ExtratorCandidatos(Protocol):
    def extrair_auditada(
        self,
        texto: str,
    ) -> tuple[list[CandidatoCitacao], dict[str, Any]]: ...


class ExtratorCandidatosAsync(Protocol):
    async def extrair_auditada_async(
        self,
        texto: str,
    ) -> tuple[list[CandidatoCitacao], dict[str, Any]]: ...


class ClassificadorCompletude(Protocol):
    def classificar_auditada(
        self,
        candidato: CandidatoCitacao,
        contexto: str,
    ) -> tuple[ResultadoCompletude, list[dict[str, Any]]]: ...


class ClassificadorCompletudeAsync(Protocol):
    async def classificar_auditada_async(
        self,
        candidato: CandidatoCitacao,
        contexto: str,
    ) -> tuple[ResultadoCompletude, list[dict[str, Any]]]: ...


class ExtratorEntidadesAsync(Protocol):
    async def extrair_auditada_async(
        self,
        candidato: CandidatoCitacao,
    ) -> tuple[ConsultaJurisprudencia | ConsultaLegislacao, dict[str, Any]]: ...


class VerificadorConsulta(Protocol):
    def verificar(
        self,
        consulta: ConsultaJurisprudencia | ConsultaLegislacao,
    ) -> tuple[ResultadoVeracidade, dict[str, Any]]: ...
