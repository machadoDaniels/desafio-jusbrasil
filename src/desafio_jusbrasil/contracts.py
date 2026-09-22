"""Contratos compartilhados entre as etapas do pipeline."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Any, Literal, Protocol

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


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
    debug: bool = False


class PipelineConfig(Contract):
    input_dir: Path
    workdir: Path
    database: Path
    extractor: StageConfig
    completeness: StageConfig
    veracity: StageConfig

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


class ConsultaJurisprudencia(Contract):
    valores_fts: list[str] = Field(min_length=1)
    natureza: Literal["acordao", "sumula"]
    tribunal: str | None = None
    ano: int | None = None
    relator: str | None = None


class ConsultaLegislacao(Contract):
    valores_fts: list[str] = Field(min_length=1)


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


class ClassificadorVeracidade(Protocol):
    def classificar_auditada(
        self,
        candidato: CandidatoCitacao,
    ) -> tuple[ResultadoVeracidade, dict[str, Any]]: ...


class ClassificadorVeracidadeAsync(Protocol):
    async def classificar_auditada_async(
        self,
        candidato: CandidatoCitacao,
    ) -> tuple[ResultadoVeracidade, dict[str, Any]]: ...
