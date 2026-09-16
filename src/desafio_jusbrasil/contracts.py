"""Contratos compartilhados entre as etapas do pipeline."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Protocol

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


class PipelineConfig(Contract):
    input_dir: Path
    workdir: Path
    database: Path
    model: str = "gpt-4.1-mini"
    base_url: str | None = None

    @classmethod
    def from_yaml(cls, caminho: Path) -> PipelineConfig:
        dados = yaml.safe_load(caminho.read_text(encoding="utf-8"))
        return cls.model_validate(dados)


class CandidatoCitacao(Contract):
    tipo: TipoCitacao
    trecho: str = Field(min_length=1)
    inicio: int = Field(ge=0)
    fim: int = Field(gt=0)
    confianca_extracao: float | None = Field(default=None, ge=0, le=1)

    @model_validator(mode="after")
    def validar_intervalo(self) -> CandidatoCitacao:
        if self.fim <= self.inicio:
            raise ValueError("fim deve ser maior que inicio")
        return self


class LoteCandidatos(Contract):
    candidatos: list[CandidatoCitacao]


class DocumentoExtraido(Contract):
    documento_id: str = Field(min_length=1)
    texto: str
    candidatos: list[CandidatoCitacao]

    @model_validator(mode="after")
    def validar_spans(self) -> DocumentoExtraido:
        for candidato in self.candidatos:
            if candidato.fim > len(self.texto):
                raise ValueError("span fora dos limites do documento")
            if self.texto[candidato.inicio : candidato.fim] != candidato.trecho:
                raise ValueError("trecho não corresponde ao span do documento")
        return self


class ConsultaCanonica(Contract):
    tipo: TipoCitacao
    classe_processual: str | None = None
    numero: str | None = None
    tribunal: str | None = None
    uf: str | None = None
    ano: int | None = None
    relator: str | None = None
    dispositivo: str | None = None


class ResultadoCompletude(Contract):
    completa: bool
    consulta: ConsultaCanonica | None = None
    justificativa: str = Field(min_length=1)
    confianca: float | None = Field(default=None, ge=0, le=1)

    @model_validator(mode="after")
    def validar_consulta(self) -> ResultadoCompletude:
        if self.completa and self.consulta is None:
            raise ValueError("citação completa exige consulta")
        if not self.completa and self.consulta is not None:
            raise ValueError("citação incompleta não deve conter consulta")
        return self


class CandidatoAnalisado(Contract):
    candidato: CandidatoCitacao
    completude: ResultadoCompletude


class DocumentoCompletude(Contract):
    documento_id: str = Field(min_length=1)
    texto: str
    candidatos: list[CandidatoAnalisado]


class RegistroCanonico(Contract):
    documento_id: str
    id_canonico: int
    tribunal: str | None = None
    ano: int | None = None
    relator: str | None = None
    natureza: str
    tipo: TipoCitacao
    texto: str


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
    texto: str
    candidatos: list[CandidatoClassificado]


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
    def extrair(self, texto: str) -> list[CandidatoCitacao]: ...


class ClassificadorCompletude(Protocol):
    def classificar(
        self,
        candidato: CandidatoCitacao,
        contexto: str,
    ) -> ResultadoCompletude: ...


class ClassificadorVeracidade(Protocol):
    def classificar(self, consulta: ConsultaCanonica) -> ResultadoVeracidade: ...
