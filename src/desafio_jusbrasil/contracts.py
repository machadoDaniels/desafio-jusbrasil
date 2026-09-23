"""Contratos compartilhados entre as etapas do pipeline."""

from __future__ import annotations

import re
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
ClasseProcessualAgente = Literal[
    "AI — Agravo de Instrumento",
    "APL — Apelação",
    "AR — Ação Rescisória",
    "AREsp — Agravo em Recurso Especial",
    "AREspe — Agravo em Recurso Especial Eleitoral",
    "ARR — Recurso de Revista com Agravo",
    "AgInt — Agravo Interno",
    "AgRg — Agravo Regimental",
    "E — Embargos",
    "EDcl — Embargos de Declaração",
    "HC — Habeas Corpus",
    "RE — Recurso Extraordinário",
    "REsp — Recurso Especial",
    "REspe — Recurso Especial Eleitoral",
    "RHC — Recurso em Habeas Corpus",
    "RMS — Recurso em Mandado de Segurança",
    "RR — Recurso de Revista",
    "RSE — Recurso em Sentido Estrito",
    "Rcl — Reclamação",
    "Rp — Representação",
    "SLS — Suspensão de Liminar e de Sentença",
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
DiplomaAgente = Literal[
    "CF — Constituição Federal",
    "CC — Código Civil",
    "CDC — Código de Defesa do Consumidor",
    "CPC — Código de Processo Civil",
    "CPP — Código de Processo Penal",
    "CPM — Código Penal Militar",
    "CE — Código Eleitoral",
    "CLT — Consolidação das Leis do Trabalho",
    "LC — Lei Complementar",
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
    numero_processo_cnj: str | None = Field(
        default=None,
        description=(
            "Número único CNJ do processo julgado, somente com 20 dígitos e zeros à "
            "esquerda. Não use CNJ de processo apenas mencionado."
        ),
    )
    numero_classe_tribunal: str | None = Field(
        default=None,
        pattern=r"^\d+$",
        description=(
            "Somente os dígitos do número sequencial associado à classe no tribunal. "
            "Não inclua classe, pontuação, UF ou número de registro. Exemplos: em "
            "'Rcl 68.244/SP', retorne '68244'; em 'REsp 1.741.784', retorne '1741784'."
        ),
        examples=["68244", "1741784"],
    )
    numero_registro_tribunal: str | None = Field(
        default=None,
        pattern=r"^\d+$",
        description=(
            "Somente os dígitos do número de registro interno explicitamente apresentado "
            "no formato AAAA/NNNNNNN-D. Não use o número associado à classe. Exemplos: "
            "'2022/0187319-4' vira '202201873194'; '2018/0116304-1' vira '201801163041'."
        ),
        examples=["202201873194", "201801163041"],
    )

    @field_validator(
        "numero_classe_tribunal", "numero_registro_tribunal", mode="before"
    )
    @classmethod
    def normalizar_identificadores(cls, valor: object) -> str | None:
        return re.sub(r"\D", "", str(valor or "")) or None

    classe_processual: ClasseProcessualAgente | None = Field(
        default=None,
        description="Classe do processo ou recurso principal. A última classe da cadeia recursal.",
    )
    cadeia_recursal: list[ClasseProcessualAgente] | None = Field(
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
        description="Ano do julgamento",
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


class ConsultaJurisprudencia(ConsultaJurisprudenciaAgente):
    """Consulta de jurisprudência normalizada para busca na base canônica."""

    classe_processual: ClasseProcessual | None = None
    cadeia_recursal: list[ClasseProcessual] | None = None
    relator_norm: str | None = None

    @field_validator("classe_processual", mode="before")
    @classmethod
    def normalizar_classe(cls, valor: object) -> object:
        return valor.split(" — ", 1)[0] if isinstance(valor, str) else valor

    @field_validator("cadeia_recursal", mode="before")
    @classmethod
    def normalizar_classes(cls, valor: object) -> object:
        if not isinstance(valor, list):
            return valor
        return [
            item.split(" — ", 1)[0] if isinstance(item, str) else item for item in valor
        ]

    @model_validator(mode="after")
    def anular_ano_de_cnj(self) -> ConsultaJurisprudencia:
        if self.numero_processo_cnj is not None:
            self.ano = None
        return self

    @field_validator(
        "numero_processo_cnj", "numero_classe_tribunal", "numero_registro_tribunal"
    )
    @classmethod
    def validar_numeros_normalizados(cls, valor: str | None, info: Any) -> str | None:
        if valor is None:
            return None
        if not valor.isascii() or not valor.isdigit():
            raise ValueError(f"{info.field_name} deve conter somente dígitos ASCII")
        if info.field_name == "numero_processo_cnj" and len(valor) != 20:
            raise ValueError("numero_processo_cnj deve conter 20 dígitos")
        return valor


class ConsultaLegislacaoAgente(Contract):
    """Entidades de legislação solicitadas ao modelo."""

    numero_artigo: str | None = Field(
        default=None, description="Número do artigo sem o prefixo 'Art.'."
    )

    @field_validator("numero_artigo")
    @classmethod
    def normalizar_numero_artigo(cls, valor: str | None) -> str | None:
        return re.sub(r"\D", "", valor or "") or None

    diploma: DiplomaAgente | None = Field(
        default=None, description="Sigla e nome canônico do diploma."
    )
    numero_diploma: str | None = Field(
        default=None, description="Número do diploma somente com dígitos."
    )

    @field_validator("numero_diploma", mode="before")
    @classmethod
    def normalizar_numero_diploma(cls, valor: object) -> str | None:
        return re.sub(r"\D", "", str(valor or "")) or None

    ano_diploma: int | None = Field(
        default=None, ge=1, le=9999, description="Ano do diploma com quatro dígitos."
    )


class ConsultaLegislacao(ConsultaLegislacaoAgente):
    """Consulta de legislação normalizada para busca na base canônica."""

    diploma: Diploma | None = None

    @field_validator("diploma", mode="before")
    @classmethod
    def normalizar_diploma(cls, valor: object) -> object:
        return valor.split(" — ", 1)[-1] if isinstance(valor, str) else valor


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
    ) -> tuple[ConsultaJurisprudencia | ConsultaLegislacao, list[dict[str, Any]]]: ...


class VerificadorConsulta(Protocol):
    def verificar(
        self,
        consulta: ConsultaJurisprudencia | ConsultaLegislacao,
    ) -> tuple[ResultadoVeracidade, dict[str, Any]]: ...
