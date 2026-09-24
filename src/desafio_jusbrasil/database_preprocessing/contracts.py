"""Contratos estritos para o enriquecimento do banco canônico."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationInfo,
    field_validator,
    model_validator,
)

type NaturezaDocumento = Literal["acordao", "sumula", "dispositivo"]
type UF = Literal[
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
type ClasseProcessual = Literal[
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
type Diploma = Literal[
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


class Contract(BaseModel):
    """Base que rejeita campos não previstos em respostas do modelo."""

    model_config = ConfigDict(extra="forbid")


class DocumentoFonte(Contract):
    """Documento canônico lido do SQLite antes do enriquecimento."""

    documento_id: str = Field(min_length=1)
    id: int
    natureza: NaturezaDocumento
    tribunal: str | None = None
    ano: int | None = None
    relator: str | None = None
    texto: str = Field(min_length=1)


class _MetadadosAcordaoAgente(Contract):
    """Metadados de acórdão que devem ser extraídos pelo agente."""

    numero_processo_cnj: str | None = Field(
        default=None,
        pattern=r"^[0-9]+$",
        description=(
            "Número único CNJ do processo julgado, somente com 20 dígitos. Não extraia "
            "CNJ de processo apenas mencionado no texto."
        ),
    )
    numero_classe_tribunal: str | None = Field(
        default=None,
        pattern=r"^[0-9]+$",
        description=(
            "Número sequencial do feito junto da classe do tribunal, somente com dígitos: "
            "por exemplo, Rcl 76532 ou REsp 1741784."
        ),
    )
    numero_registro_tribunal: str | None = Field(
        default=None,
        pattern=r"^[0-9]+$",
        description=(
            "Número de registro do tribunal, somente com dígitos: por exemplo, o STJ "
            "2022/0187319-4 vira 202201873194."
        ),
    )
    classe_processual: ClasseProcessual | None = Field(
        default=None,
        description=(
            "Classe principal do processo ou do recurso principal, usando uma das siglas "
            "permitidas. A última classe da cadeia recursal."
        ),
    )
    cadeia_recursal: list[ClasseProcessual] | None = Field(
        default=None,
        validate_default=True,
        description=(
            "Todas as classes processuais explicitamente presentes na cadeia recursal, "
            "sem ordem nem repetições. A cadeia recursal se repete aqui."
        ),
    )

    @field_validator("cadeia_recursal")
    @classmethod
    def incluir_classe_na_cadeia(
        cls, valor: list[ClasseProcessual] | None, info: ValidationInfo
    ) -> list[ClasseProcessual] | None:
        classe = info.data.get("classe_processual")
        if classe is not None:
            valor = [*(valor or []), classe]
        return list(dict.fromkeys(valor)) if valor else None

    uf: UF | None = Field(
        default=None,
        description=(
            "Sigla da unidade federativa à qual o processo está vinculado. Não confunda "
            "a sigla RR da classe Recurso de Revista com o estado de Roraima. Use null "
            "quando a UF não estiver sustentada pelo documento."
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
            raise ValueError("números devem conter somente dígitos ASCII")
        return valor



class MetadadosAcordao(_MetadadosAcordaoAgente):
    """Metadados de acórdão completos após normalização determinística."""

    relator_norm: str | None = None

    @field_validator("numero_processo_cnj")
    @classmethod
    def validar_cnj_normalizado(cls, valor: str | None) -> str | None:
        if valor is not None and len(valor) != 20:
            raise ValueError("numero_processo_cnj deve conter 20 dígitos")
        return valor

    @field_validator("relator_norm")
    @classmethod
    def validar_relator_norm(cls, valor: str | None) -> str | None:
        if valor is not None and valor != " ".join(valor.lower().split()):
            raise ValueError(
                "relator_norm deve estar em minúsculas e sem espaços duplicados"
            )
        return valor


class MetadadosSumula(Contract):
    """Metadados extraídos de uma súmula."""

    numero_sumula: int | None = Field(
        default=None,
        ge=1,
        description="Número inteiro da súmula, sem prefixos, pontuação ou texto adicional.",
    )
    sumula_vinculante: bool | None = Field(
        default=None,
        description=(
            "Use true somente quando o documento identificar expressamente uma Súmula "
            "Vinculante do STF. Use false para súmulas comuns, inclusive do STF, STJ, "
            "TST e outros tribunais. Use null apenas se não for possível determinar."
        ),
    )


class MetadadosDispositivo(Contract):
    """Metadados extraídos de um dispositivo legal."""

    diploma: Diploma | None = Field(
        default=None,
        description=(
            "Tipo canônico do diploma. Normalize Decreto-Lei 5.452/1943 como "
            "'Consolidação das Leis do Trabalho', Decreto-Lei 3.689/1941 como "
            "'Código de Processo Penal', Decreto-Lei 1.001/1969 como 'Código Penal "
            "Militar', Lei 10.406/2002 como 'Código Civil', Lei 8.078/1990 como "
            "'Código de Defesa do Consumidor' e Lei 13.105/2015 como 'Código de "
            "Processo Civil'. Use somente um dos valores permitidos."
        ),
    )
    numero_diploma: str | None = Field(
        default=None,
        pattern=r"^[0-9]+$",
        description=(
            "Número do diploma contendo somente dígitos ASCII, sem pontos. Use null para "
            "a Constituição Federal ou quando o documento não trouxer número."
        ),
    )
    ano_diploma: int | None = Field(
        default=None,
        ge=1,
        le=9999,
        description="Ano de promulgação ou publicação do diploma, com quatro dígitos.",
    )
    numero_artigo: str | None = Field(
        default=None,
        description=(
            "Identificador do artigo sem o prefixo 'Art.'; preserve letras, hífens, "
            "ordinal e demais sufixos explicitamente presentes."
        ),
    )

    @model_validator(mode="after")
    def validar_numeros(self) -> MetadadosDispositivo:
        if self.numero_diploma is not None and (
            not self.numero_diploma.isascii() or not self.numero_diploma.isdigit()
        ):
            raise ValueError("numero_diploma deve conter somente dígitos ASCII")
        if self.numero_artigo is not None and not self.numero_artigo.strip():
            raise ValueError("numero_artigo não pode ser vazio")
        return self


type MetadadosDocumento = MetadadosAcordao | MetadadosSumula | MetadadosDispositivo


class DocumentoEnriquecido(Contract):
    """Resultado validado e pronto para materialização no SQLite."""

    documento_id: str = Field(min_length=1)
    id: int
    natureza: NaturezaDocumento
    campos: MetadadosDocumento

    @model_validator(mode="after")
    def validar_campos_por_natureza(self) -> DocumentoEnriquecido:
        contratos = {
            "acordao": MetadadosAcordao,
            "sumula": MetadadosSumula,
            "dispositivo": MetadadosDispositivo,
        }
        contrato = contratos[self.natureza]
        if not isinstance(self.campos, contrato):
            raise ValueError(  # noqa: TRY004 - Pydantic converte ValueError em ValidationError.
                f"natureza {self.natureza} exige {contrato.__name__}"
            )
        return self
