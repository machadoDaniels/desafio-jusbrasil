"""Contratos estritos para o enriquecimento do banco canônico."""

from __future__ import annotations

from typing import get_args, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationInfo,
    create_model,
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
type ClasseProcessualAgente = Literal[
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
type DiplomaAgente = Literal[
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
    classe_processual: ClasseProcessualAgente | None = Field(
        default=None,
        description=(
            "Classe principal do processo ou do recurso principal, usando uma das siglas "
            "permitidas. Identifique-a independentemente da posição em cadeia_recursal; "
            "não escolha automaticamente o recurso incidental mais recente."
        ),
    )
    cadeia_recursal: list[ClasseProcessualAgente] | None = Field(
        default=None,
        validate_default=True,
        # Sem repetições, uma entrada por classe permitida é o teto. Um array limitado
        # mantém a decodificação guiada finita: sem isso o modelo repete o mesmo item
        # até o limite de tokens (14/100 documentos na rodada de 2026-09-29).
        max_length=len(get_args(ClasseProcessualAgente.__value__)),
        description=(
            "Todas as classes processuais explicitamente presentes na cadeia recursal, "
            "sem ordem nem repetições."
        ),
    )

    @field_validator("cadeia_recursal")
    @classmethod
    def incluir_classe_na_cadeia(
        cls, valor: list[str] | None, info: ValidationInfo
    ) -> list[str] | None:
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

    classe_processual: ClasseProcessual | None = None
    cadeia_recursal: list[ClasseProcessual] | None = Field(default=None, validate_default=True)
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


class _MetadadosDispositivoAgente(Contract):
    """Metadados de dispositivo legal que devem ser extraídos pelo agente."""

    diploma: DiplomaAgente | None = Field(
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
    def validar_numeros(self) -> _MetadadosDispositivoAgente:
        if self.numero_diploma is not None and (
            not self.numero_diploma.isascii() or not self.numero_diploma.isdigit()
        ):
            raise ValueError("numero_diploma deve conter somente dígitos ASCII")
        if self.numero_artigo is not None and not self.numero_artigo.strip():
            raise ValueError("numero_artigo não pode ser vazio")
        return self


class MetadadosDispositivo(_MetadadosDispositivoAgente):
    """Metadados de dispositivo legal com o diploma canônico."""

    diploma: Diploma | None = None

    @field_validator("diploma", mode="before")
    @classmethod
    def normalizar_diploma(cls, valor: object) -> object:
        return valor.split(" — ", 1)[-1] if isinstance(valor, str) else valor


type MetadadosDocumento = MetadadosAcordao | MetadadosSumula | MetadadosDispositivo


def _com_trechos(contrato: type[Contract]) -> type[Contract]:
    """Contrato de requisição com os trechos literais antes dos campos normalizados.

    Os validadores do contrato original são aplicados depois, ao revalidar a resposta.
    """
    campos = {
        nome: (info.annotation, info)
        for nome, info in contrato.model_fields.items()
        if nome != "relator_norm"
    }
    trechos = create_model(
        f"Trechos{contrato.__name__.removeprefix('_')}",
        __base__=Contract,
        **{
            nome: (
                list[str] | None if nome == "cadeia_recursal" else str | None,
                Field(
                    default=None,
                    description=(
                        (
                            "Um trecho por classe da cadeia recursal, na mesma ordem. "
                            if nome == "cadeia_recursal"
                            else ""
                        )
                        + f"Trecho copiado exatamente como aparece no texto e que sustenta "
                        f"{nome}, antes de qualquer normalização: mantenha pontos, "
                        "barras, hífens, abreviações e maiúsculas (por exemplo, "
                        "'2019/0172353-7', 'AG.REG. NA RECLAMAÇÃO 41.908', 'SANTA "
                        "CATARINA', 'Decreto-Lei nº 5.452'). Nunca copie o valor já "
                        "normalizado. Use null se o campo for nulo."
                    ),
                ),
            )
            for nome in campos
        },
    )
    return create_model(
        f"{contrato.__name__.removeprefix('_')}ComTrechos",
        __base__=Contract,
        trechos=(trechos, ...),
        **campos,
    )


CONTRATOS_COM_TRECHOS: dict[NaturezaDocumento, type[Contract]] = {
    "acordao": _com_trechos(_MetadadosAcordaoAgente),
    "sumula": _com_trechos(MetadadosSumula),
    "dispositivo": _com_trechos(_MetadadosDispositivoAgente),
}


class DocumentoEnriquecido(Contract):
    """Resultado validado e pronto para materialização no SQLite."""

    documento_id: str = Field(min_length=1)
    id: int
    natureza: NaturezaDocumento
    campos: MetadadosDocumento
    trechos: dict[str, str | list[str] | None] | None = None

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
