"""Focused extraction of classe_processual fields."""

from typing import get_args

from pydantic import Field

from ..contracts import CandidatoCitacao, Contract, ClasseProcessualAgente
from .shared import ExtractionRunner, format_examples


class ProcessClassFields(Contract):
    classe_processual: ClasseProcessualAgente | None = None
    cadeia_recursal: list[ClasseProcessualAgente] | None = Field(
        default=None, max_length=len(get_args(ClasseProcessualAgente)),
    )


_BASE = "REsp — Recurso Especial"
_AGINT = "AgInt — Agravo Interno"
_EDCL = "EDcl — Embargos de Declaração"
_ARESP = "AREsp — Agravo em Recurso Especial"
_AGRG = "AgRg — Agravo Regimental"


EXAMPLES = [
    {
        "text": "No AgInt no ARESP 482731/RS, a turma manteve a decisão.",
        "expected": {"classe_processual": _ARESP, "cadeia_recursal": [_AGINT, _ARESP]},
    },
    {
        "text": "Foram opostos embargos de declaração no agravo regimental no agravo em recurso especial.",
        "expected": {
            "classe_processual": _ARESP,
            "cadeia_recursal": [_EDCL, _AGRG, _ARESP],
        },
    },
    {
        "text": "O recurso ordinário em mandado de segurança foi julgado pela turma.",
        "expected": {
            "classe_processual": "RMS — Recurso em Mandado de Segurança",
            "cadeia_recursal": ["RMS — Recurso em Mandado de Segurança"],
        },
    },
    {
        "text": "O processo TST-ED-E-RR-31800-22.2019.5.03.0008 teve os embargos rejeitados.",
        "expected": {
            "classe_processual": "RR — Recurso de Revista",
            "cadeia_recursal": [
                _EDCL, "E — Embargos", "RR — Recurso de Revista",
            ],
        },
    },
    {
        "text": "O ED no AgR-REspe nº 731-42 foi apreciado pelo colegiado eleitoral.",
        "expected": {
            "classe_processual": "REspe — Recurso Especial Eleitoral",
            "cadeia_recursal": [
                _EDCL,
                _AGRG, "REspe — Recurso Especial Eleitoral",
            ],
        },
    },
    {
        "text": "O agravo em recurso especial eleitoral foi distribuído para julgamento.",
        "expected": {
            "classe_processual": "AREspe — Agravo em Recurso Especial Eleitoral",
            "cadeia_recursal": ["AREspe — Agravo em Recurso Especial Eleitoral"],
        },
    },
    {
        "text": "O AgREsp 81924 foi julgado pela seção competente.",
        "expected": {
            "classe_processual": _BASE,
            "cadeia_recursal": [_AGRG, _BASE],
        },
    },
    {
        "text": "A decisão transitou em julgado em 2022, sem indicação da classe processual.",
        "expected": {"classe_processual": None, "cadeia_recursal": None},
    },
]


PROMPT = """Extract every explicitly stated Brazilian case or appeal class from the
citation. Return only the exact canonical values allowed by the schema. Recognize
class abbreviations and their unambiguous full names; do not infer a class from a
court, a case number, or generic words such as "recurso" or "processo".

classe_processual is the underlying/base class: in nested appeal wording, it is the
innermost class after the final "no" (for example, in "AgInt no REsp", it is REsp).
cadeia_recursal lists the explicitly named classes in citation order, from the
outermost proceeding to the underlying class. Include each class once, preserving
its first occurrence. When exactly one class is stated, return it in both fields.
When no class is stated, return null for both. Do not include numbers or other
descriptive words in either field.

Class abbreviations may be packed into a docket string separated by hyphens,
periods, or spaces (for example, "TST-ED-E-RR"). Read each class token in order,
left to right, and keep distinct tokens in the chain. E means Embargos; ED, EDcl,
EDs, and "embargos de declaração" mean EDcl. Repeated occurrences of EDcl count
once, while E is a separate class and must remain in the chain. AgR, AgRg, and
Ag.Reg. mean AgRg. In a packed sequence, AgR-REsp or AgR-REspe means two classes
(AgRg followed by REsp or REspe); do not collapse them into AREsp/AREspe.
The compact form AgREsp means "AgRg no REsp": return AgRg then REsp in the chain and
REsp as classe_processual. AgARR means "AgRg no ARR": include both classes and set
ARR as classe_processual. ARR remains the base class even though its name contains
"Recurso de Revista"; do not shorten it to RR. Class abbreviation capitalization
can vary; ARESP and AREsp both mean the single class AREsp. The full
phrase "Agravo em Recurso Especial" is the single class AREsp, and its full
electoral form "Agravo em Recurso Especial Eleitoral" is AREspe; in a nested chain,
that is also the base class. AREsp and AREspe, including minor punctuation such as
A.REsp, each name one class. Preserve the distinction between REsp, REspe, AREsp,
and AREspe.

Common abbreviations and names include: AI = Agravo de Instrumento; APL = Apelação;
AR = Ação Rescisória; AREsp = Agravo em Recurso Especial; AREspe = Agravo em
Recurso Especial Eleitoral; ARR = Recurso de Revista com Agravo; AgInt = Agravo
Interno; AgRg = Agravo Regimental; E = Embargos; EDcl = Embargos de Declaração;
HC = Habeas Corpus; RE = Recurso Extraordinário; REsp = Recurso Especial; REspe =
Recurso Especial Eleitoral; RHC = Recurso em Habeas Corpus; RMS = Recurso em
Mandado de Segurança (also called Recurso Ordinário em Mandado de Segurança);
RR = Recurso de Revista; RSE = Recurso em Sentido Estrito;
Rcl = Reclamação; Rp = Representação; SLS = Suspensão de Liminar e de Sentença.
Keep distinct classes distinct: AgInt and AgRg are not AI; REspe is not REsp;
AREsp is not REsp. A class mentioned as a cited precedent still counts when
explicitly written, but do not infer an omitted class from context.

""" + format_examples(EXAMPLES)


async def extract(candidate: CandidatoCitacao, runner: ExtractionRunner):
    return await runner.extract("classe_processual", PROMPT, ProcessClassFields, candidate)
