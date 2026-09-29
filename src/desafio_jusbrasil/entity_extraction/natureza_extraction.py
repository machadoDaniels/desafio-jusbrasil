"""Focused extraction of natureza fields."""

from pydantic import Field

from ..contracts import CandidatoCitacao, Contract, NaturezaJurisprudencia
from .shared import ExtractionRunner, format_examples


class NaturezaFields(Contract):
    natureza: NaturezaJurisprudencia | None = Field(
        default=None,
        description=(
            "Use 'sumula' for an explicitly cited súmula, including an unambiguous "
            "OCR variant; use 'acordao' for a case, appeal, or judgment citation."
        ),
    )
    numero_sumula: int | None = Field(
        default=None,
        ge=1,
        description="The integer printed as the number of the cited súmula; otherwise null.",
    )
    sumula_vinculante: bool | None = Field(
        default=None,
        description=(
            "True only when the cited item is explicitly labeled Súmula Vinculante; "
            "false for an ordinary numbered or unnumbered súmula; null when no súmula is cited."
        ),
    )


EXAMPLES = [
    {
        "text": "Súmula Vinculante nº 7 do STF.",
        "expected": {"natureza": "sumula", "numero_sumula": 7, "sumula_vinculante": True},
    },
    {
        "text": "Súmula 344 do Tribunal Superior do Trabalho.",
        "expected": {"natureza": "sumula", "numero_sumula": 344, "sumula_vinculante": False},
    },
    {
        "text": "5úmula n. 119, órgão julgador: TST.",
        "expected": {"natureza": "sumula", "numero_sumula": 119, "sumula_vinculante": False},
    },
    {
        "text": "Súmula Vinculante sem indicação do número.",
        "expected": {"natureza": "sumula", "numero_sumula": None, "sumula_vinculante": True},
    },
    {
        "text": "Recurso Especial nº 456.789/RS, julgado pela Segunda Turma.",
        "expected": {"natureza": "acordao", "numero_sumula": None, "sumula_vinculante": None},
    },
    {
        "text": "Agravo Interno no Recurso Extraordinário 7654321-09.2030.1.00.0000.",
        "expected": {"natureza": "acordao", "numero_sumula": None, "sumula_vinculante": None},
    },
]


PROMPT = """Classify the cited jurisprudence item and extract only its súmula attributes.

Use "sumula" when the citation explicitly names a Súmula, including the label
"Súmula Vinculante". A clear OCR substitution in the word (such as "5úmula") may
be corrected when the surrounding citation makes the intended label unambiguous.
Use "acordao" for a citation identifying a court case, appeal, or judgment when no
súmula is named. Do not infer that an item is a súmula merely from its number,
court, or legal subject.

For a súmula, copy only the integer directly identified as its number; if none is
given, use null. Set sumula_vinculante to true only when the cited label explicitly
includes "Vinculante". Set it to false for an ordinary súmula, including one whose
number is absent. When the citation is not a súmula, both sumula fields must be
null. Do not return a court name or any other fields.

""" + format_examples(EXAMPLES)


async def extract(candidate: CandidatoCitacao, runner: ExtractionRunner):
    return await runner.extract("natureza", PROMPT, NaturezaFields, candidate)
