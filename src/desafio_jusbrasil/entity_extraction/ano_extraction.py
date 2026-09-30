"""Focused extraction of ano fields."""

from pydantic import Field

from ..contracts import CandidatoCitacao, Contract
from .shared import ExtractionRunner, format_examples


class YearFields(Contract):
    ano: int | None = Field(
        default=None,
        ge=1,
        le=9999,
        description=(
            "Year explicitly identified as the year the cited judgment was decided. "
            "Do not use years from case numbers, filing or publication dates, "
            "statutes, or decisions whose years are not attributed to the citation."
        ),
    )


EXAMPLES = [
    {
        "text": "Apelação Cível nº 0004321-15.2020.8.26.0000, julgada em 17 de março de 2022.",
        "expected": {"ano": 2022},
    },
    {
        "text": "No REsp 987.654/RS, o julgamento ocorreu em 2021.",
        "expected": {"ano": 2021},
    },
    {
        "text": "Rcl de 2031, Rel. Min. Helena Prado.",
        "expected": {"ano": 2031},
    },
    {
        "text": "Precedente citado: acórdão de 2032, Rel. Des. Caio Nogueira.",
        "expected": {"ano": 2032},
    },
    {
        "text": "O processo nº 0007654-32.2019.4.02.0000 menciona a Lei nº 8.078/1990.",
        "expected": {"ano": None},
    },
    {
        "text": "Decisão publicada no DJe de 14/08/2023, sem indicação de quando foi julgada.",
        "expected": {"ano": None},
    },
    {
        "text": "O acórdão foi julgado em 6/5/2020 e publicado em 2021.",
        "expected": {"ano": 2020},
    },
]


PROMPT = """Extract the year explicitly attributed to the cited judgment and return it as an
integer. Direct wording such as "julgado em" or "julgamento em" qualifies. A year
directly attached to the citation's decision label also qualifies, such as
"acórdão de 2020" or a compact reference "Rcl de 2020, Rel. Min. ..."; it is an
explicit citation year, not a year inferred from the case number.

Ignore years that belong to a CNJ or other case/registration number, a statute or
legal identifier, a filing date, or a publication or DJe date. A year explicitly
attributed to a cited precedent is the year of that cited judgment and may be
extracted. Do not infer a year from context, numbering, publication, or legal
knowledge. If no year is explicitly tied to the cited judgment, return null. If
multiple years appear, select only the year directly tied to the cited judgment;
return null when that relation is unclear.

""" + format_examples(EXAMPLES)


async def extract(candidate: CandidatoCitacao, runner: ExtractionRunner):
    return await runner.extract("ano", PROMPT, YearFields, candidate)
