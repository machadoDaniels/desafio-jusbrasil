"""Focused extraction of numero_diploma fields."""

from pydantic import Field

from ..contracts import CandidatoCitacao, Contract
from .shared import ExtractionRunner, format_examples


class LawNumberFields(Contract):
    numero_diploma: str | None = Field(default=None, pattern=r"^\d+$")


EXAMPLES = [
    {
        "text": "Aplica-se a Lei nº 14.208/2041 ao caso, conforme seu art. 7º.",
        "expected": {"numero_diploma": "14208"},
    },
    {
        "text": "A Lei Complementar n. 83, de 2042, disciplina a matéria.",
        "expected": {"numero_diploma": "83"},
    },
    {
        "text": "O Decreto-Lei 2.619/1940 foi mencionado no voto.",
        "expected": {"numero_diploma": "2619"},
    },
    {
        "text": "O art. 19 da Lei 7.451/2043 é aplicável.",
        "expected": {"numero_diploma": "7451"},
    },
    {
        "text": "A defesa fundamentou o pedido no Código de Processo Civil (CPC).",
        "expected": {"numero_diploma": None},
    },
    {
        "text": "A Constituição Federal de 1988 foi citada, sem número de lei.",
        "expected": {"numero_diploma": None},
    },
    {
        "text": "O processo nº 0012345-67.2021.8.26.0001 menciona o art. 12, mas não identifica lei numerada.",
        "expected": {"numero_diploma": None},
    },
]


PROMPT = """Extract the number of the law or other legal instrument only when the
instrument's number is explicitly written in the citation. Return the number as
digits only, removing separators such as periods and spaces. For forms like
"Lei nº 14.208/2041" or "Lei Complementar 83, de 2042", return the instrument
number (14208 or 83), never the year after a slash or the words "de".

Do not mistake an article, paragraph, subsection, item, or case number for the
instrument number. In "art. 19 da Lei 7.451/2043", the law number is 7451, not 19
or 2043. A code's name or abbreviation alone (for example, CPC, Código Civil, or
CLT) does not state the number of its instituting law: return null instead of
recalling that number from legal knowledge. A constitution cited by name or year
also has no explicit instrument number here; do not return 1988. Return null
whenever the citation gives no explicit number for a legal instrument.

""" + format_examples(EXAMPLES)


async def extract(candidate: CandidatoCitacao, runner: ExtractionRunner):
    return await runner.extract("numero_diploma", PROMPT, LawNumberFields, candidate)
