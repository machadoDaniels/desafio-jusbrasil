"""Focused extraction of tribunal fields."""

from ..contracts import CandidatoCitacao, Contract, Tribunal
from .shared import ExtractionRunner, format_examples


class TribunalFields(Contract):
    tribunal: Tribunal | None = None


EXAMPLES = [
    {
        "text": "RE 812.345, julgado pelo Supremo Tribunal Federal em 12/03/2022.",
        "expected": {"tribunal": "STF"},
    },
    {
        "text": "AgInt no REsp 765432/RS — decisão do STJ.",
        "expected": {"tribunal": "STJ"},
    },
    {
        "text": "Recurso Ordinário 00421-2024, Tribunal Superior do Trabalho.",
        "expected": {"tribunal": "TST"},
    },
    {
        "text": "Recurso Especial Eleitoral 91, Tribunal Superior Eleitoral.",
        "expected": {"tribunal": "TSE"},
    },
    {
        "text": "Apelação 18/2023, Superior Tribunal Militar (STM).",
        "expected": {"tribunal": "STM"},
    },
    {
        "text": "Súmula Vinculante 28, aplicada no julgamento; sem identificação do tribunal.",
        "expected": {"tribunal": None},
    },
]


PROMPT = """Identify the court that issued the decision cited in the original citation.
Return a value only when the citation explicitly writes the court's abbreviation or
full name. Normalize these full names to the allowed abbreviations: Supremo Tribunal
Federal → STF; Superior Tribunal de Justiça → STJ; Superior Tribunal Militar → STM;
Tribunal Superior Eleitoral → TSE; Tribunal Superior do Trabalho → TST. Normalize
recognized abbreviations to uppercase and ignore surrounding punctuation.

Return null when no allowed court is explicitly named. Do not infer a court from a
procedural class or acronym (including RE, REsp, RR, or recurso eleitoral), a CNJ
number, state or region, judge, or Súmula Vinculante. The phrase “Súmula Vinculante”
alone does not identify STF. A named court outside the allowed values also maps to
null. Use evidence from the citation only; do not copy values from these examples.

""" + format_examples(EXAMPLES)


async def extract(candidate: CandidatoCitacao, runner: ExtractionRunner):
    return await runner.extract("tribunal", PROMPT, TribunalFields, candidate)
