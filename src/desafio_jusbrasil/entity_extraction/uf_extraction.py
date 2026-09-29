"""Focused extraction of uf fields."""

from ..contracts import CandidatoCitacao, Contract, UF
from .shared import ExtractionRunner, format_examples


class UFFields(Contract):
    uf: UF | None = None


EXAMPLES = [
    {"text": "Apelação 48271/PR.", "expected": {"uf": "PR"}},
    {"text": "Recurso Especial 7319 – CE, julgado em sessão pública.", "expected": {"uf": "CE"}},
    {"text": "Agravo 9056 (MA).", "expected": {"uf": "MA"}},
    {"text": "RR 1842, com julgamento pelo Tribunal Superior do Trabalho.", "expected": {"uf": None}},
    {"text": "Processo 0001234-56.2024.8.26.0001, do Tribunal de Justiça de São Paulo.", "expected": {"uf": None}},
    {"text": "Ação 3321, ajuizada na comarca de Belavista.", "expected": {"uf": None}},
]


PROMPT = """Extract a Brazilian state abbreviation only when it is explicitly written
as a suffix or parenthetical attached to the cited case number. Separators and
spacing may vary; normalize the two-letter abbreviation to uppercase. A CNJ court
code, a court or judge name, a city, or other geographic context does not supply
the UF. Do not guess a state from legal knowledge. Abbreviations that name a
procedural class, such as RR (Recurso de Revista), are not state evidence. Return
null whenever no explicit state suffix is present or its role is ambiguous.

""" + format_examples(EXAMPLES)


async def extract(candidate: CandidatoCitacao, runner: ExtractionRunner):
    return await runner.extract("uf", PROMPT, UFFields, candidate)
