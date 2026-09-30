"""Focused extraction of relator fields."""

from ..contracts import CandidatoCitacao, Contract
from .shared import ExtractionRunner, format_examples


class RelatorFields(Contract):
    relator: str | None = None


EXAMPLES = [
    {
        "text": "Apelação Cível 0048217-35.2021.8.26.0000. Relatora: Helena Duarte Nogueira.",
        "expected": {"relator": "Helena Duarte Nogueira"},
    },
    {
        "text": "AgInt no REsp 8.431/RS. Rel. Min. Otávio Leme Farias.",
        "expected": {"relator": "Otávio Leme Farias"},
    },
    {
        "text": "RE 741.208/PR, relator Des. Federal Caio Monteiro Paes.",
        "expected": {"relator": "Caio Monteiro Paes"},
    },
    {
        "text": "HC 0062/GO. A defesa foi apresentada pela advogada Marina Teles; não há relator identificado.",
        "expected": {"relator": None},
    },
    {
        "text": "No julgamento da Apelação 13/2030, acompanhou-se o voto do relator, sem indicação de seu nome.",
        "expected": {"relator": None},
    },
]


PROMPT = """Extract the name explicitly identified as the reporting judge (relator or
relatora) in the citation. Copy the name as written, preserving spelling, accents,
initials, partial names, and name order; an explicitly attributed initial or partial
name is still evidence and must not be discarded for being incomplete. Remove only a
title or role prefix attached to that name,
such as Min., Des., Des. Federal, or Juiz(a); do not expand initials or abbreviations.
Do not return names of lawyers, parties, authors, or other judges unless explicitly
identified as the relator. A generic mention of "o relator" without any name is not a
name. Return null when no reporting judge is explicitly named or attribution is
ambiguous. Do not infer or complete names from legal knowledge.
The downstream process handles deterministic name normalization.

""" + format_examples(EXAMPLES)


async def extract(candidate: CandidatoCitacao, runner: ExtractionRunner):
    return await runner.extract("relator", PROMPT, RelatorFields, candidate)
