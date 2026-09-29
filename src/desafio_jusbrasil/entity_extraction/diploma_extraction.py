"""Focused extraction of diploma fields."""

from pydantic import Field

from ..contracts import CandidatoCitacao, Contract, Diploma
from .shared import ExtractionRunner, format_examples


class DiplomaFields(Contract):
    numero_artigo: str | None = Field(default=None, pattern=r"^\d+$")
    diploma: Diploma | None = None


EXAMPLES = [
    {
        "text": "Art. 37, § 4º, da Constituição Federal (CF).",
        "expected": {"numero_artigo": "37", "diploma": "Constituição Federal"},
    },
    {
        "text": "O art. 12-A do CDC foi citado no parecer.",
        "expected": {"numero_artigo": "12", "diploma": "Código de Defesa do Consumidor"},
    },
    {
        "text": "Aplica-se o art. 9º, II, da Lei nº 12.987/2021.",
        "expected": {"numero_artigo": "9", "diploma": "Lei"},
    },
    {
        "text": "Nos termos do § 3º do artigo 21 da LC nº 87/2020.",
        "expected": {"numero_artigo": "21", "diploma": "Lei Complementar"},
    },
    {
        "text": "O Código Civil foi mencionado, sem indicação de artigo.",
        "expected": {"numero_artigo": None, "diploma": "Código Civil"},
    },
    {
        "text": "Artigo 43, § 1º, inciso IV.",
        "expected": {"numero_artigo": "43", "diploma": None},
    },
]


PROMPT = """Extract two fields from the cited legal text. Use only explicit citation
evidence, plus an unambiguous legal-code identity when the citation explicitly names
that code or its known instituting law. Do not assess whether the citation is valid.

For numero_artigo, return only the article's digits. Remove the `art.`/`artigo`
prefix, ordinal marker, and any paragraph, item, inciso, or alínea numbers. Preserve
the article number itself if it contains a letter or suffix only to the extent the
schema permits; this schema accepts digits only, so return its numeric portion.
Never use a paragraph or inciso number as the article number.

For diploma, return the canonical allowed value. Normalize clear names and
abbreviations, such as CF/CRFB to Constituição Federal; CC to Código Civil; CDC to
Código de Defesa do Consumidor; CPC to Código de Processo Civil; CPP to Código de
Processo Penal; CPM to Código Penal Militar; CE to Código Eleitoral; and CLT to
Consolidação das Leis do Trabalho. The establishing statute also identifies its
code when cited by number: Lei 10.406/2002 is CC; Lei 8.078/1990 is CDC; Lei
13.105/2015 is CPC; Decreto-Lei 3.689/1941 is CPP; Decreto-Lei 1.001/1969 is CPM;
Lei 4.737/1965 is CE; and Decreto-Lei 5.452/1943 is CLT. Normalize these identities
to the code's canonical name even when the citation says only Lei or Decreto-Lei.
Do not default to Lei merely because the instrument is written as a numbered law.
For other numbered laws, use Lei or Lei Complementar according to the citation.
An article may be returned even when its diploma is
absent, and a diploma may be returned when no article is cited. Do not return a law
number, invent an article number, or infer a diploma merely from the topic.

Missing or ambiguous fields are null. Return only the two schema fields.""" + "\n\n" + format_examples(EXAMPLES)


async def extract(candidate: CandidatoCitacao, runner: ExtractionRunner):
    return await runner.extract("diploma", PROMPT, DiplomaFields, candidate)
