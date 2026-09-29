"""Focused extraction of numero_processo_cnj fields."""

from pydantic import Field

from ..contracts import CandidatoCitacao, Contract
from .shared import ExtractionRunner, format_examples


class ProcessNumberFields(Contract):
    numero_processo_cnj: str | None = None
    numero_classe_tribunal: str | None = Field(default=None, pattern=r"^\d+$")
    numero_registro_tribunal: str | None = Field(default=None, pattern=r"^\d+$")


EXAMPLES = [
    {
        "text": "Apelação Cível 0034567-89.2023.8.21.0042.",
        "expected": {
            "numero_processo_cnj": "00345678920238210042",
            "numero_classe_tribunal": None,
            "numero_registro_tribunal": None,
        },
    },
    {
        "text": "AgR 1234567-08.2022.8.13.\n0009, conforme consulta aos autos.",
        "expected": {
            "numero_processo_cnj": "12345670820228130009",
            "numero_classe_tribunal": None,
            "numero_registro_tribunal": None,
        },
    },
    {
        "text": "Recurso Especial n. 1.234.567/PR.",
        "expected": {
            "numero_processo_cnj": None,
            "numero_classe_tribunal": "1234567",
            "numero_registro_tribunal": None,
        },
    },
    {
        "text": "AgInt no REsp 2.345.678; registro interno 2024/0123456-7.",
        "expected": {
            "numero_processo_cnj": None,
            "numero_classe_tribunal": "2345678",
            "numero_registro_tribunal": "202401234567",
        },
    },
    {
        "text": "Registro interno do processo: 2023/0987654-2.",
        "expected": {
            "numero_processo_cnj": None,
            "numero_classe_tribunal": None,
            "numero_registro_tribunal": "202309876542",
        },
    },
    {
        "text": "Súmula 742 do Tribunal Superior do Trabalho.",
        "expected": {
            "numero_processo_cnj": None,
            "numero_classe_tribunal": None,
            "numero_registro_tribunal": None,
        },
    },
]


PROMPT = """Extract only identifiers for the case being cited. Return null for each
identifier that is absent or ambiguous. Do not use article, paragraph, súmula,
judgment-year, or unrelated-case numbers as case identifiers.

numero_processo_cnj is the national CNJ process number: six digit groups of
lengths 7-2-4-1-2-4, in that order. It may use punctuation, omit punctuation,
or be split by whitespace or a line break. Copy every identifier digit in order
and remove separators. The result must contain exactly 20 digits: never prepend
zeros, including when the citation already prints a seven-digit first group;
never drop or rearrange a visible digit. If the first group is printed with
fewer than seven digits, preserve the visible digits and let normalization
restore any conventional leading zeros. Use this field for a number matching
the full six-group CNJ structure, even if a process class is also written.

numero_classe_tribunal is the serial number written with a process-class label,
such as REsp, Rcl, AREsp, RE, or HC, when no CNJ number identifies that same
cited case. Return only its digits and remove thousands separators. A process
class alone supplies no number. Do not use a year or an adjacent CNJ group as
this serial.

numero_registro_tribunal is an internal registration number only when the text
explicitly gives the registration in the form YYYY/NNNNNNN-D. Return all its
digits in order, removing slash and hyphen. Keep it separate from a class serial
and from a CNJ number; both registration and class serial may be present.

Repair an OCR character only when its numeric reading is unambiguous from the
surrounding number. Never skip or delete a nonnumeric character inside a number;
if its digit value is unclear, leave that identifier null. Do not complete
truncated numbers or infer missing digits.
For súmula citations, all three process-identifier fields are null.""" + "\n\n" + format_examples(EXAMPLES)


async def extract(candidate: CandidatoCitacao, runner: ExtractionRunner):
    return await runner.extract("numero_processo_cnj", PROMPT, ProcessNumberFields, candidate)
