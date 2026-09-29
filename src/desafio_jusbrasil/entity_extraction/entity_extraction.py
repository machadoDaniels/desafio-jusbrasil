"""Coordinate focused entity extractors and normalize their merged output."""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Mapping
from typing import Any

from openai import AsyncOpenAI

from ..contracts import (
    CandidatoCitacao, ConsultaJurisprudencia, ConsultaLegislacao,
    ModelConfig, PipelineConfig, StageConfig, TipoCitacao,
)
from ..utils import normalizar_numero_cnj, normalizar_relator
from . import (
    ano_extraction, classe_processual_extraction, diploma_extraction,
    natureza_extraction, numero_diploma_extraction, numero_processo_cnj_extraction,
    relator_extraction, tribunal_extraction, uf_extraction,
)
from .shared import EntityExtractionError, ExtractionRunner

JURISPRUDENCE_EXTRACTORS = (
    natureza_extraction, numero_processo_cnj_extraction, classe_processual_extraction,
    tribunal_extraction, uf_extraction, ano_extraction, relator_extraction,
)
LEGISLATION_EXTRACTORS = (diploma_extraction, numero_diploma_extraction)


class EntityExtractor:
    """Run independent field groups concurrently through one shared request runner."""

    def __init__(
        self, client: AsyncOpenAI, config: ModelConfig,
        relatores: Mapping[str, str] | None = None,
    ) -> None:
        self.runner = ExtractionRunner(client, StageConfig.model_validate(config.model_dump()))
        self.judge_names = relatores or {}

    @classmethod
    def from_config(cls, client: AsyncOpenAI, config: PipelineConfig) -> EntityExtractor:
        path = config.database.parent / "relatores_padronizacao.json"
        return cls(client, config.entities, json.loads(path.read_text(encoding="utf-8")))

    def normalize(
        self, fields: dict[str, Any], candidate: CandidatoCitacao,
    ) -> ConsultaJurisprudencia | ConsultaLegislacao:
        """Preserve the existing deterministic CNJ, súmula and judge normalization."""
        if candidate.tipo == TipoCitacao.LEI:
            return ConsultaLegislacao.model_validate(fields)
        fields = dict(fields)
        number = fields.get("numero_sumula")
        nature = "sumula" if number is not None else fields.get("natureza") or "acordao"
        binding = fields.get("sumula_vinculante")
        if nature == "sumula" and binding is None:
            binding = bool(re.search(r"\bs[uú]mula\s+vinculante\b", candidate.trecho, re.IGNORECASE))
        cnj = normalizar_numero_cnj(fields.get("numero_processo_cnj"), candidate.trecho)
        if nature == "sumula":
            cnj = None
            fields.update(numero_classe_tribunal=None, numero_registro_tribunal=None)
        fields.update(
            natureza=nature, numero_processo_cnj=cnj, sumula_vinculante=binding,
            relator_norm=normalizar_relator(fields.get("relator"), self.judge_names),
        )
        return ConsultaJurisprudencia.model_validate(fields)

    async def extrair_auditada_async(
        self, candidate: CandidatoCitacao,
    ) -> tuple[ConsultaJurisprudencia | ConsultaLegislacao, list[dict[str, Any]]]:
        # Keep the public protocol used by both the pipeline and stage CLI.
        extractors = JURISPRUDENCE_EXTRACTORS if candidate.tipo == TipoCitacao.JURISPRUDENCIA else LEGISLATION_EXTRACTORS
        results = await asyncio.gather(
            *(extractor.extract(candidate, self.runner) for extractor in extractors),
            return_exceptions=True,
        )
        fields: dict[str, Any] = {}
        audits: list[dict[str, Any]] = []
        failures = []
        for result in results:
            if isinstance(result, BaseException):
                if isinstance(result, asyncio.CancelledError):
                    raise result
                failures.append(result)
                if isinstance(result, EntityExtractionError):
                    audits.extend(result.auditorias)
                continue
            extracted, call_audits = result
            audits.extend(call_audits)
            overlap = fields.keys() & extracted.keys()
            if overlap:
                failures.append(ValueError(f"Extractors returned overlapping fields: {sorted(overlap)}"))
            fields.update(extracted)
        if failures:
            raise EntityExtractionError(
                "Entity extraction failed: " + "; ".join(str(error) for error in failures), audits,
            ) from failures[0]
        try:
            return self.normalize(fields, candidate), audits
        except ValueError as error:
            raise EntityExtractionError(f"Merged entity validation failed: {error}", audits) from error
