"""Pré-processamento e enriquecimento do banco canônico."""

from .contracts import (
    DocumentoEnriquecido,
    DocumentoFonte,
    MetadadosAcordao,
    MetadadosDispositivo,
    MetadadosSumula,
    contrato_para_natureza,
)

__all__ = [
    "DocumentoEnriquecido",
    "DocumentoFonte",
    "MetadadosAcordao",
    "MetadadosDispositivo",
    "MetadadosSumula",
    "contrato_para_natureza",
]
