"""Stage 3: extract citation entities with parallel, focused model calls."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import AsyncOpenAI
from tqdm import tqdm

from .contracts import (
    CandidatoAnalisado, CandidatoEntidades, DocumentoCompletude,
    DocumentoEntidades, ExtratorEntidadesAsync, PipelineConfig,
)
from .entity_extraction import (
    EntityExtractionError as ErroExtracaoEntidades,
    EntityExtractor as AgenteExtratorEntidades,
)
from .utils import escrever_manifesto_etapa, escrever_saida_documento, ler_documento, listar_resultados


async def _extrair_entidade(
    analisado: CandidatoAnalisado,
    extrator: ExtratorEntidadesAsync,
    semaforo: asyncio.Semaphore,
) -> tuple[CandidatoEntidades, list[dict[str, Any]]]:
    async with semaforo:
        campos, chamadas = await extrator.extrair_auditada_async(analisado.candidato)
    return CandidatoEntidades(
        candidato=analisado.candidato,
        completude=analisado.completude,
        campos_extraidos=campos,
    ), chamadas


async def _processar_documento_entities(
    arquivo: Path,
    output_file: Path,
    extrator: ExtratorEntidadesAsync,
    semaforo: asyncio.Semaphore,
    progresso: Any,
) -> None:
    documento = ler_documento(arquivo, DocumentoCompletude)
    candidatos: list[CandidatoEntidades] = []
    chamadas: list[dict[str, Any]] = []
    if not documento.candidatos:
        escrever_saida_documento(
            output_file,
            DocumentoEntidades(documento_id=documento.documento_id, candidatos=[]),
            [],
        )
    for analisado in documento.candidatos:
        try:
            candidato, chamadas_citacao = await _extrair_entidade(
                analisado, extrator, semaforo
            )
        except ErroExtracaoEntidades as erro:
            chamadas.extend(erro.auditorias)
            escrever_saida_documento(
                output_file,
                DocumentoEntidades(
                    documento_id=documento.documento_id,
                    candidatos=candidatos,
                ),
                chamadas,
            )
            raise
        candidatos.append(candidato)
        chamadas.extend(chamadas_citacao)
        escrever_saida_documento(
            output_file,
            DocumentoEntidades(
                documento_id=documento.documento_id,
                candidatos=candidatos,
            ),
            chamadas,
        )
        progresso.update()


async def executar_entities_async(
    input_file: Path,
    output_file: Path,
    extrator: ExtratorEntidadesAsync,
    max_concurrency: int,
) -> None:
    arquivos = listar_resultados(input_file)
    total_citacoes = sum(
        len(ler_documento(arquivo, DocumentoCompletude).candidatos)
        for arquivo in arquivos
    )
    semaforo = asyncio.Semaphore(max_concurrency)
    progresso = tqdm(
        total=total_citacoes,
        desc="Extracting entities",
        unit="citation",
    )
    try:
        await asyncio.gather(
            *(
                _processar_documento_entities(
                    arquivo, output_file, extrator, semaforo, progresso
                )
                for arquivo in arquivos
            )
        )
    finally:
        progresso.close()


async def _executar_entities_async(config: PipelineConfig, destino: Path) -> None:
    etapa = config.entities
    async with AsyncOpenAI(base_url=etapa.base_url, max_retries=0) as cliente:
        await executar_entities_async(
            config.workdir / "02-completeness",
            destino,
            AgenteExtratorEntidades.from_config(cliente, config),
            etapa.max_concurrency,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("pipeline.yaml"))
    args = parser.parse_args()
    load_dotenv()
    config = PipelineConfig.from_yaml(args.config)
    destino = config.workdir / "03-entities"
    etapa = config.entities
    escrever_manifesto_etapa(destino, "entities", etapa)
    asyncio.run(_executar_entities_async(config, destino))
    print(f"{destino}: entity extraction complete")


if __name__ == "__main__":
    main()
