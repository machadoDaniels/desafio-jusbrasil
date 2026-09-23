"""Orquestra o enriquecimento auditável dos documentos canônicos."""

from __future__ import annotations

import asyncio
import csv
import hashlib
import json
import logging
import os
import re
import time
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI, OpenAIError, omit
from pydantic import ValidationError
from tqdm.auto import tqdm

from ..utils import normalizar_numero_processo, normalizar_relator

_PROMPT_VERSION = "2"
_SCHEMA_VERSION = "4"
_LOG = logging.getLogger(__name__)

_PROMPTS = {
    "acordao": """Extraia somente metadados jurídicos explicitamente sustentados pelo documento.
Não use conhecimento externo e não invente valores. Retorne número do processo somente com
algarismos; classifique-o como cnj, classico ou sem_numero. Extraia a classe principal, todas as
classes da cadeia recursal e a UF. A ordem da cadeia recursal não tem significado. Campos ausentes
devem ser nulos.""",
    "sumula": """Extraia somente os metadados explicitamente sustentados pela súmula. Não use
conhecimento externo e não invente valores. Informe o número da súmula e se ela é vinculante;
use nulo quando o documento não sustentar o campo.""",
    "dispositivo": """Extraia somente os metadados legislativos explicitamente sustentados pelo
dispositivo. Não use conhecimento externo e não invente valores. Normalize o diploma para o
vocabulário permitido e retorne números de diploma somente com algarismos. Preserve sufixos
alfanuméricos do artigo. Campos ausentes devem ser nulos.""",
}


def _agora() -> str:
    return datetime.now(UTC).isoformat()


def _json(valor: Any) -> Any:
    if hasattr(valor, "model_dump"):
        return valor.model_dump(mode="json")
    if isinstance(valor, Path):
        return str(valor)
    if isinstance(valor, Mapping):
        return {str(chave): _json(item) for chave, item in valor.items()}
    if isinstance(valor, (list, tuple)):
        return [_json(item) for item in valor]
    return valor


def _escrever_json(caminho: Path, dados: Any) -> None:
    caminho.parent.mkdir(parents=True, exist_ok=True)
    temporario = caminho.with_suffix(caminho.suffix + ".tmp")
    temporario.write_text(
        json.dumps(_json(dados), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporario.replace(caminho)


def _hash_arquivo(caminho: Path) -> str:
    digest = hashlib.sha256()
    with caminho.open("rb") as arquivo:
        for bloco in iter(lambda: arquivo.read(1024 * 1024), b""):
            digest.update(bloco)
    return digest.hexdigest()


def _configuracao(configuracao: Mapping[str, Any]) -> dict[str, Any]:
    if not configuracao.get("model"):
        raise ValueError("database_preprocessing.model é obrigatório")
    referencia_api_key = configuracao.get("api_key")
    api_key = None
    api_key_env = None
    if referencia_api_key:
        if not isinstance(referencia_api_key, str) or not referencia_api_key.startswith(
            "$"
        ):
            raise ValueError(
                "database_preprocessing.api_key deve referenciar $VARIAVEL"
            )
        api_key_env = referencia_api_key[1:]
        api_key = os.getenv(api_key_env)
        if not api_key:
            raise ValueError(f"variável de ambiente ausente: {api_key_env}")
    valores = {
        "_api_key": api_key,
        "api_key_env": api_key_env,
        "model": configuracao["model"],
        "base_url": configuracao.get("base_url"),
        "temperature": configuracao.get("temperature"),
        "top_p": configuracao.get("top_p"),
        "top_k": configuracao.get("top_k"),
        "reasoning_effort": configuracao.get("reasoning_effort"),
        "max_concurrency": configuracao.get("max_concurrency", 4),
        "max_retries": configuracao.get("max_retries", 3),
        "retry_delay_seconds": configuracao.get("retry_delay_seconds", 1),
        "request_interval_seconds": configuracao.get("request_interval_seconds", 0.75),
        "header_char_limit": configuracao.get("header_char_limit", 6000),
    }
    if valores["max_concurrency"] < 1 or valores["max_retries"] < 1:
        raise ValueError("max_concurrency e max_retries devem ser maiores que zero")
    if (
        valores["header_char_limit"] < 1
        or valores["retry_delay_seconds"] < 0
        or valores["request_interval_seconds"] < 0
    ):
        raise ValueError(
            "header_char_limit deve ser positivo e intervalos não podem ser negativos"
        )
    return valores


def _config_checkpoint(configuracao: Mapping[str, Any]) -> dict[str, Any]:
    chaves = (
        "model",
        "base_url",
        "temperature",
        "top_p",
        "top_k",
        "reasoning_effort",
        "header_char_limit",
    )
    return {chave: configuracao.get(chave) for chave in (*chaves, "api_key_env")}


def _manifesto(configuracao: Mapping[str, Any], origem: Path) -> dict[str, Any]:
    return {
        "schema_version": _SCHEMA_VERSION,
        "prompt_version": _PROMPT_VERSION,
        "criado_em": _agora(),
        "origem": str(origem),
        "hash_origem": _hash_arquivo(origem),
        "configuracao_modelo": _json(
            {
                chave: valor
                for chave, valor in configuracao.items()
                if not chave.startswith("_")
            }
        ),
    }


def _checkpoint_reutilizavel(
    caminho: Path,
    documento: Any,
    documento_enriquecido: type[Any],
    configuracao: Mapping[str, Any],
) -> Any | None:
    caminho_auditoria = caminho.with_name("0001.json")
    if not caminho.is_file() or not caminho_auditoria.is_file():
        return None
    try:
        resultado = documento_enriquecido.model_validate_json(
            caminho.read_text(encoding="utf-8")
        )
        auditoria = json.loads(caminho_auditoria.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValidationError, ValueError):
        return None
    if auditoria.get("modelo") != configuracao["model"] or auditoria.get(
        "parametros", {}
    ).get("base_url") != configuracao.get("base_url"):
        return None
    if (
        resultado.documento_id != documento.documento_id
        or resultado.id != documento.id
        or resultado.natureza != documento.natureza
    ):
        return None
    return resultado


def _entrada_agente(
    documento: Any,
    header_char_limit: int,
) -> dict[str, Any]:
    entrada = {
        "documento_id": documento.documento_id,
        "id": documento.id,
        "tribunal": documento.tribunal,
    }
    if documento.natureza == "acordao":
        entrada.update(
            ano=documento.ano,
            texto=documento.texto[:header_char_limit],
        )
    else:
        entrada["texto"] = documento.texto
    return entrada


def _requisicao(
    documento: Any,
    contrato: type[Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    natureza = str(documento.natureza)
    return {
        "model": config["model"],
        "temperature": config["temperature"]
        if config["temperature"] is not None
        else omit,
        "top_p": config["top_p"] if config["top_p"] is not None else omit,
        "reasoning_effort": (
            config["reasoning_effort"]
            if config["reasoning_effort"] is not None
            else omit
        ),
        "messages": [
            {"role": "system", "content": _PROMPTS[natureza]},
            {
                "role": "user",
                "content": json.dumps(
                    _entrada_agente(documento, config["header_char_limit"]),
                    ensure_ascii=False,
                ),
            },
        ],
        "response_format": contrato,
        **(
            {"extra_body": {"top_k": config["top_k"]}}
            if config["top_k"] is not None
            else {}
        ),
    }


class _LimitadorTaxa:
    def __init__(self, intervalo: float) -> None:
        self._intervalo = intervalo
        self._proxima = 0.0
        self._lock = asyncio.Lock()

    async def aguardar(self) -> None:
        async with self._lock:
            agora = time.monotonic()
            espera = max(0.0, self._proxima - agora)
            if espera:
                await asyncio.sleep(espera)
            self._proxima = time.monotonic() + self._intervalo


def _uso(resposta: Any) -> Any:
    usage = getattr(resposta, "usage", None)
    return _json(usage) if usage is not None else None


def _input_auditoria(
    requisicao: Mapping[str, Any], contrato: type[Any]
) -> dict[str, Any]:
    entrada = {
        chave: valor
        for chave, valor in requisicao.items()
        if chave != "response_format" and valor is not omit
    }
    entrada["response_format"] = contrato.model_json_schema()
    return entrada


async def _enriquecer_documento(
    documento: Any,
    *,
    cliente: AsyncOpenAI,
    contrato_para_natureza: Any,
    documento_enriquecido: type[Any],
    metadados_acordao: type[Any],
    metadados_sumula: type[Any],
    config: Mapping[str, Any],
    semaforo: asyncio.Semaphore,
    limitador: _LimitadorTaxa,
    relatores: Mapping[str, str],
) -> tuple[Any, dict[str, Any]]:
    contrato = contrato_para_natureza(documento.natureza)
    requisicao = _requisicao(documento, contrato, config)
    ultimo_erro: Exception | None = None

    for tentativa in range(1, config["max_retries"] + 1):
        try:
            await limitador.aguardar()
            async with semaforo:
                resposta = await cliente.chat.completions.parse(**requisicao)
            campos = resposta.choices[0].message.parsed
            if campos is None:
                raise ValueError("o modelo não retornou uma resposta estruturada")
            if documento.natureza == "acordao":
                dados = campos.model_dump(exclude={"relator_norm"})
                numero, formato = normalizar_numero_processo(
                    campos.numero_processo, documento.texto
                )
                dados.update(numero_processo=numero, formato_numero=formato)
                campos = metadados_acordao(
                    **dados,
                    relator_norm=normalizar_relator(documento.relator, relatores),
                )
            elif documento.natureza == "sumula":
                dados = campos.model_dump()
                dados["sumula_vinculante"] = bool(
                    re.search(
                        r"\bs[uú]mula\s+vinculante\b", documento.texto, re.IGNORECASE
                    )
                )
                campos = metadados_sumula.model_validate(dados)
            resultado = documento_enriquecido(
                documento_id=documento.documento_id,
                id=documento.id,
                natureza=documento.natureza,
                campos=campos,
            )
            return resultado, {
                "tentativa": tentativa,
                "modelo": config["model"],
                "parametros": {
                    chave: valor
                    for chave, valor in config.items()
                    if not chave.startswith("_")
                    and chave
                    not in {
                        "model",
                        "max_concurrency",
                        "max_retries",
                        "retry_delay_seconds",
                        "header_char_limit",
                    }
                    and valor is not None
                },
                "uso": _uso(resposta),
                "input": _input_auditoria(requisicao, contrato),
                "output": resposta.model_dump(
                    mode="json",
                    exclude={"choices": {"__all__": {"message": {"parsed"}}}},
                ),
            }
        except (ValidationError, ValueError, IndexError, AttributeError) as erro:
            ultimo_erro = erro
            requisicao["messages"].append(
                {
                    "role": "user",
                    "content": (
                        "A resposta anterior foi inválida. Corrija sem inventar dados: "
                        f"{str(erro)[:2000]}"
                    ),
                }
            )
        except OpenAIError as erro:
            ultimo_erro = erro

        _LOG.info(
            "documento %s: tentativa %d/%d falhou: %s",
            documento.documento_id,
            tentativa,
            config["max_retries"],
            ultimo_erro,
        )
        if tentativa < config["max_retries"]:
            await asyncio.sleep(config["retry_delay_seconds"] * tentativa)

    assert ultimo_erro is not None
    raise RuntimeError(
        f"{documento.documento_id}: retries esgotados: {ultimo_erro}"
    ) from ultimo_erro


def _cobertura(resultados: Iterable[Any]) -> dict[str, dict[str, int]]:
    totais: dict[str, Counter[str]] = defaultdict(Counter)
    for resultado in resultados:
        for campo, valor in resultado.campos.model_dump(mode="json").items():
            totais[resultado.natureza][campo] += valor is not None
            totais[resultado.natureza][f"{campo}_nulo"] += valor is None
    return {natureza: dict(contagens) for natureza, contagens in totais.items()}


def _diagnostico(valor: Any) -> Any:
    return _json(valor) if valor is not None else None


def _diploma_gold(valor: str) -> tuple[str, str | None, int | None]:
    nomes = {
        "CF": "Constituição Federal",
        "CLT": "Consolidação das Leis do Trabalho",
        "CPC": "Código de Processo Civil",
        "CDC": "Código de Defesa do Consumidor",
        "LC": "Lei Complementar",
        "Codigo Eleitoral": "Código Eleitoral",
        "CPM": "Código Penal Militar",
        "CC": "Código Civil",
        "CPP": "Código de Processo Penal",
    }
    sigla = valor.split("/")[0].split(" (")[0]
    if valor.startswith("Lei "):
        sigla = "Lei"
    numeros = re.search(r"(\d[\d.]*)/(\d{4})", valor)
    numero = re.sub(r"\D", "", numeros.group(1)) if numeros and sigla != "CF" else None
    ano = int(numeros.group(2)) if numeros else None
    return nomes.get(sigla, sigla), numero, ano


def _avaliar_gold(caminho: Path, resultados: Iterable[Any]) -> dict[str, Any] | None:
    if not caminho.is_file():
        return None
    por_id = {str(resultado.id): resultado for resultado in resultados}
    contagens: dict[str, Counter[str]] = defaultdict(Counter)
    with caminho.open(encoding="utf-8", newline="") as arquivo:
        linhas = csv.DictReader(arquivo)
        for linha in linhas:
            if linha["classificacao"] != "real" or not linha["id_canonico"]:
                continue
            resultado = por_id.get(linha["id_canonico"])
            if resultado is None:
                continue
            campos = resultado.campos.model_dump(mode="json")
            esperados: dict[str, Any] = {}
            if linha["natureza"] == "acordao":
                numero = re.sub(r"\D", "", linha["numero_do_processo"])
                cadeia = linha["cadeia_recursal"].split(">")
                esperados = {
                    "numero_processo": numero,
                    "formato_numero": linha["formato_do_numero"],
                    "classe_processual": cadeia[-1],
                    "cadeia_recursal": cadeia,
                    "uf": linha["uf"] or None,
                }
            elif linha["natureza"] == "sumula":
                sumula = linha["numero_da_sumula"]
                esperados = {
                    "numero_sumula": int(sumula.split()[0]),
                    "sumula_vinculante": "vinculante" in sumula,
                }
            else:
                diploma, numero, ano = _diploma_gold(linha["diploma_legal"])
                esperados = {
                    "diploma": diploma,
                    "numero_diploma": numero,
                    "ano_diploma": ano,
                    "numero_artigo": linha["numero_do_artigo"],
                }
            for campo, esperado in esperados.items():
                contagens[resultado.natureza][f"{campo}_total"] += 1
                obtido = campos.get(campo)
                correto = (
                    Counter(obtido or []) == Counter(esperado or [])
                    if campo == "cadeia_recursal"
                    else obtido == esperado
                )
                contagens[resultado.natureza][f"{campo}_corretos"] += correto

    avaliacao = {}
    for natureza, valores in contagens.items():
        campos = {}
        for chave, total in valores.items():
            if not chave.endswith("_total"):
                continue
            campo = chave.removesuffix("_total")
            corretos = valores[f"{campo}_corretos"]
            campos[campo] = {
                "corretos": corretos,
                "total": total,
                "acuracia": corretos / total if total else 0.0,
            }
        avaliacao[natureza] = campos
    return avaliacao


async def executar_async(
    *,
    origem: Path,
    destino: Path,
    diretorio_auditoria: Path,
    configuracao: Mapping[str, Any],
    force: bool = False,
) -> dict[str, Any]:
    """Enriquece checkpoints e materializa a cópia SQLite somente após validação."""
    # Estes contratos pertencem ao módulo especializado e não ao pipeline de citações.
    from .contracts import (
        DocumentoEnriquecido,
        MetadadosAcordao,
        MetadadosSumula,
        contrato_para_natureza,
    )
    from .database import listar_documentos, materializar_banco

    config = _configuracao(configuracao)
    diretorio_auditoria.mkdir(parents=True, exist_ok=True)
    manifesto_anterior: dict[str, Any] = {}
    caminho_manifesto = diretorio_auditoria / "manifest.json"
    if caminho_manifesto.is_file():
        try:
            manifesto_anterior = json.loads(
                caminho_manifesto.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError):
            pass
    manifesto_atual = _manifesto(config, origem)
    mesmo_checkpoint = all(
        manifesto_anterior.get(chave) == manifesto_atual[chave]
        for chave in ("prompt_version", "hash_origem")
    ) and _config_checkpoint(manifesto_anterior.get("configuracao_modelo", {})) == (
        _config_checkpoint(manifesto_atual["configuracao_modelo"])
    )
    _escrever_json(caminho_manifesto, manifesto_atual)

    caminhos_relatores = (
        origem.parent / "relatores_padronizacao.json",
        destino.parent / "relatores_padronizacao.json",
    )
    caminho_relatores = next(
        (caminho for caminho in caminhos_relatores if caminho.is_file()),
        caminhos_relatores[0],
    )
    with caminho_relatores.open(encoding="utf-8") as arquivo:
        relatores = json.load(arquivo)
    if not isinstance(relatores, dict) or not all(
        isinstance(chave, str) and isinstance(valor, str)
        for chave, valor in relatores.items()
    ):
        raise ValueError(f"dicionário de relatores inválido: {caminho_relatores}")

    documentos = listar_documentos(origem)
    resultados: list[Any] = []
    pendentes: list[Any] = []
    reutilizados = 0
    for documento in documentos:
        checkpoint = (
            diretorio_auditoria
            / "documentos"
            / documento.documento_id
            / "resultado.json"
        )
        resultado = (
            _checkpoint_reutilizavel(
                checkpoint, documento, DocumentoEnriquecido, config
            )
            if mesmo_checkpoint
            else None
        )
        if resultado is None:
            pendentes.append(documento)
        else:
            if documento.natureza == "acordao":
                campos = resultado.campos.model_dump(exclude={"relator_norm"})
                numero, formato = normalizar_numero_processo(
                    resultado.campos.numero_processo, documento.texto
                )
                campos.update(numero_processo=numero, formato_numero=formato)
                resultado = resultado.model_copy(
                    update={
                        "campos": MetadadosAcordao(
                            **campos,
                            relator_norm=normalizar_relator(
                                documento.relator, relatores
                            ),
                        )
                    }
                )
                _escrever_json(checkpoint, resultado)
            elif documento.natureza == "sumula":
                campos = resultado.campos.model_dump()
                campos["sumula_vinculante"] = bool(
                    re.search(
                        r"\bs[uú]mula\s+vinculante\b", documento.texto, re.IGNORECASE
                    )
                )
                resultado = resultado.model_copy(
                    update={"campos": MetadadosSumula.model_validate(campos)}
                )
                _escrever_json(checkpoint, resultado)
            resultados.append(resultado)
            reutilizados += 1

    revisoes: list[dict[str, Any]] = []
    semaforo = asyncio.Semaphore(config["max_concurrency"])
    limitador = _LimitadorTaxa(config["request_interval_seconds"])

    async def processar(documento: Any, cliente: AsyncOpenAI) -> tuple[Any, Any]:
        try:
            resposta = await _enriquecer_documento(
                documento,
                cliente=cliente,
                contrato_para_natureza=contrato_para_natureza,
                documento_enriquecido=DocumentoEnriquecido,
                metadados_acordao=MetadadosAcordao,
                metadados_sumula=MetadadosSumula,
                config=config,
                semaforo=semaforo,
                limitador=limitador,
                relatores=relatores,
            )
        except RuntimeError as erro:
            resposta = erro
        return documento, resposta

    async with AsyncOpenAI(
        base_url=config["base_url"], api_key=config["_api_key"], max_retries=0
    ) as cliente:
        tarefas = [
            asyncio.create_task(processar(documento, cliente))
            for documento in pendentes
        ]
        with tqdm(
            total=len(documentos),
            initial=reutilizados,
            desc="Pré-processando banco",
            unit="documento",
        ) as progresso:
            for tarefa in asyncio.as_completed(tarefas):
                documento, resposta = await tarefa
                if isinstance(resposta, BaseException):
                    revisoes.append(
                        {
                            "documento_id": documento.documento_id,
                            "id": documento.id,
                            "natureza": documento.natureza,
                            "motivo": str(resposta),
                        }
                    )
                else:
                    resultado, auditoria = resposta
                    pasta = diretorio_auditoria / "documentos" / documento.documento_id
                    _escrever_json(pasta / "resultado.json", resultado)
                    _escrever_json(pasta / "0001.json", auditoria)
                    resultados.append(resultado)
                progresso.update()

    with (diretorio_auditoria / "revisao.jsonl").open("w", encoding="utf-8") as arquivo:
        for revisao in revisoes:
            arquivo.write(json.dumps(revisao, ensure_ascii=False) + "\n")

    resultados.sort(key=lambda resultado: resultado.id)
    relatorio: dict[str, Any] = {
        "gerado_em": _agora(),
        "origem": {"caminho": str(origem), "sha256": _hash_arquivo(origem)},
        "destino": {"caminho": str(destino)},
        "modelo": config["model"],
        "prompt_version": _PROMPT_VERSION,
        "schema_version": _SCHEMA_VERSION,
        "total_documentos": len(documentos),
        "processados": len(pendentes) - len(revisoes),
        "reutilizados": reutilizados,
        "invalidos": len(revisoes),
        "enviados_para_revisao": len(revisoes),
        "cobertura_por_natureza": _cobertura(resultados),
        "avaliacao_gold": _avaliar_gold(
            origem.parent / "goldenset_offsets_anotado_gustavo.csv", resultados
        ),
        "materializacao": None,
    }
    if revisoes:
        _escrever_json(diretorio_auditoria / "relatorio.json", relatorio)
        raise RuntimeError(
            f"{len(revisoes)} documentos exigem revisão; banco não materializado"
        )

    diagnostico = materializar_banco(origem, destino, resultados, force=force)
    relatorio["materializacao"] = _diagnostico(diagnostico)
    if destino.is_file():
        relatorio["destino"]["sha256"] = _hash_arquivo(destino)
    _escrever_json(diretorio_auditoria / "relatorio.json", relatorio)
    return relatorio


def executar(
    *,
    origem: Path,
    destino: Path,
    diretorio_auditoria: Path,
    configuracao: Mapping[str, Any],
    force: bool = False,
) -> dict[str, Any]:
    """Executa o pipeline assíncrono a partir de uma CLI síncrona."""
    return asyncio.run(
        executar_async(
            origem=origem,
            destino=destino,
            diretorio_auditoria=diretorio_auditoria,
            configuracao=configuracao,
            force=force,
        )
    )
