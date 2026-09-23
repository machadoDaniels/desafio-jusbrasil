# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "numpy>=2.0",
#   "pandas>=2.0",
#   "pydantic>=2.0",
#   "pyyaml>=6.0",
# ]
# ///

"""Avalia as entidades estruturadas produzidas pela etapa 03-entities."""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "desafio-jusbrasil-bracis-2026"), str(ROOT / "src")]

from kaggle_metric import _casar

from desafio_jusbrasil.contracts import (
    ConsultaJurisprudencia,
    ConsultaLegislacao,
    DocumentoEntidades,
)
from desafio_jusbrasil.utils import listar_resultados

_CAMPOS = {
    "jurisprudencia": (
        "natureza",
        "numero_processo_cnj",
        "sequencias_numericas_identificadoras",
        "classe_processual",
        "cadeia_recursal",
        "tribunal",
        "uf",
        "ano",
        "relator",
        "numero_sumula",
        "sumula_vinculante",
    ),
    "lei": (
        "numero_artigo",
        "diploma",
        "numero_diploma",
        "ano_diploma",
    ),
}


def _pasta_etapa(pasta: Path) -> Path:
    etapa = pasta / "03-entities"
    return etapa if etapa.is_dir() else pasta


def _ler(pasta: Path) -> dict[str, DocumentoEntidades]:
    pasta = _pasta_etapa(pasta)
    return {
        documento.documento_id: documento
        for arquivo in listar_resultados(pasta)
        for documento in [
            DocumentoEntidades.model_validate_json(arquivo.read_text(encoding="utf-8"))
        ]
    }


def _dividir(a: int, b: int) -> float:
    return a / b if b else 0.0


def _nivel(documento_id: str) -> str:
    encontrado = re.search(r"_n(\d+)_", documento_id)
    return encontrado.group(1) if encontrado else "desconhecido"


def _valor_comparavel(campo: str, valor: Any) -> Any:
    if campo == "cadeia_recursal" and valor is not None:
        return frozenset(valor)
    return valor


def _comparar_campos(
    esperado: ConsultaJurisprudencia | ConsultaLegislacao,
    obtido: ConsultaJurisprudencia | ConsultaLegislacao | None,
    tipo: str,
) -> tuple[list[str], dict[str, tuple[Any, Any]]]:
    divergentes = []
    valores = {}
    for campo in _CAMPOS[tipo]:
        valor_esperado = getattr(esperado, campo)
        valor_obtido = getattr(obtido, campo, None) if obtido is not None else None
        valores[campo] = (valor_esperado, valor_obtido)
        if _valor_comparavel(campo, valor_esperado) != _valor_comparavel(
            campo, valor_obtido
        ):
            divergentes.append(campo)
    return divergentes, valores


def _novo_acumulador() -> dict[str, int]:
    return {
        "spans_casados": 0,
        "consultas_exatas": 0,
        "consultas_gold": 0,
        "consultas_preditas": 0,
    }


def _metricas_consultas(acumulador: dict[str, int]) -> dict[str, int | float]:
    precisao = _dividir(
        acumulador["consultas_exatas"], acumulador["consultas_preditas"]
    )
    recall = _dividir(acumulador["consultas_exatas"], acumulador["consultas_gold"])
    return {
        **acumulador,
        "precisao_consulta_exata": precisao,
        "recall_consulta_exata": recall,
        "f1_consulta_exata": (
            2 * precisao * recall / (precisao + recall) if precisao + recall else 0.0
        ),
    }


def _metricas_campos(
    contagens: dict[str, Counter[str]],
) -> dict[str, dict[str, int | float]]:
    resultado = {}
    for campo, valores in sorted(contagens.items()):
        resultado[campo] = {
            "corretos": valores["corretos"],
            "total": valores["total"],
            "acuracia": _dividir(valores["corretos"], valores["total"]),
            "valores_corretos": valores["valores_corretos"],
            "valores_esperados": valores["valores_esperados"],
            "valores_preditos": valores["valores_preditos"],
            "precisao_valores": _dividir(
                valores["valores_corretos"], valores["valores_preditos"]
            ),
            "recall_valores": _dividir(
                valores["valores_corretos"], valores["valores_esperados"]
            ),
        }
    return resultado


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "predicoes",
        nargs="?",
        type=Path,
        default=Path("outputs/run-001/03-entities"),
    )
    parser.add_argument(
        "--gold",
        type=Path,
        default=Path("outputs/gold/03-entities"),
    )
    args = parser.parse_args()

    golds = _ler(args.gold)
    preditos = _ler(args.predicoes)
    pasta_predicoes = _pasta_etapa(args.predicoes)
    global_ = _novo_acumulador()
    por_tipo: dict[str, dict[str, int]] = {}
    por_nivel: dict[str, dict[str, int]] = {}
    campos: dict[str, Counter[str]] = {}
    documentos_incorretos = []

    for documento_id, gold in golds.items():
        pred = preditos.get(documento_id)
        itens_preditos = pred.candidatos if pred else []
        localizados = [
            item
            for item in itens_preditos
            if item.candidato.inicio is not None and item.candidato.fim is not None
        ]
        pares, golds_sem_par, preditos_sem_par = _casar(
            [item.candidato.model_dump() for item in gold.candidatos],
            [item.candidato.model_dump() for item in localizados],
        )
        acumulador_nivel = por_nivel.setdefault(
            _nivel(documento_id), _novo_acumulador()
        )
        global_["spans_casados"] += len(pares)
        acumulador_nivel["spans_casados"] += len(pares)

        for item in gold.candidatos:
            if item.campos_extraidos is None:
                continue
            tipo = item.candidato.tipo.value
            global_["consultas_gold"] += 1
            acumulador_nivel["consultas_gold"] += 1
            por_tipo.setdefault(tipo, _novo_acumulador())["consultas_gold"] += 1
        for item in localizados:
            if item.campos_extraidos is None:
                continue
            tipo = item.candidato.tipo.value
            global_["consultas_preditas"] += 1
            acumulador_nivel["consultas_preditas"] += 1
            por_tipo.setdefault(tipo, _novo_acumulador())["consultas_preditas"] += 1

        avaliacao_pares = []
        for gi, pi in pares:
            esperado = gold.candidatos[gi]
            obtido = localizados[pi]
            tipo = esperado.candidato.tipo.value
            acumulador_tipo = por_tipo.setdefault(tipo, _novo_acumulador())
            acumulador_tipo["spans_casados"] += 1
            divergentes = []
            valores: dict[str, tuple[Any, Any]] = {}
            if esperado.campos_extraidos is None:
                correto = obtido.campos_extraidos is None
                if not correto:
                    divergentes = ["campos_extraidos"]
            else:
                divergentes, valores = _comparar_campos(
                    esperado.campos_extraidos,
                    obtido.campos_extraidos,
                    tipo,
                )
                correto = not divergentes
                for campo, (valor_esperado, valor_obtido) in valores.items():
                    chave = f"{tipo}.{campo}"
                    contador = campos.setdefault(chave, Counter())
                    contador["total"] += 1
                    igual = _valor_comparavel(
                        campo, valor_esperado
                    ) == _valor_comparavel(campo, valor_obtido)
                    contador["corretos"] += igual
                    esperado_preenchido = valor_esperado is not None
                    obtido_preenchido = valor_obtido is not None
                    contador["valores_esperados"] += esperado_preenchido
                    contador["valores_preditos"] += obtido_preenchido
                    contador["valores_corretos"] += (
                        esperado_preenchido and obtido_preenchido and igual
                    )
                if correto:
                    for acumulador in (global_, acumulador_nivel, acumulador_tipo):
                        acumulador["consultas_exatas"] += 1
            avaliacao_pares.append(
                {
                    "correto": correto,
                    "campos_incorretos": divergentes,
                    "esperado": esperado.model_dump(mode="json"),
                    "obtido": obtido.model_dump(mode="json"),
                }
            )

        correto_documento = (
            all(item["correto"] for item in avaliacao_pares)
            and not golds_sem_par
            and not preditos_sem_par
        )
        resumo = {
            "pares": len(pares),
            "pares_corretos": sum(item["correto"] for item in avaliacao_pares),
            "golds_nao_encontrados": len(golds_sem_par),
            "predicoes_extras": len(preditos_sem_par),
        }
        avaliacao_documento = {
            "documento_id": documento_id,
            "correto": correto_documento,
            "resumo": resumo,
            "pares": avaliacao_pares,
            "golds_nao_encontrados": [
                gold.candidatos[indice].model_dump(mode="json")
                for indice in golds_sem_par
            ],
            "predicoes_extras": [
                localizados[indice].model_dump(mode="json")
                for indice in preditos_sem_par
            ],
        }
        if not correto_documento:
            documentos_incorretos.append({"documento_id": documento_id, **resumo})
        if pred is not None:
            pasta_documento = pasta_predicoes / documento_id
            pasta_documento.mkdir(parents=True, exist_ok=True)
            (pasta_documento / "resultado_eval.json").write_text(
                json.dumps(avaliacao_documento, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )

    metricas = {
        **_metricas_consultas(global_),
        "metricas_por_campo": _metricas_campos(campos),
        "metricas_por_nivel": {
            chave: _metricas_consultas(valor)
            for chave, valor in sorted(por_nivel.items())
        },
        "metricas_por_tipo": {
            chave: _metricas_consultas(valor)
            for chave, valor in sorted(por_tipo.items())
        },
    }
    relatorio = {**metricas, "documentos_incorretos": documentos_incorretos}
    pasta_predicoes.mkdir(parents=True, exist_ok=True)
    (pasta_predicoes / "avaliacao.json").write_text(
        json.dumps(relatorio, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metricas, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
