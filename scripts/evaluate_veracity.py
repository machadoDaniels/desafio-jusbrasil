# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "numpy>=2.0",
#   "pandas>=2.0",
#   "pydantic>=2.0",
#   "pyyaml>=6.0",
# ]
# ///

"""Avalia classes e IDs canônicos produzidos pela etapa de veracidade."""

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "desafio-jusbrasil-bracis-2026"), str(ROOT / "src")]

from kaggle_metric import _casar

from desafio_jusbrasil.contracts import Classificacao, DocumentoClassificado
from desafio_jusbrasil.utils import listar_resultados


def _pasta_etapa(pasta: Path) -> Path:
    etapa = pasta / "04-veracity"
    return etapa if etapa.is_dir() else pasta


def _ler(pasta: Path) -> dict[str, DocumentoClassificado]:
    pasta = _pasta_etapa(pasta)
    return {
        documento.documento_id: documento
        for arquivo in listar_resultados(pasta)
        for documento in [
            DocumentoClassificado.model_validate_json(
                arquivo.read_text(encoding="utf-8")
            )
        ]
    }


def _dividir(a: int, b: int) -> float:
    return a / b if b else 0.0


def _novo_acumulador() -> dict[str, int]:
    return {
        "spans_casados": 0,
        "corretos": 0,
        "preditos": 0,
        "golds": 0,
        "classes_corretas": 0,
        "ids_corretos": 0,
        "reais_preditos_real": 0,
    }


def _nivel(documento_id: str) -> str:
    encontrado = re.search(r"_n(\d+)_", documento_id)
    return encontrado.group(1) if encontrado else "desconhecido"


def _calcular_metricas(acumulador: dict[str, int]) -> dict[str, int | float]:
    precisao = _dividir(acumulador["corretos"], acumulador["preditos"])
    recall = _dividir(acumulador["corretos"], acumulador["golds"])
    return {
        "spans_casados": acumulador["spans_casados"],
        "predicoes_totalmente_corretas": acumulador["corretos"],
        "preditos": acumulador["preditos"],
        "golds": acumulador["golds"],
        "acuracia_classe_nos_pares": _dividir(
            acumulador["classes_corretas"], acumulador["spans_casados"]
        ),
        "acuracia_id_reais": _dividir(
            acumulador["ids_corretos"], acumulador["reais_preditos_real"]
        ),
        "precisao_end_to_end": precisao,
        "recall_end_to_end": recall,
        "f1_end_to_end": (
            2 * precisao * recall / (precisao + recall) if precisao + recall else 0.0
        ),
    }


def _finalizar_grupos(
    grupos: dict[str, dict[str, int]],
) -> dict[str, dict[str, int | float]]:
    return {chave: _calcular_metricas(valor) for chave, valor in grupos.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "predicoes",
        nargs="?",
        type=Path,
        default=Path("outputs/run-001/04-veracity"),
    )
    parser.add_argument(
        "--gold",
        type=Path,
        default=Path("outputs/gold/04-veracity"),
    )
    args = parser.parse_args()

    golds = _ler(args.gold)
    preditos = _ler(args.predicoes)
    pasta_predicoes = _pasta_etapa(args.predicoes)
    casados = classes_corretas = corretos = ids_corretos = reais_preditos_real = 0
    total_gold = total_predito = 0
    documentos_incorretos = []
    metricas_por_nivel: dict[str, dict[str, int]] = {}
    metricas_por_tipo: dict[str, dict[str, int]] = {}

    for documento_id, gold in golds.items():
        pred = preditos.get(documento_id)
        itens = pred.candidatos if pred else []
        localizados = [
            item
            for item in itens
            if item.candidato.inicio is not None and item.candidato.fim is not None
        ]
        pares, golds_sem_par, preditos_sem_par = _casar(
            [item.candidato.model_dump() for item in gold.candidatos],
            [item.candidato.model_dump() for item in localizados],
        )
        nivel = _nivel(documento_id)
        acumulador_nivel = metricas_por_nivel.setdefault(nivel, _novo_acumulador())
        acumulador_nivel["golds"] += len(gold.candidatos)
        acumulador_nivel["preditos"] += len(localizados)
        for item in gold.candidatos:
            metricas_por_tipo.setdefault(item.candidato.tipo.value, _novo_acumulador())[
                "golds"
            ] += 1
        for item in localizados:
            metricas_por_tipo.setdefault(item.candidato.tipo.value, _novo_acumulador())[
                "preditos"
            ] += 1
        avaliacao_pares = []
        casados += len(pares)
        total_gold += len(gold.candidatos)
        total_predito += len(localizados)
        for gi, pi in pares:
            esperado = gold.candidatos[gi].veracidade
            obtido = localizados[pi].veracidade
            classe_correta = esperado.classificacao == obtido.classificacao
            classes_corretas += classe_correta
            if classe_correta and esperado.classificacao == Classificacao.REAL:
                reais_preditos_real += 1
                ids_corretos += esperado.id_canonico == obtido.id_canonico
            id_correto = (
                esperado.classificacao != Classificacao.REAL
                or esperado.id_canonico == obtido.id_canonico
            )
            correto = classe_correta and id_correto
            corretos += correto
            acumulador_tipo = metricas_por_tipo[
                gold.candidatos[gi].candidato.tipo.value
            ]
            for acumulador in (acumulador_nivel, acumulador_tipo):
                acumulador["spans_casados"] += 1
                acumulador["classes_corretas"] += classe_correta
                acumulador["corretos"] += correto
                if classe_correta and esperado.classificacao == Classificacao.REAL:
                    acumulador["reais_preditos_real"] += 1
                    acumulador["ids_corretos"] += (
                        esperado.id_canonico == obtido.id_canonico
                    )
            avaliacao_pares.append(
                {
                    "correto": correto,
                    "motivo": (
                        "correto"
                        if correto
                        else "classe_incorreta"
                        if not classe_correta
                        else "id_canonico_incorreto"
                    ),
                    "esperado": gold.candidatos[gi].model_dump(mode="json"),
                    "obtido": localizados[pi].model_dump(mode="json"),
                }
            )

        avaliacao_documento = {
            "documento_id": documento_id,
            "correto": (
                all(item["correto"] for item in avaliacao_pares)
                and not golds_sem_par
                and not preditos_sem_par
            ),
            "resumo": {
                "pares": len(pares),
                "pares_corretos": sum(item["correto"] for item in avaliacao_pares),
                "golds_nao_encontrados": len(golds_sem_par),
                "predicoes_extras": len(preditos_sem_par),
            },
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
        if not avaliacao_documento["correto"]:
            documentos_incorretos.append(
                {
                    "documento_id": documento_id,
                    **avaliacao_documento["resumo"],
                }
            )
        if pred is not None:
            pasta_documento = pasta_predicoes / documento_id
            pasta_documento.mkdir(parents=True, exist_ok=True)
            (pasta_documento / "resultado_eval.json").write_text(
                json.dumps(avaliacao_documento, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )

    precisao = _dividir(corretos, total_predito)
    recall = _dividir(corretos, total_gold)
    f1 = 2 * precisao * recall / (precisao + recall) if precisao + recall else 0.0
    metricas = {
        "spans_casados": casados,
        "predicoes_totalmente_corretas": corretos,
        "preditos": total_predito,
        "golds": total_gold,
        "acuracia_classe_nos_pares": _dividir(classes_corretas, casados),
        "acuracia_id_reais": _dividir(ids_corretos, reais_preditos_real),
        "precisao_end_to_end": precisao,
        "recall_end_to_end": recall,
        "f1_end_to_end": f1,
        "metricas_por_nivel": _finalizar_grupos(metricas_por_nivel),
        "metricas_por_tipo": _finalizar_grupos(metricas_por_tipo),
    }
    relatorio = {
        **metricas,
        "documentos_incorretos": documentos_incorretos,
    }
    (pasta_predicoes / "avaliacao.json").write_text(
        json.dumps(relatorio, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    principais = {
        chave: valor
        for chave, valor in relatorio.items()
        if not isinstance(valor, dict | list)
    }
    print(json.dumps(principais, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
