# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "pydantic>=2.0",
# ]
# ///

"""Localiza no texto original os trechos literais extraídos pelo pré-processamento.

Requer uma run feita com ``extrair_verbatim: true``. Grava ``spans.jsonl`` na pasta da
run, com um span por campo (offsets em caracteres do texto original).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from desafio_jusbrasil.database_preprocessing import DocumentoEnriquecido
from desafio_jusbrasil.database_preprocessing.database import listar_documentos


def localizar(texto: str, trecho: str) -> tuple[int, int, str] | None:
    """Retorna (início, fim, método) da primeira ocorrência do trecho no texto."""
    inicio = texto.find(trecho)
    if inicio >= 0:
        return inicio, inicio + len(trecho), "exato"
    tokens = trecho.split()
    if not tokens:
        return None
    padrao = r"\s*".join(re.escape(token) for token in tokens)
    encontrado = re.search(padrao, texto, re.IGNORECASE)
    if encontrado:
        return encontrado.start(), encontrado.end(), "regex"
    return None


def _na_janela(inicio: int, fim: int, tamanho: int, limites: dict) -> bool:
    """Indica se o span está no trecho do texto que foi enviado ao modelo."""
    comeco, final = limites.get("text_start_char_limit"), limites.get("text_end_char_limit")
    if comeco is None and final is None:
        return True
    if comeco is not None and final is not None and tamanho <= comeco + final:
        return True
    return (comeco is not None and fim <= comeco) or (
        final is not None and inicio >= tamanho - final
    )


def _percentil(valores: list[float], q: float) -> float:
    ordenados = sorted(valores)
    return ordenados[min(len(ordenados) - 1, int(q * len(ordenados)))]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predicoes", type=Path)
    parser.add_argument(
        "--origem",
        type=Path,
        default=Path("desafio-jusbrasil-bracis-2026/desafio1_bracis.db"),
    )
    args = parser.parse_args()
    textos = {documento.documento_id: documento.texto for documento in listar_documentos(args.origem)}
    manifest = args.predicoes / "manifest.json"
    limites = (
        json.loads(manifest.read_text(encoding="utf-8"))["configuracao_modelo"]
        if manifest.is_file()
        else {}
    )

    spans = []
    for arquivo in sorted((args.predicoes / "documentos").glob("*/resultado.json")):
        resultado = DocumentoEnriquecido.model_validate_json(arquivo.read_text(encoding="utf-8"))
        if not resultado.trechos:
            continue
        texto = textos[resultado.documento_id]
        valores = resultado.campos.model_dump(mode="json")
        for campo, trechos in resultado.trechos.items():
            for trecho in trechos if isinstance(trechos, list) else [trechos]:
                if trecho is None:
                    continue
                posicao = localizar(texto, trecho)
                spans.append(
                    {
                        "documento_id": resultado.documento_id,
                        "natureza": resultado.natureza,
                        "campo": campo,
                        "valor": valores.get(campo),
                        "trecho": trecho,
                        "inicio": posicao[0] if posicao else None,
                        "fim": posicao[1] if posicao else None,
                        "tamanho_texto": len(texto),
                        "metodo": posicao[2] if posicao else "nao_encontrado",
                        "na_janela_do_modelo": (
                            _na_janela(posicao[0], posicao[1], len(texto), limites)
                            if posicao
                            else None
                        ),
                    }
                )
    if not spans:
        raise ValueError(f"nenhum trecho em {args.predicoes}; rode com extrair_verbatim: true")

    with (args.predicoes / "spans.jsonl").open("w", encoding="utf-8") as arquivo:
        for span in spans:
            arquivo.write(json.dumps(span, ensure_ascii=False) + "\n")

    totais: Counter[str] = Counter()
    encontrados: Counter[str] = Counter()
    for span in spans:
        chave = f"{span['natureza']}.{span['campo']}"
        totais[chave] += 1
        encontrados[chave] += span["metodo"] != "nao_encontrado"
    posicoes: dict[str, list[dict]] = {}
    for span in spans:
        if span["inicio"] is not None:
            posicoes.setdefault(f"{span['natureza']}.{span['campo']}", []).append(span)
    principais = {
        "trechos": len(spans),
        "encontrados": sum(encontrados.values()),
        "taxa_encontrados": sum(encontrados.values()) / len(spans),
        "por_metodo": dict(Counter(span["metodo"] for span in spans)),
        "taxa_encontrados_por_campo": {
            chave: encontrados[chave] / totais[chave] for chave in sorted(totais)
        },
        "posicao_por_campo": {
            chave: {
                "inicio_p50": _percentil([s["inicio"] for s in itens], 0.5),
                "inicio_p90": _percentil([s["inicio"] for s in itens], 0.9),
                "inicio_max": max(s["inicio"] for s in itens),
                "distancia_fim_p50": _percentil(
                    [s["tamanho_texto"] - s["fim"] for s in itens], 0.5
                ),
                "posicao_relativa_p50": round(
                    _percentil([s["inicio"] / s["tamanho_texto"] for s in itens], 0.5), 3
                ),
                "taxa_na_janela_do_modelo": sum(
                    bool(s["na_janela_do_modelo"]) for s in itens
                )
                / len(itens),
            }
            for chave, itens in sorted(posicoes.items())
        },
    }
    print(json.dumps(principais, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
