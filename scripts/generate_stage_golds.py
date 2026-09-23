# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "pandas>=2.0",
#   "pydantic>=2.0",
#   "pyyaml>=6.0",
# ]
# ///

"""Gera os checkpoints gold de cada etapa a partir do goldenset oficial."""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from desafio_jusbrasil.contracts import (
    CandidatoAnalisado,
    CandidatoCitacao,
    CandidatoClassificado,
    CandidatoEntidades,
    Classificacao,
    ConsultaJurisprudencia,
    ConsultaLegislacao,
    DocumentoClassificado,
    DocumentoCompletude,
    DocumentoEntidades,
    DocumentoExtraido,
    ResultadoCompletude,
    ResultadoVeracidade,
    TipoCitacao,
)
from desafio_jusbrasil.utils import normalizar_numero_cnj, normalizar_relator

ETAPAS = ("01-extraction", "02-completeness", "03-entities", "04-veracity")

# Correções da revisão manual de 26 documentos / 192 citações. Cada valor foi conferido
# contra o trecho e o TXT completo conforme docs/gold-rules.md.
_OVERRIDES_ENTIDADES = {
    ("gen_n1_001", "g9"): {"relator_norm": "marco antônio de farias"},
    ("gen_n1_001", "g2"): {"relator_norm": "josé antônio dias toffoli"},
    ("gen_n1_006", "g6"): {"classe_processual": None, "cadeia_recursal": None},
    ("gen_n1_008", "g4"): {"uf": None},
    ("gen_n1_008", "g3"): {"uf": None},
    ("gen_n1_012", "g6"): {"uf": None},
    ("gen_n1_012", "g1"): {"uf": None},
    ("gen_n2_001", "g6"): {"relator_norm": "sérgio kukina"},
    ("gen_n2_003", "g2"): {"uf": None},
    ("gen_n2_005", "g8"): {"uf": None},
    ("gen_n2_006", "g7"): {"numero_classe_tribunal": "44921"},
    ("gen_n2_008", "g2"): {"numero_classe_tribunal": "2467648"},
    ("gen_n2_009", "g5"): {
        "classe_processual": "AREsp",
        "cadeia_recursal": ["AREsp"],
    },
    ("gen_n2_009", "g6"): {"numero_classe_tribunal": "2173718"},
    ("gen_n2_011", "g5"): {"numero_classe_tribunal": None},
    ("gen_n2_011", "g7"): {"numero_classe_tribunal": "1700760"},
    ("gen_n2_012", "g3"): {"relator_norm": "sebastião reis júnior"},
    ("gen_n2_012", "g8"): {"numero_classe_tribunal": None},
    ("gen_n2_012", "g1"): {"numero_classe_tribunal": "1528455"},
}

_TRIBUNAIS = {
    "STF": (r"\bSTF\b", r"Supremo Tribunal Federal"),
    "STJ": (r"\bSTJ\b", r"Superior Tribunal de Justiça"),
    "STM": (r"\bSTM\b", r"Superior Tribunal Militar"),
    "TSE": (r"\bTSE\b", r"Tribunal Superior Eleitoral"),
    "TST": (r"\bTST\b", r"Tribunal Superior do Trabalho"),
}


def _salvar(
    destino: Path,
    documento: (
        DocumentoExtraido
        | DocumentoCompletude
        | DocumentoEntidades
        | DocumentoClassificado
    ),
) -> None:
    pasta = destino / documento.documento_id
    pasta.mkdir(parents=True, exist_ok=True)
    (pasta / "resultado.json").write_text(
        documento.model_dump_json(indent=2, exclude_none=True) + "\n",
        encoding="utf-8",
    )


def _opcional(valor: str) -> str | None:
    valor = valor.strip()
    return valor or None


def _tribunal_explicito(valor: str, trecho: str) -> str | None:
    tribunal = _opcional(valor)
    if tribunal is None:
        return None
    return (
        tribunal
        if any(
            re.search(padrao, trecho, re.IGNORECASE) for padrao in _TRIBUNAIS[tribunal]
        )
        else None
    )


def _diploma(valor: str, trecho: str) -> tuple[str | None, str | None]:
    if not valor:
        return None, None
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
    for prefixo in (
        "CF",
        "CPC",
        "CLT",
        "CDC",
        "LC",
        "Codigo Eleitoral",
        "CPM",
        "CC",
        "CPP",
    ):
        if valor.startswith(prefixo):
            sigla = prefixo
            break
    if valor.startswith("Lei "):
        sigla = "Lei"
    nome = nomes.get(sigla, sigla)
    numeros = re.search(r"(\d[\d.]*)\s*/\s*(\d{4})", trecho)
    numero = re.sub(r"\D", "", numeros.group(1)) if numeros else None
    return nome, numero


def _campos_entidades(anotacao: dict[str, str], trecho: str, relatores: dict[str, str]):
    if anotacao["tipo"] == "jurisprudencia":
        sumula = _opcional(anotacao["numero_da_sumula"])
        numero_bruto = _opcional(anotacao["numero_do_processo"])
        numero_cnj = normalizar_numero_cnj(numero_bruto, trecho)
        numero_classe = (
            None if numero_cnj else re.sub(r"\D", "", numero_bruto or "") or None
        )
        cadeia = (
            anotacao["cadeia_recursal"].split(">")
            if anotacao["cadeia_recursal"]
            else None
        )
        return ConsultaJurisprudencia(
            natureza="sumula" if sumula else "acordao",
            numero_processo_cnj=numero_cnj,
            numero_classe_tribunal=numero_classe,
            classe_processual=cadeia[-1] if cadeia else None,
            cadeia_recursal=cadeia,
            tribunal=_tribunal_explicito(anotacao["tribunal"], trecho),
            uf=_opcional(anotacao["uf"]),
            ano=int(anotacao["data"]) if anotacao["data"] else None,
            relator_norm=normalizar_relator(_opcional(anotacao["relator"]), relatores),
            numero_sumula=int(sumula.split()[0]) if sumula else None,
            sumula_vinculante="vinculante" in sumula.casefold() if sumula else None,
        )
    diploma, numero = _diploma(anotacao["diploma_legal"], trecho)
    return ConsultaLegislacao(
        numero_artigo=_opcional(anotacao["numero_do_artigo"]),
        diploma=diploma,
        numero_diploma=numero,
    )


def gerar(dataset: Path, destino: Path) -> None:
    relatores = json.loads(
        (ROOT / "data" / "relatores_padronizacao.json").read_text(encoding="utf-8")
    )
    relatores_canonicos = set(relatores.values())
    gold = pd.read_csv(
        dataset / "goldenset_offsets.csv",
        dtype=str,
        keep_default_na=False,
    )
    anotacoes = pd.read_csv(
        dataset / "goldenset_offsets_anotado_gustavo.csv",
        dtype=str,
        keep_default_na=False,
    ).set_index(["documento_id", "citacao_id"])

    for etapa in ETAPAS:
        shutil.rmtree(destino / etapa, ignore_errors=True)
    shutil.rmtree(destino / "03-veracity", ignore_errors=True)
    shutil.rmtree(destino / "03-entitys", ignore_errors=True)

    submissao = []
    for documento_id, linhas in gold.groupby("documento_id", sort=True):
        texto = (dataset / "txt" / f"{documento_id}.txt").read_text(encoding="utf-8")
        linhas = linhas.sort_values(["inicio", "fim"])
        candidatos = []
        for linha in linhas.itertuples():
            inicio, fim = int(linha.inicio), int(linha.fim)
            candidatos.append(
                CandidatoCitacao(
                    trecho=texto[inicio:fim],
                    tipo=TipoCitacao(linha.tipo),
                    inicio=inicio,
                    fim=fim,
                )
            )

        extracao = DocumentoExtraido(
            documento_id=documento_id,
            texto=texto,
            candidatos=candidatos,
        )
        analisados = []
        entidades = []
        classificados = []
        for linha, candidato in zip(linhas.itertuples(), candidatos, strict=True):
            classificacao = Classificacao(linha.classificacao)
            completa = classificacao != Classificacao.INCOMPLETA
            completude = ResultadoCompletude(completa=completa)
            analisados.append(
                CandidatoAnalisado(candidato=candidato, completude=completude)
            )
            anotacao = anotacoes.loc[(documento_id, linha.citacao_id)].to_dict()
            campos = _campos_entidades(anotacao, candidato.trecho, relatores)
            override = _OVERRIDES_ENTIDADES.get((documento_id, linha.citacao_id))
            if override:
                campos = type(campos).model_validate(
                    {**campos.model_dump(), **override}
                )
            if (
                isinstance(campos, ConsultaJurisprudencia)
                and campos.relator_norm is not None
                and campos.relator_norm not in relatores_canonicos
            ):
                raise ValueError(
                    f"relator_norm fora do dicionário: {campos.relator_norm!r}"
                )
            entidades.append(
                CandidatoEntidades(
                    candidato=candidato,
                    completude=completude,
                    campos_extraidos=campos,
                )
            )
            classificados.append(
                CandidatoClassificado(
                    candidato=candidato,
                    completude=completude,
                    veracidade=ResultadoVeracidade(
                        classificacao=classificacao,
                        id_canonico=(
                            int(linha.id_canonico)
                            if classificacao == Classificacao.REAL
                            else None
                        ),
                        justificativa="Derivado do goldenset.",
                    ),
                )
            )

        _salvar(destino / "01-extraction", extracao)
        _salvar(
            destino / "02-completeness",
            DocumentoCompletude(
                documento_id=documento_id,
                texto=texto,
                candidatos=analisados,
            ),
        )
        _salvar(
            destino / "03-entities",
            DocumentoEntidades(
                documento_id=documento_id,
                candidatos=entidades,
            ),
        )
        _salvar(
            destino / "04-veracity",
            DocumentoClassificado(
                documento_id=documento_id,
                texto=texto,
                candidatos=classificados,
            ),
        )
        citacoes = []
        for linha in linhas.itertuples():
            id_canonico = linha.id_canonico if linha.classificacao == "real" else "-"
            citacoes.append(
                f"{linha.inicio},{linha.fim},{linha.classificacao},{id_canonico},1.0000"
            )
        submissao.append((documento_id, "|".join(citacoes)))

    destino.mkdir(parents=True, exist_ok=True)
    with (destino / "submission_gold.csv").open(
        "w", newline="", encoding="utf-8"
    ) as arquivo:
        escritor = csv.writer(arquivo)
        escritor.writerow(["documento_id", "citacoes"])
        escritor.writerows(submissao)

    for etapa in ETAPAS:
        arquivos = list((destino / etapa).glob("*/resultado.json"))
        print(f"{destino / etapa}: {len(arquivos)} documentos")
    print(f"citações: {len(gold)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("desafio-jusbrasil-bracis-2026"),
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/gold"))
    args = parser.parse_args()
    gerar(args.dataset, args.output)


if __name__ == "__main__":
    main()
