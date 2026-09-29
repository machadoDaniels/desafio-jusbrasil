"""Parser determinístico de cabeçalhos para o modo híbrido do pré-processamento.

As chaves de busca (CNJ, número de classe, número de registro e UF) seguem um
formato estável por tribunal e ficam nas primeiras linhas do acórdão; súmulas e
dispositivos têm um cabeçalho fixo. Este módulo as extrai por expressão regular,
e o modelo fica só com os campos semânticos (classe, cadeia e UF residual).

Formatos reconhecidos (por tribunal), descritos por padrão e não por instância:
  STF  "<DATA> <TURMA> <CLASSE> <NÚMERO> <ESTADO> RELATOR : ..."
  STJ  "<CLASSE> Nº <NÚMERO> - <UF> (<AAAA/NNNNNNN-D>) RELATOR : ..."
  STM  "... <CLASSE> Nº <CNJ>/<UF> RELATOR: ..."
  TSE  "... <CLASSE> Nº <CNJ> - <CIDADE> - <ESTADO> Relator: ..."
       ou a numeração antiga "Nº <NÚMERO> (<CNJ>) - CLASSE <NN> - <CIDADE> - <ESTADO>"
  TST  sem cabeçalho; o número aparece como "TST-<CLASSES>-<CNJ>" no corpo e no rodapé,
       e a UF decorre da região do TRT contida no próprio CNJ.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any

from .contracts import DocumentoFonte

JANELA_CABECALHO_PADRAO = 500
JANELA_TST_PADRAO = 1500

_ESTADOS = {
    "ACRE": "AC", "ALAGOAS": "AL", "AMAPA": "AP", "AMAZONAS": "AM", "BAHIA": "BA",
    "CEARA": "CE", "DISTRITO FEDERAL": "DF", "ESPIRITO SANTO": "ES", "GOIAS": "GO",
    "MARANHAO": "MA", "MATO GROSSO DO SUL": "MS", "MATO GROSSO": "MT",
    "MINAS GERAIS": "MG", "PARA": "PA", "PARAIBA": "PB", "PARANA": "PR",
    "PERNAMBUCO": "PE", "PIAUI": "PI", "RIO DE JANEIRO": "RJ",
    "RIO GRANDE DO NORTE": "RN", "RIO GRANDE DO SUL": "RS", "RONDONIA": "RO",
    "RORAIMA": "RR", "SANTA CATARINA": "SC", "SAO PAULO": "SP", "SERGIPE": "SE",
    "TOCANTINS": "TO",
}
_UFS = set(_ESTADOS.values())
# Regiões da Justiça do Trabalho -> UF sede (regiões com mais de um estado usam a sede).
_TRT_UF = {
    "01": "RJ", "02": "SP", "03": "MG", "04": "RS", "05": "BA", "06": "PE", "07": "CE",
    "08": "PA", "09": "PR", "10": "DF", "11": "AM", "12": "SC", "13": "PB", "14": "RO",
    "15": "SP", "16": "MA", "17": "ES", "18": "GO", "19": "AL", "20": "SE", "21": "RN",
    "22": "PI", "23": "MT", "24": "MS",
}
_DIPLOMAS_POR_NUMERO = {
    "10406": "Código Civil",
    "8078": "Código de Defesa do Consumidor",
    "13105": "Código de Processo Civil",
    "3689": "Código de Processo Penal",
    "1001": "Código Penal Militar",
    "4737": "Código Eleitoral",
    "5452": "Consolidação das Leis do Trabalho",
}

_CNJ = r"(\d{1,7})\s*-\s*(\d{2})\s*\.\s*(\d{4})\s*\.\s*(\d)\s*\.\s*(\d{2})\s*\.\s*(\d{4})"
_RE_CNJ = re.compile(r"(?<!\d)" + _CNJ + r"(?!\d)")
_RE_TST = re.compile(r"\bTST\s*-\s*(?:[A-Z]{1,6}\s*-\s*)+" + _CNJ + r"(?!\d)")
_RE_NUMERO = re.compile(r"N[º°O.]?\s*(\d{1,3}(?:\.\s?\d{3})+|\d{4,8})(?!\d)")
_RE_REGISTRO_STJ = re.compile(r"\(\s*(\d{4})\s*/\s*(\d{7})\s*-\s*(\d)\s*\)")
_RE_UF_STJ = re.compile(r"\d\s*[-–]\s*([A-Z]{2})\s*\(")
_RE_UF_STM = re.compile(r"\.\d{4}\s*/\s*([A-Z]{2})\b")
_RE_NUMERO_TSE_ANTIGO = re.compile(r"N[º°O.]?\s*(\d{1,3}(?:\.\d{3})*)\s*\(\s*\d{1,7}\s*-\s*\d{2}\s*\.")
_RE_CLASSE_STF = re.compile(
    r"\b(?:RECLAMACAO|RECURSO EXTRAORDINARIO|AGRAVO|HABEAS CORPUS|MANDADO DE SEGURANCA|"
    r"ACAO|EMBARGOS|ARGUICAO|EXTRADICAO|INQUERITO|PETICAO|SUSPENSAO|"
    r"RCL|RE|ARE|HC|MS|ADI|ADPF|ADC|ADO|PET|INQ|EXT)\b[^0-9]{0,40}?"
    r"(\d{1,3}(?:\.\d{3})+|\d{2,7})(?!\d)"
)
_RE_ESTADO = re.compile(
    r"\b(" + "|".join(sorted(map(re.escape, _ESTADOS), key=len, reverse=True)) + r")\b"
)
_RE_SUMULA = re.compile(r"S[UÚ]MULA\s+(VINCULANTE\s+)?N?[º°O.]?\s*(\d+)")
_RE_DISPOSITIVO = re.compile(
    r"^\s*ARTIGO\s+(\S+)\s+D[AO]\s+(?:"
    r"CONSTITUICAO FEDERAL DE (\d{4})"
    r"|(LEI COMPLEMENTAR|DECRETO-LEI|LEI)\s+N[º°O.]?\s*([\d.]+)\s*,?\s+DE\s+(?:.*?\bDE\s+)?(\d{4})"
    r")"
)


def normalizar(texto: str) -> str:
    """Maiúsculas, sem acentos e com espaços colapsados; preserva as posições relativas."""
    sem_acento = "".join(
        c for c in unicodedata.normalize("NFKD", texto) if not unicodedata.combining(c)
    )
    return re.sub(r"\s+", " ", sem_acento).upper()


def _cnj(match: re.Match[str]) -> str:
    return match.group(1).zfill(7) + "".join(match.groups()[1:])


def _digitos(valor: str) -> str:
    return re.sub(r"\D", "", valor)


def _estado(texto: str, *, depois_de: int = 0, antes_de: int | None = None) -> str | None:
    fim = len(texto) if antes_de is None else antes_de
    encontrados = [m for m in _RE_ESTADO.finditer(texto, depois_de, fim)]
    return _ESTADOS[encontrados[0].group(1)] if encontrados else None


def chaves_acordao(documento: DocumentoFonte, *, janela_chars: int, janela_tst_chars: int) -> dict[str, Any]:
    """Chaves determinísticas de um acórdão e a janela de texto para o modelo."""
    texto = documento.texto
    cabecalho = normalizar(texto[:800])
    chaves: dict[str, Any] = {
        "numero_processo_cnj": None,
        "numero_classe_tribunal": None,
        "numero_registro_tribunal": None,
        "uf": None,
    }
    janela = texto[:janela_chars]
    tribunal = (documento.tribunal or "").upper()

    if tribunal == "STF":
        numero = _RE_CLASSE_STF.search(cabecalho)
        if numero:
            chaves["numero_classe_tribunal"] = _digitos(numero.group(1))
            chaves["uf"] = _estado(cabecalho, depois_de=numero.end())
    elif tribunal == "STJ":
        numero = _RE_NUMERO.search(cabecalho)
        if numero:
            chaves["numero_classe_tribunal"] = _digitos(numero.group(1))
        registro = _RE_REGISTRO_STJ.search(cabecalho)
        if registro:
            chaves["numero_registro_tribunal"] = "".join(registro.groups())
        uf = _RE_UF_STJ.search(cabecalho)
        if uf and uf.group(1) in _UFS:
            chaves["uf"] = uf.group(1)
    elif tribunal == "STM":
        cnj = _RE_CNJ.search(cabecalho)
        if cnj:
            chaves["numero_processo_cnj"] = _cnj(cnj)
            uf = _RE_UF_STM.search(cabecalho, cnj.start())
            if uf and uf.group(1) in _UFS:
                chaves["uf"] = uf.group(1)
    elif tribunal == "TSE":
        cnj = _RE_CNJ.search(cabecalho)
        if cnj:
            chaves["numero_processo_cnj"] = _cnj(cnj)
        antigo = _RE_NUMERO_TSE_ANTIGO.search(cabecalho)
        if antigo:
            chaves["numero_classe_tribunal"] = _digitos(antigo.group(1))
        relator = cabecalho.find("RELATOR")
        chaves["uf"] = _estado(
            cabecalho, antes_de=relator if relator > 0 else None
        )
    elif tribunal == "TST":
        codigo = _RE_TST.search(normalizar(texto))
        if codigo:
            chaves["numero_processo_cnj"] = _cnj(codigo)
            chaves["uf"] = _TRT_UF.get(chaves["numero_processo_cnj"][14:16])
            # O código "TST-<classes>-<CNJ>" carrega a cadeia recursal: entra na janela.
            texto_norm_pos = codigo.start()
            trecho = texto[max(0, texto_norm_pos - 250): texto_norm_pos + 250]
            janela = texto[:janela_tst_chars] + "\n[...]\n" + trecho
        else:
            janela = texto[:janela_tst_chars]
    else:
        cnj = _RE_CNJ.search(cabecalho)
        if cnj:
            chaves["numero_processo_cnj"] = _cnj(cnj)
    return {"campos": chaves, "janela": janela}


def campos_sumula(documento: DocumentoFonte) -> dict[str, Any] | None:
    """Número e caráter vinculante a partir do cabeçalho 'Súmula [Vinculante] n. N'."""
    match = _RE_SUMULA.search(normalizar(documento.texto[:200]))
    if not match:
        return None
    return {"numero_sumula": int(match.group(2)), "sumula_vinculante": bool(match.group(1))}


def campos_dispositivo(documento: DocumentoFonte) -> dict[str, Any] | None:
    """Artigo, diploma, número e ano a partir de 'Artigo N da Lei nº X, de ... de AAAA'."""
    match = _RE_DISPOSITIVO.search(normalizar(documento.texto[:300]))
    if not match:
        return None
    artigo = match.group(1)
    # Restaura o artigo como escrito (com ordinal/sufixos), a partir do texto original.
    original = re.search(r"Artigo\s+(\S+)", documento.texto[:300])
    if original:
        artigo = original.group(1)
    if match.group(2):
        return {
            "diploma": "Constituição Federal",
            "numero_diploma": None,
            "ano_diploma": int(match.group(2)),
            "numero_artigo": artigo,
        }
    tipo, numero, ano = match.group(3), _digitos(match.group(4)), int(match.group(5))
    if tipo == "LEI COMPLEMENTAR":
        diploma = "Lei Complementar"
    else:
        diploma = _DIPLOMAS_POR_NUMERO.get(numero, "Lei")
    return {"diploma": diploma, "numero_diploma": numero, "ano_diploma": ano, "numero_artigo": artigo}
