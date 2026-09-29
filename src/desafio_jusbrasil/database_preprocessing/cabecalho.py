"""Parser determinístico de cabeçalhos para o modo híbrido do pré-processamento.

As chaves de busca (CNJ, número de classe, número de registro e UF) seguem um
formato estável por tribunal e ficam nas primeiras linhas do acórdão; súmulas e
dispositivos têm um cabeçalho fixo. Este módulo as extrai por expressão regular,
e o modelo fica só com os campos semânticos (classe, cadeia e UF residual).

As regras são aplicadas em cascata. O nome do tribunal é só uma prioridade: as regras
do formato conhecido correm primeiro e as regras genéricas (CNJ em qualquer posição do
cabeçalho, registro no formato AAAA/NNNNNNN-D, número após "Nº", estado por extenso,
UF derivada dos segmentos J.TR do CNJ) preenchem o que ficou vazio, inclusive em
tribunais sem formato conhecido. Cada chave recebe um nível de confiança e o documento
recebe o nível da sua melhor chave; o chamador decide, por nível, quanto delegar ao modelo.

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
# Código TR do CNJ nas Justiças Estadual (J=8) e Eleitoral (J=6): ordem alfabética das UFs.
_UF_POR_CODIGO_ESTADO = dict(
    zip(
        (f"{i:02d}" for i in range(1, 28)),
        ("AC", "AL", "AP", "AM", "BA", "CE", "DF", "ES", "GO", "MA", "MT", "MS", "MG", "PA",
         "PB", "PR", "PE", "PI", "RJ", "RN", "RS", "RO", "RR", "SC", "SE", "SP", "TO"),
    )
)
ALTA, MEDIA, BAIXA = "alta", "media", "baixa"
NIVEIS = (ALTA, MEDIA, BAIXA)
JANELA_FALLBACK_PADRAO = 3000
CHAVES_ACORDAO = ("numero_processo_cnj", "numero_classe_tribunal", "numero_registro_tribunal", "uf")
CHAVES_IDENTIFICADORAS = CHAVES_ACORDAO[:3]
# Nas regras genéricas, um número solto só conta nos primeiros caracteres: depois disso
# o texto já é ementa e os números são de citações.
_JANELA_NUMERO_GENERICO = 300
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
_RE_NUMERO = re.compile(r"N[º°O.]?\s*(\d{1,3}(?:\.\s?\d{3})+|\d{1,8})(?![\d.])")
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
_RE_UF_SUFIXO = re.compile(r"\d\s*[-–/]\s*([A-Z]{2})\b(?!\s*\d)")
# Número que acompanha o nome de uma classe processual, em qualquer tribunal.
_RE_CLASSE_NUMERO = re.compile(
    r"\b(?:RECLAMACAO|RECURSO EXTRAORDINARIO|RECURSO ESPECIAL|RECURSO DE REVISTA|RECURSO ORDINARIO|"
    r"RECURSO EM HABEAS CORPUS|AGRAVO|APELACAO|HABEAS CORPUS|MANDADO DE SEGURANCA|ACAO|EMBARGOS|"
    r"ARGUICAO|EXTRADICAO|INQUERITO|PETICAO|SUSPENSAO|CONFLITO|REPRESENTACAO|"
    r"RCL|RE|ARE|HC|MS|ADI|ADPF|ADC|ADO|PET|INQ|EXT|RESP|ARESP|AGINT|AGRG|EDCL|RMS|RHC|APL|AI|RR|ARR)"
    r"\b[^0-9]{0,40}?(\d{1,3}(?:\.\d{3})+|\d{2,8})(?![\d.])"
)
_RE_EMENTA = re.compile(r"\bEMENTA\b")
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


# Classes que só existem em um tribunal: o modelo confunde "REsp" com "REspe" (e vice-versa).
_CLASSES_TSE = {"REsp": "REspe", "AREsp": "AREspe"}
_CLASSES_NAO_TSE = {valor: chave for chave, valor in _CLASSES_TSE.items()}


def classe_coerente_com_tribunal(classe: str | None, tribunal: str | None) -> str | None:
    """Corrige a variante eleitoral/não eleitoral da classe conforme o tribunal."""
    if classe is None:
        return None
    sigla, sep, resto = classe.partition(" — ")
    mapa = _CLASSES_TSE if (tribunal or "").upper() == "TSE" else _CLASSES_NAO_TSE
    nova = mapa.get(sigla, sigla)
    if nova == sigla:
        return classe
    return nova  # a sigla sem descrição é aceita pelo contrato final


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


def cnj_valido(cnj: str | None) -> bool:
    """Confere o dígito verificador (módulo 97) de um CNJ com 20 dígitos."""
    if not cnj or len(cnj) != 20 or not cnj.isdigit():
        return False
    sequencial, dv, resto = cnj[:7], int(cnj[7:9]), cnj[9:]
    return dv == 98 - (int(sequencial + resto + "00") % 97)


def uf_do_cnj(cnj: str | None) -> str | None:
    """UF implícita nos segmentos J.TR do CNJ (Justiça do Trabalho, Eleitoral e Estadual)."""
    if not cnj or len(cnj) != 20:
        return None
    justica, tribunal = cnj[13], cnj[14:16]
    if justica == "5":
        return _TRT_UF.get(tribunal)
    if justica in {"6", "8"}:
        return _UF_POR_CODIGO_ESTADO.get(tribunal)
    return None


class _Cascata:
    """Acumula chaves com confiança e o nome da regra que as preencheu."""

    def __init__(self) -> None:
        self.campos: dict[str, Any] = dict.fromkeys(CHAVES_ACORDAO)
        self.confianca: dict[str, str | None] = dict.fromkeys(CHAVES_ACORDAO)
        self.regras: list[str] = []

    def definir(self, campo: str, valor: Any, nivel: str, regra: str) -> bool:
        if valor is None or self.campos[campo] is not None:
            return False
        self.campos[campo] = valor
        self.confianca[campo] = nivel
        self.regras.append(f"{campo}:{regra}")
        return True

    def cnj(self, match: re.Match[str], regra: str) -> None:
        valor = _cnj(match)
        self.definir("numero_processo_cnj", valor, ALTA if cnj_valido(valor) else MEDIA, regra)

    @property
    def nivel(self) -> str:
        """Nível do documento: o da melhor chave identificadora (a UF não identifica)."""
        niveis = {self.confianca[chave] for chave in CHAVES_IDENTIFICADORAS if self.confianca[chave]}
        if ALTA in niveis:
            return ALTA
        return MEDIA if niveis else BAIXA


def _regras_stf(c: _Cascata, cabecalho: str) -> None:
    numero = _RE_CLASSE_STF.search(cabecalho)
    if numero:
        c.definir("numero_classe_tribunal", _digitos(numero.group(1)), ALTA, "stf_classe_numero")
        c.definir("uf", _estado(cabecalho, depois_de=numero.end()), ALTA, "stf_estado_apos_numero")


def _regras_stj(c: _Cascata, cabecalho: str) -> None:
    numero = _RE_NUMERO.search(cabecalho)
    if numero:
        c.definir("numero_classe_tribunal", _digitos(numero.group(1)), ALTA, "stj_numero")
    registro = _RE_REGISTRO_STJ.search(cabecalho)
    if registro:
        c.definir("numero_registro_tribunal", "".join(registro.groups()), ALTA, "stj_registro")
    uf = _RE_UF_STJ.search(cabecalho)
    if uf and uf.group(1) in _UFS:
        c.definir("uf", uf.group(1), ALTA, "stj_uf")


def _regras_stm(c: _Cascata, cabecalho: str) -> None:
    cnj = _RE_CNJ.search(cabecalho)
    if cnj:
        c.cnj(cnj, "stm_cnj")
        uf = _RE_UF_STM.search(cabecalho, cnj.start())
        if uf and uf.group(1) in _UFS:
            c.definir("uf", uf.group(1), ALTA, "stm_uf_sufixo")


def _regras_tse(c: _Cascata, cabecalho: str) -> None:
    cnj = _RE_CNJ.search(cabecalho)
    if cnj:
        c.cnj(cnj, "tse_cnj")
    antigo = _RE_NUMERO_TSE_ANTIGO.search(cabecalho)
    if antigo:
        c.definir("numero_classe_tribunal", _digitos(antigo.group(1)), ALTA, "tse_numero_antigo")
    relator = cabecalho.find("RELATOR")
    c.definir(
        "uf", _estado(cabecalho, antes_de=relator if relator > 0 else None), ALTA, "tse_estado"
    )


def _regras_tst(c: _Cascata, texto_norm: str) -> re.Match[str] | None:
    codigo = _RE_TST.search(texto_norm)
    if codigo:
        c.cnj(codigo, "tst_codigo")
        c.definir("uf", uf_do_cnj(c.campos["numero_processo_cnj"]), ALTA, "tst_regiao_trt")
    return codigo


_REGRAS_POR_TRIBUNAL = {"STF": _regras_stf, "STJ": _regras_stj, "STM": _regras_stm, "TSE": _regras_tse}


def _regras_genericas(c: _Cascata, cabecalho: str, texto_norm: str, *, formato_com_cnj: bool) -> None:
    """Preenche o que as regras do tribunal deixaram vazio; vale para qualquer tribunal.

    ``formato_com_cnj`` indica que o formato conhecido do tribunal traz o CNJ do próprio
    processo (ou que o tribunal é desconhecido). Quando o formato não o traz (STF, STJ), um
    CNJ no cabeçalho é de outro processo (origem, precedente) e não é extraído.
    """
    ementa = _RE_EMENTA.search(cabecalho)
    inicio = cabecalho if ementa is None else cabecalho[: ementa.start()]
    if c.campos["numero_processo_cnj"] is None and formato_com_cnj:
        codigo = _RE_TST.search(texto_norm)
        if codigo:
            c.cnj(codigo, "generica_codigo_tst")
        else:
            cnj = _RE_CNJ.search(inicio)
            if cnj:
                c.cnj(cnj, "generica_cnj_cabecalho")
    if c.campos["numero_registro_tribunal"] is None:
        registro = _RE_REGISTRO_STJ.search(inicio)
        if registro:
            c.definir("numero_registro_tribunal", "".join(registro.groups()), ALTA, "generica_registro")
    if c.campos["numero_classe_tribunal"] is None:
        antigo = _RE_NUMERO_TSE_ANTIGO.search(inicio)
        if antigo:
            c.definir("numero_classe_tribunal", _digitos(antigo.group(1)), ALTA, "generica_numero_antigo")
        elif c.campos["numero_processo_cnj"] is None:
            # Sem CNJ, o processo é identificado pelo número que acompanha a classe.
            topo = inicio[:_JANELA_NUMERO_GENERICO]
            por_classe = next(
                (m for m in _RE_CLASSE_NUMERO.finditer(topo) if _numero_isolado(topo, m)), None
            )
            if por_classe:
                c.definir("numero_classe_tribunal", _digitos(por_classe.group(1)), ALTA, "generica_classe_numero")
            else:
                numero = next((m for m in _RE_NUMERO.finditer(topo) if _numero_isolado(topo, m)), None)
                if numero:
                    c.definir("numero_classe_tribunal", _digitos(numero.group(1)), MEDIA, "generica_numero")
    if c.campos["uf"] is None:
        uf = _RE_UF_STJ.search(inicio)
        if uf and uf.group(1) in _UFS:
            c.definir("uf", uf.group(1), ALTA, "generica_uf_registro")
    if c.campos["uf"] is None:
        c.definir("uf", uf_do_cnj(c.campos["numero_processo_cnj"]), ALTA, "generica_uf_do_cnj")
    if c.campos["uf"] is None:
        relator = inicio.find("RELATOR")
        if relator > 0:
            c.definir("uf", _estado(inicio, antes_de=relator), ALTA, "generica_estado_antes_relator")
        else:
            c.definir("uf", _estado(inicio), MEDIA, "generica_estado")
    if c.campos["uf"] is None:
        sufixo = _RE_UF_SUFIXO.search(inicio)
        if sufixo and sufixo.group(1) in _UFS:
            c.definir("uf", sufixo.group(1), MEDIA, "generica_uf_sufixo")


def _numero_isolado(texto: str, match: re.Match[str]) -> bool:
    """Rejeita um número que é prefixo de CNJ mal formatado ("7000045-53.2019...")."""
    return not re.match(r"\s*-\s*\d", texto[match.end(1):])


def chaves_acordao(
    documento: DocumentoFonte,
    *,
    janela_chars: int,
    janela_tst_chars: int,
    janela_fallback_chars: int = JANELA_FALLBACK_PADRAO,
) -> dict[str, Any]:
    """Chaves determinísticas de um acórdão, sua confiança e a janela de texto para o modelo.

    Devolve ``campos`` (as quatro chaves), ``confianca`` (nível por chave), ``nivel`` (do
    documento: alta, media ou baixa), ``regras`` (quais dispararam) e ``janela``. Com nível
    baixo a janela cresce para ``janela_fallback_chars``, porque o modelo passa a ser a única
    fonte das chaves.
    """
    texto = documento.texto
    texto_norm = normalizar(texto)
    cabecalho = normalizar(texto[:800])
    tribunal = (documento.tribunal or "").upper()
    c = _Cascata()
    codigo_tst: re.Match[str] | None = None

    if tribunal == "TST":
        codigo_tst = _regras_tst(c, texto_norm)
    elif tribunal in _REGRAS_POR_TRIBUNAL:
        _REGRAS_POR_TRIBUNAL[tribunal](c, cabecalho)
    _regras_genericas(c, cabecalho, texto_norm, formato_com_cnj=tribunal not in {"STF", "STJ"})
    if codigo_tst is None and any(regra.endswith("codigo_tst") for regra in c.regras):
        codigo_tst = _RE_TST.search(texto_norm)

    if codigo_tst is not None:
        # O código "TST-<classes>-<CNJ>" carrega a cadeia recursal: entra na janela.
        posicao = codigo_tst.start()
        trecho = texto[max(0, posicao - 250): posicao + 250]
        janela = texto[:janela_tst_chars] + "\n[...]\n" + trecho
    elif tribunal == "TST":
        janela = texto[:janela_tst_chars]
    elif c.nivel == BAIXA:
        janela = texto[:janela_fallback_chars]
    else:
        janela = texto[:janela_chars]
    return {
        "campos": c.campos,
        "confianca": c.confianca,
        "nivel": c.nivel,
        "regras": c.regras,
        "janela": janela,
    }


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
