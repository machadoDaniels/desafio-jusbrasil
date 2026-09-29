"""Parser determinístico de cabeçalhos: formatos por tribunal com números fictícios."""

from __future__ import annotations

import unittest

from desafio_jusbrasil.database_preprocessing import cabecalho
from desafio_jusbrasil.database_preprocessing.contracts import DocumentoFonte


def _doc(tribunal: str | None, texto: str, natureza: str = "acordao") -> DocumentoFonte:
    return DocumentoFonte(
        documento_id="doc-teste", id=1, natureza=natureza, tribunal=tribunal,
        ano=2024, relator="Ministro Teste", texto=texto,
    )


def _chaves(tribunal: str, texto: str) -> dict:
    return cabecalho.chaves_acordao(
        _doc(tribunal, texto), janela_chars=500, janela_tst_chars=1500
    )["campos"]


class CabecalhoTest(unittest.TestCase):
    def test_stf_numero_da_classe_e_estado_por_extenso(self) -> None:
        texto = "19/08/2024 PRIMEIRA TURMA AG.REG. NA RECLAMAÇÃO 12.345 MINAS GERAIS RELATOR : MIN. FULANO"
        chaves = _chaves("STF", texto)
        self.assertEqual(chaves["numero_classe_tribunal"], "12345")
        self.assertEqual(chaves["uf"], "MG")
        self.assertIsNone(chaves["numero_processo_cnj"])

    def test_stj_numero_registro_e_uf(self) -> None:
        texto = "AgInt no RECURSO ESPECIAL Nº 1.234.567 - PR (2016/0123456-7) RELATOR : MINISTRO FULANO"
        chaves = _chaves("STJ", texto)
        self.assertEqual(chaves["numero_classe_tribunal"], "1234567")
        self.assertEqual(chaves["numero_registro_tribunal"], "201601234567")
        self.assertEqual(chaves["uf"], "PR")

    def test_stj_numero_curto(self) -> None:
        chaves = _chaves("STJ", "QO na CAUTELAR INOMINADA CRIMINAL Nº 87 - DF (2022/0187319-4) RELATORA : MINISTRA FULANA")
        self.assertEqual(chaves["numero_classe_tribunal"], "87")
        self.assertEqual(chaves["numero_registro_tribunal"], "202201873194")
        self.assertEqual(chaves["uf"], "DF")

    def test_classe_eleitoral_so_no_tse(self) -> None:
        self.assertEqual(cabecalho.classe_coerente_com_tribunal("REspe — Recurso Especial Eleitoral", "STJ"), "REsp")
        self.assertEqual(cabecalho.classe_coerente_com_tribunal("REsp — Recurso Especial", "TSE"), "REspe")
        self.assertEqual(cabecalho.classe_coerente_com_tribunal("AgInt — Agravo Interno", "STJ"), "AgInt — Agravo Interno")
        self.assertIsNone(cabecalho.classe_coerente_com_tribunal(None, "STJ"))

    def test_stm_cnj_e_uf_no_sufixo(self) -> None:
        texto = "Poder Judiciário STM EXTRATO DE ATA APELAÇÃO CRIMINAL Nº 7000123-45.2021.7.00.0000/AM RELATOR: MINISTRO FULANO"
        chaves = _chaves("STM", texto)
        self.assertEqual(chaves["numero_processo_cnj"], "70001234520217000000")
        self.assertEqual(chaves["uf"], "AM")

    def test_tse_cnj_com_espaco_de_ocr_e_estado_antes_do_relator(self) -> None:
        texto = "TRIBUNAL SUPERIOR ELEITORAL ACÓRDÃO RECURSO ESPECIAL ELEITORAL Nº 123-45. 2016.6.09.0065 - CLASSE 32 - CIDADE - GOIÁS Relator: Ministro Fulano"
        chaves = _chaves("TSE", texto)
        self.assertEqual(chaves["numero_processo_cnj"], "00001234520166090065")
        self.assertEqual(chaves["uf"], "GO")

    def test_tse_numeracao_antiga_tem_numero_de_classe(self) -> None:
        texto = "TRIBUNAL SUPERIOR ELEITORAL ACÓRDÃO RECURSO ORDINÁRIO Nº 1.234 ( 47142-16.2008.6.00.0000) - CLASSE 37 - CIDADE - PARANÁ Relator: Ministro Fulano"
        chaves = _chaves("TSE", texto)
        self.assertEqual(chaves["numero_classe_tribunal"], "1234")
        self.assertEqual(chaves["numero_processo_cnj"], "00471421620086000000")
        self.assertEqual(chaves["uf"], "PR")

    def test_tst_cnj_pelo_codigo_e_uf_pela_regiao(self) -> None:
        texto = (
            "A C Ó R D Ã O (5ª Turma) RECURSO DE REVISTA. TEMA. " * 10
            + "Vistos os autos do Recurso de Revista nº TST-RR-1234-56.2010.5.04.0231, em que é "
            + "Recorrente FULANO. " + "corpo " * 50
            + "PROCESSO Nº TST-RR-1234-56.2010.5.04.0231 Firmado por assinatura digital"
        )
        resultado = cabecalho.chaves_acordao(_doc("TST", texto), janela_chars=500, janela_tst_chars=1500)
        self.assertEqual(resultado["campos"]["numero_processo_cnj"], "00012345620105040231")
        self.assertEqual(resultado["campos"]["uf"], "RS")
        self.assertIn("TST-RR-1234-56.2010.5.04.0231", resultado["janela"])

    def test_sem_cabecalho_reconhecido_deixa_chaves_nulas(self) -> None:
        chaves = _chaves("STJ", "Texto sem número nem registro.")
        self.assertEqual(
            chaves,
            {"numero_processo_cnj": None, "numero_classe_tribunal": None,
             "numero_registro_tribunal": None, "uf": None},
        )

    def test_sumula_comum_e_vinculante(self) -> None:
        comum = cabecalho.campos_sumula(_doc("STJ", "Súmula n. 123 do STJ ...", "sumula"))
        vinculante = cabecalho.campos_sumula(_doc("STF", "Súmula Vinculante n. 9 do STF ...", "sumula"))
        self.assertEqual(comum, {"numero_sumula": 123, "sumula_vinculante": False})
        self.assertEqual(vinculante, {"numero_sumula": 9, "sumula_vinculante": True})

    def test_dispositivo_lei_codigo_e_constituicao(self) -> None:
        cpc = cabecalho.campos_dispositivo(_doc(None, "Artigo 373 da Lei nº 13.105, de 16 de março de 2015 Art. 373. ...", "dispositivo"))
        self.assertEqual(cpc, {"diploma": "Código de Processo Civil", "numero_diploma": "13105", "ano_diploma": 2015, "numero_artigo": "373"})
        dl = cabecalho.campos_dispositivo(_doc(None, "Artigo 818 do Decreto-Lei nº 5.452, de 1º de maio de 1943 Art. 818. ...", "dispositivo"))
        self.assertEqual(dl["diploma"], "Consolidação das Leis do Trabalho")
        self.assertEqual(dl["ano_diploma"], 1943)
        lc = cabecalho.campos_dispositivo(_doc(None, "Artigo 1º da Lei Complementar nº 64, de 18 de maio de 1990 Art. 1º ...", "dispositivo"))
        self.assertEqual((lc["diploma"], lc["numero_diploma"], lc["numero_artigo"]), ("Lei Complementar", "64", "1º"))
        cf = cabecalho.campos_dispositivo(_doc(None, "Artigo 5º da Constituição Federal de 1988 Art. 5º Todos ...", "dispositivo"))
        self.assertEqual(cf, {"diploma": "Constituição Federal", "numero_diploma": None, "ano_diploma": 1988, "numero_artigo": "5º"})


if __name__ == "__main__":
    unittest.main()
