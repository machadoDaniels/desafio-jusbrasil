"""Parser determinístico de cabeçalhos: formatos por tribunal com números fictícios."""

from __future__ import annotations

import unittest

from desafio_jusbrasil.database_preprocessing import agent, cabecalho
from desafio_jusbrasil.database_preprocessing.contracts import DocumentoFonte


def _doc(tribunal: str | None, texto: str, natureza: str = "acordao") -> DocumentoFonte:
    return DocumentoFonte(
        documento_id="doc-teste", id=1, natureza=natureza, tribunal=tribunal,
        ano=2024, relator="Ministro Teste", texto=texto,
    )


def _extrair(tribunal: str | None, texto: str) -> dict:
    return cabecalho.chaves_acordao(
        _doc(tribunal, texto), janela_chars=500, janela_tst_chars=1500, janela_fallback_chars=3000
    )


def _chaves(tribunal: str | None, texto: str) -> dict:
    return _extrair(tribunal, texto)["campos"]


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


class CascataTest(unittest.TestCase):
    """Regras genéricas para tribunais sem formato conhecido e níveis de confiança."""

    def test_digito_verificador_do_cnj(self) -> None:
        self.assertTrue(cabecalho.cnj_valido("12345674920208130024"))
        self.assertFalse(cabecalho.cnj_valido("12345675020208130024"))
        self.assertFalse(cabecalho.cnj_valido("1234567"))

    def test_uf_pelos_segmentos_do_cnj(self) -> None:
        self.assertEqual(cabecalho.uf_do_cnj("12345674920208130024"), "MG")  # 8.13 = TJMG
        self.assertEqual(cabecalho.uf_do_cnj("00012347320215020301"), "SP")  # 5.02 = TRT2
        self.assertEqual(cabecalho.uf_do_cnj("06001235120246090000"), "GO")  # 6.09 = TRE-GO
        self.assertIsNone(cabecalho.uf_do_cnj("10012344820224013400"))  # 4.01 = TRF1: várias UFs

    def test_tribunal_desconhecido_com_cnj_valido_e_nivel_alto(self) -> None:
        texto = (
            "TRIBUNAL DE JUSTIÇA DO ESTADO DE MINAS GERAIS APELAÇÃO CÍVEL Nº 1234567-49.2020.8.13.0024 "
            "COMARCA DE BELO HORIZONTE APELANTE: FULANO APELADO: BELTRANO RELATOR: DES. SICRANO EMENTA: ..."
        )
        saida = _extrair("TJMG", texto)
        self.assertEqual(saida["campos"]["numero_processo_cnj"], "12345674920208130024")
        self.assertEqual(saida["campos"]["uf"], "MG")
        self.assertIsNone(saida["campos"]["numero_classe_tribunal"])
        self.assertEqual(saida["nivel"], "alta")
        self.assertIn("numero_processo_cnj:generica_cnj_cabecalho", saida["regras"])

    def test_tribunal_desconhecido_com_codigo_no_corpo(self) -> None:
        texto = "ACÓRDÃO 3ª TURMA " + "x" * 900 + " PROCESSO Nº TST-AIRR-0001234-73.2021.5.02.0301 EMENTA ..."
        saida = _extrair(None, texto)
        self.assertEqual(saida["campos"]["numero_processo_cnj"], "00012347320215020301")
        self.assertEqual(saida["campos"]["uf"], "SP")
        self.assertIn("[...]", saida["janela"])

    def test_cnj_com_digito_invalido_tem_confianca_media(self) -> None:
        texto = "APELAÇÃO CÍVEL Nº 1234567-50.2020.8.13.0024 RELATOR: DES. FULANO"
        saida = _extrair("TJMG", texto)
        self.assertEqual(saida["campos"]["numero_processo_cnj"], "12345675020208130024")
        self.assertEqual(saida["confianca"]["numero_processo_cnj"], "media")
        self.assertEqual(saida["nivel"], "media")

    def test_numero_solto_sem_cnj_tem_confianca_media(self) -> None:
        saida = _extrair("TJXX", "PROCESSO Nº 4321 RELATOR: DES. FULANO EMENTA: ...")
        self.assertEqual(saida["campos"]["numero_classe_tribunal"], "4321")
        self.assertEqual(saida["confianca"]["numero_classe_tribunal"], "media")
        self.assertEqual(saida["nivel"], "media")

    def test_numero_apos_classe_sem_cnj_tem_confianca_alta(self) -> None:
        saida = _extrair(None, "MANDADO DE SEGURANÇA Nº 12.345 SÃO PAULO RELATOR: ...")
        self.assertEqual(saida["campos"]["numero_classe_tribunal"], "12345")
        self.assertEqual(saida["confianca"]["numero_classe_tribunal"], "alta")
        self.assertEqual(saida["campos"]["uf"], "SP")

    def test_sem_chaves_nivel_baixo_e_janela_maior(self) -> None:
        texto = "ACÓRDÃO. EMENTA: " + "palavra " * 800
        saida = _extrair("TJXX", texto)
        self.assertEqual(saida["nivel"], "baixa")
        self.assertTrue(all(valor is None for valor in saida["campos"].values()))
        self.assertEqual(len(saida["janela"]), 3000)

    def test_uf_sozinha_nao_eleva_o_nivel(self) -> None:
        saida = _extrair("TJXX", "ACÓRDÃO GOIÁS RELATOR: DES. FULANO EMENTA: ...")
        self.assertEqual(saida["campos"]["uf"], "GO")
        self.assertEqual(saida["nivel"], "baixa")

    def test_documento_com_cnj_nao_recebe_numero_solto(self) -> None:
        texto = "APELAÇÃO Nº 7000123-45.2019.7.00.0000 RELATOR: MIN. FULANO PROCESSO Nº 7000045-53.2019.7.00.0000"
        chaves = _chaves("STM", texto)
        self.assertEqual(chaves["numero_processo_cnj"], "70001234520197000000")
        self.assertIsNone(chaves["numero_classe_tribunal"])

    def test_prefixo_de_cnj_quebrado_nao_vira_numero(self) -> None:
        chaves = _chaves("TSE", "RECURSO ESPECIAL ELEITORAL Nº 102-81.2012.620.0019 CLASSE 32 - CIDADE - RIO GRANDE DO NORTE RELATOR:")
        self.assertIsNone(chaves["numero_classe_tribunal"])
        self.assertEqual(chaves["uf"], "RN")

    def test_cnj_depois_da_ementa_nao_e_do_processo(self) -> None:
        texto = "RECURSO ESPECIAL Nº 1.234.567 - PR (2016/0123456-7) RELATOR : MIN. FULANO EMENTA: cumprimento da ação coletiva 1234567-49.2020.8.13.0024"
        chaves = _chaves("STJ", texto)
        self.assertIsNone(chaves["numero_processo_cnj"])
        self.assertEqual(chaves["numero_classe_tribunal"], "1234567")

    def test_cabecalho_de_pagina_com_ementa_colada_nao_corta(self) -> None:
        texto = "SUPREMO TRIBUNAL FEDERAL EMENTAEACORDAO INTEIRO TEOR DO ACORDAO - PAGINA 1 DE 57 02/02/2021 SEGUNDA TURMA HABEAS CORPUS 123.456 SÃO PAULO RELATOR : MIN. FULANO"
        saida = _extrair(None, texto)
        self.assertEqual(saida["campos"]["numero_classe_tribunal"], "123456")
        self.assertEqual(saida["campos"]["uf"], "SP")
        self.assertEqual(saida["nivel"], "alta")


class FusaoFallbackTest(unittest.TestCase):
    """Fusão das chaves do cabeçalho com a resposta do modelo nos níveis de fallback."""

    def test_chave_alta_prevalece_e_media_so_preenche_vazio(self) -> None:
        chaves = {"numero_processo_cnj": "1" * 20, "numero_classe_tribunal": "4321", "uf": "GO"}
        confianca = {"numero_processo_cnj": "alta", "numero_classe_tribunal": "media", "uf": "alta"}
        modelo = {"numero_processo_cnj": "2" * 20, "numero_classe_tribunal": "9999", "uf": None}
        dados = agent._fundir_chaves(modelo, chaves, confianca)
        self.assertEqual(dados["numero_processo_cnj"], "1" * 20)
        self.assertEqual(dados["numero_classe_tribunal"], "9999")
        self.assertEqual(dados["uf"], "GO")

    def test_cnj_com_zeros_a_mais_e_recomposto(self) -> None:
        dados = {"numero_processo_cnj": "000001028120126200019"}
        agent._sanear_cnj_do_modelo(dados)
        self.assertEqual(dados["numero_processo_cnj"], "00001028120126200019")
        dados = {"numero_processo_cnj": "1" * 25}
        agent._sanear_cnj_do_modelo(dados)
        self.assertIsNone(dados["numero_processo_cnj"])

    def test_cnj_invalido_cede_ao_candidato_unico_do_texto(self) -> None:
        texto = "cabeçalho ilegível ... rodapé: 1234567-49.2020.8.13.0024"
        dados = {"numero_processo_cnj": "12345675020208130024"}  # dígito verificador errado
        agent._sanear_cnj_do_modelo(dados, texto)
        self.assertEqual(dados["numero_processo_cnj"], "12345674920208130024")
        dados = {"numero_processo_cnj": "12345675020208130024"}
        agent._sanear_cnj_do_modelo(dados, "sem candidato no texto")
        self.assertEqual(dados["numero_processo_cnj"], "12345675020208130024")

    def test_stf_nao_extrai_cnj_do_processo_de_origem(self) -> None:
        texto = "19/08/2024 PRIMEIRA TURMA AG.REG. NA RECLAMAÇÃO 12.345 MINAS GERAIS RELATOR : MIN. FULANO INTDO.(A/S) : RELATOR DO AIRR Nº 1234567-49.2020.8.13.0024 DO TRIBUNAL"
        chaves = _chaves("STF", texto)
        self.assertIsNone(chaves["numero_processo_cnj"])
        self.assertEqual(chaves["numero_classe_tribunal"], "12345")

    def test_numero_que_repete_o_sequencial_do_cnj_e_descartado(self) -> None:
        dados = {"numero_processo_cnj": "00004872620126060049", "numero_classe_tribunal": "487"}
        agent._descartar_prefixo_de_cnj(dados)
        self.assertIsNone(dados["numero_classe_tribunal"])
        dados = {"numero_processo_cnj": "00004872620126060049", "numero_classe_tribunal": "3334"}
        agent._descartar_prefixo_de_cnj(dados)
        self.assertEqual(dados["numero_classe_tribunal"], "3334")


if __name__ == "__main__":
    unittest.main()
