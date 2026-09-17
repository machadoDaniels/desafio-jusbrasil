# Exploração da base canônica Jusbrasil

- **Base analisada:** `desafio-jusbrasil-bracis-2026/desafio1_bracis.db`
- **Data da exploração:** 2026-09-16 16:12 -03
- **Modo de acesso:** somente leitura (`file:...?...mode=ro`)
- **Integridade SQLite:** `PRAGMA integrity_check` retornou `ok`.
- **Tamanho do arquivo:** 94.040.064 bytes (89,7 MiB).

## Visão geral

A base contém **1.014 documentos canônicos** para verificar citações extraídas dos textos de entrada. Há uma tabela de conteúdo, `documentos`, e um índice textual virtual FTS5, `documentos_fts`.

| Métrica | Valor |
|---|---:|
| Documentos em `documentos` | 1.014 |
| Entradas em `documentos_fts` | 1.014 |
| IDs Jusbrasil distintos | 1.014 |
| Textos distintos | 976 |
| Registros além do total de textos distintos | 38 |
| Caracteres armazenados | 67.737.021 |
| Comprimento médio do texto | 66.801,8 caracteres |
| Menor / mediana / maior texto | 264 / 67.914 / 149.907 caracteres |

O número de textos distintos não deve ser lido automaticamente como número de duplicatas sem inspeção semântica: há 38 registros a mais que textos distintos, mas os registros mantêm identificadores canônicos próprios.

## Modelo de dados

### `documentos`

Tabela canônica de documentos jurídicos.

| Campo | Tipo SQLite | Regra / significado |
|---|---|---|
| `documento_id` | `TEXT` | Chave primária; identificador canônico interno (`doc_0001`, etc.). |
| `id` | `INTEGER` | Obrigatório e único; `doc_id` do Jusbrasil usado como `id_canonico` nas predições. |
| `tribunal` | `TEXT` | Opcional. |
| `ano` | `INTEGER` | Opcional. |
| `relator` | `TEXT` | Opcional. |
| `natureza` | `TEXT` | Obrigatório; restrito a `acordao`, `sumula` ou `dispositivo`. |
| `tipo` | `TEXT` | Obrigatório; restrito a `jurisprudencia` ou `lei`. |
| `texto` | `TEXT` | Obrigatório; conteúdo integral para busca e confronto. |
| `texto_len` | `INTEGER` | Obrigatório; tamanho declarado de `texto`. |

Restrições e índices observados:

- `documento_id` é a chave primária;
- `id` tem restrição `UNIQUE`;
- há `CHECK` para os domínios de `natureza` e `tipo`;
- há índices B-tree em `tribunal`, `ano` e `natureza`.

### `documentos_fts`

Índice virtual SQLite FTS5 sobre `texto`:

```sql
CREATE VIRTUAL TABLE documentos_fts USING fts5(
  texto,
  content='documentos',
  content_rowid='rowid',
  tokenize='unicode61 remove_diacritics 2'
)
```

Ele usa conteúdo externo: cada entrada é ligada a `documentos.rowid`. O tokenizador `unicode61` remove diacríticos, portanto uma consulta FTS por termos sem acento pode encontrar conteúdo acentuado. A contagem do índice e da tabela canônica é a mesma (1.014).

O diagrama correspondente está em [`database.mmd`](database.mmd) e [`database.svg`](database.svg).

## Composição jurídica

| Tipo | Natureza | Quantidade | Percentual |
|---|---|---:|---:|
| `jurisprudencia` | `acordao` | 996 | 98,22% |
| `jurisprudencia` | `sumula` | 5 | 0,49% |
| `lei` | `dispositivo` | 13 | 1,28% |

A base é, portanto, fortemente dominada por **acórdãos jurisprudenciais**. As combinações observadas são determinísticas neste recorte: acórdão e súmula aparecem como jurisprudência; dispositivo aparece como lei.

## Cobertura de metadados

| Natureza | Registros | Com tribunal | Com ano | Com relator |
|---|---:|---:|---:|---:|
| `acordao` | 996 | 996 | 996 | 993 |
| `sumula` | 5 | 5 | 0 | 0 |
| `dispositivo` | 13 | 0 | 0 | 0 |
| **Total** | **1.014** | **1.001** | **996** | **993** |

No total, `tribunal` está ausente em 13 registros, `ano` em 18 e `relator` em 21. Essa ausência é coerente com o tipo de fonte: dispositivos de lei não têm esses metadados; súmulas não possuem ano e relator neste conjunto. Entre os 996 acórdãos, apenas 3 não têm relator.

### Tribunais

| Tribunal | Quantidade |
|---|---:|
| STJ | 202 |
| STF | 201 |
| STM | 200 |
| TSE | 199 |
| TST | 199 |
| Não informado | 13 |

Os 13 registros sem tribunal são os dispositivos legais. A distribuição dos acórdãos e súmulas entre os cinco tribunais superiores é quase uniforme.

### Anos

Os metadados de ano cobrem **2009 a 2026**, com 18 anos distintos. Os anos com mais registros são 2024 (176), 2023 (141) e 2021 (105).

| Ano | Registros | Ano | Registros |
|---:|---:|---:|---:|
| 2009 | 1 | 2018 | 30 |
| 2010 | 14 | 2019 | 51 |
| 2011 | 9 | 2020 | 75 |
| 2012 | 18 | 2021 | 105 |
| 2013 | 10 | 2022 | 65 |
| 2014 | 7 | 2023 | 141 |
| 2015 | 40 | 2024 | 176 |
| 2016 | 41 | 2025 | 84 |
| 2017 | 80 | 2026 | 49 |

Há **159 relatores distintos** entre os documentos com relator informado.

## Comprimento do conteúdo

| Natureza | Mínimo | Média | Máximo | Total de caracteres |
|---|---:|---:|---:|---:|
| `acordao` | 11.357 | 67.947 | 149.907 | 67.675.559 |
| `sumula` | 273 | 593 | 1.799 | 2.967 |
| `dispositivo` | 264 | 4.500 | 16.780 | 58.495 |

Quantis globais de `texto_len`:

| Percentil | Caracteres |
|---:|---:|
| 0% | 264 |
| 25% | 48.052 |
| 50% | 67.914 |
| 75% | 85.820 |
| 90% | 100.331 |
| 95% | 108.940 |
| 100% | 149.907 |

Em todos os 1.014 registros, `texto_len` é igual a `length(texto)` no SQLite.

## Exemplos de linhas

Os exemplos abaixo são registros reais da base. Os excertos foram truncados para não reproduzir o conteúdo integral.

### Tabela `documentos`

| `documento_id` | `id` (Jusbrasil) | Tribunal | Ano | Natureza / tipo | Exemplo do conteúdo |
|---|---:|---|---:|---|---|
| `doc_0001` | 6.457.031.728 | STF | 2026 | `acordao` / `jurisprudencia` | `PRIMEIRA TURMA — AG.REG. NA RECLAMAÇÃO 76.532 — RELATOR: MIN. CRISTIANO ZANIN` |
| `doc_0201` | 2.566.535.283 | STJ | 2019 | `acordao` / `jurisprudencia` | `RECURSO ESPECIAL Nº 1.741.784 - PR — RELATOR: MINISTRO JOEL ILAN PACIORNIK` |
| `1289710642` | 1.289.710.642 | STJ | — | `sumula` / `jurisprudencia` | `Súmula n. 83 do STJ: Não se conhece do recurso especial pela divergência...` |
| `10577194` | 10.577.194 | — | — | `dispositivo` / `lei` | `Artigo 276 da Lei nº 4.737, de 15 de julho de 1965` |

Os dois últimos exemplos mostram por que `tribunal`, `ano` e `relator` são opcionais: a súmula não contém ano ou relator neste acervo; o dispositivo legal não recebe tribunal, ano ou relator.

### Resultado de uma busca em `documentos_fts`

A consulta abaixo usa o índice FTS5, não uma busca `LIKE` no texto integral:

```sql
SELECT d.documento_id, d.id, d.tribunal, d.ano,
       snippet(documentos_fts, 0, '[', ']', '…', 12)
FROM documentos_fts
JOIN documentos AS d ON d.rowid = documentos_fts.rowid
WHERE documentos_fts MATCH 'constituicao'
ORDER BY d.ano DESC, d.documento_id
LIMIT 3;
```

| `documento_id` | `id` | Tribunal | Ano | Trecho encontrado |
|---|---:|---|---:|---|
| `doc_0001` | 6.457.031.728 | STF | 2026 | `…5º, LXXVIII, da [Constituição] Federal). IV. DISPOSITIVO…` |
| `doc_0003` | 6.457.028.552 | STF | 2026 | `…57, § 4º, da [Constituição] de 1988. Pedido de interpretação conforme…` |
| `doc_0023` | 5.999.556.566 | STF | 2026 | `…Esse entendimento decorre da premissa de que a [Constituição] da Republica garante…` |

Os colchetes são apenas marcadores inseridos por `snippet` para destacar o termo que casou com a consulta.

## Implicações para o pipeline

1. Para localizar candidatos textuais, a busca FTS5 deve ser preferida a uma varredura completa de `texto`; ela foi construída precisamente para esse propósito.
2. O campo a retornar como `id_canonico` em uma classificação `real` é `documentos.id`, não `documento_id`.
3. Filtros por `tribunal`, `ano` e `natureza` são baratos por terem índices próprios e reduzem o espaço de busca antes da comparação do texto.
4. A maioria dos documentos é longa: a mediana supera 67 mil caracteres. O pipeline deve enviar ao modelo apenas trechos recuperados e metadados relevantes, não o conteúdo completo de acórdãos.
5. A categoria `lei` é pequena (13 dispositivos); uma estratégia de normalização de artigo, inciso e diploma legal tende a ser mais importante do que busca semântica ampla nessa classe.
