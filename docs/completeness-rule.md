# Regra de completude das citações

Esta regra foi inferida pela análise integral de `desafio-jusbrasil-bracis-2026/goldenset_offsets.csv` e dos 26 textos referenciados por seus offsets.

## Universo analisado

| Métrica | Valor |
|---|---:|
| Documentos | 26 |
| Citações | 192 |
| Completas (`real` ou `inventada`) | 160 |
| Incompletas | 32 |
| Reais | 96 |
| Inventadas | 64 |

Distribuição por tipo e classe:

| Tipo | Real | Inventada | Incompleta | Total |
|---|---:|---:|---:|---:|
| Jurisprudência | 82 | 50 | 32 | 164 |
| Lei | 14 | 14 | 0 | 28 |
| **Total** | **96** | **64** | **32** | **192** |

## Regra observada

Uma citação é **completa** quando contém um identificador canônico pesquisável. Ela é **incompleta** quando descreve uma fonte jurídica sem individualizá-la.

### Jurisprudência completa

Todos os 132 exemplos completos de jurisprudência têm um identificador numerado:

- 120 processos ou recursos numerados;
- 11 súmulas numeradas;
- 1 tema de repercussão geral numerado.

Exemplos:

```text
Rcl 88.178/RS
AgInt no AREsp nº 1.996.496/RJ
APL nº 7000449-40.2023.7.00.0000/RS
Súmula 331 do TST
Tema 2.680 da repercussão geral
```

O número não precisa estar perfeitamente formatado. O nível 2 do conjunto contém ruído de OCR e ainda considera completas referências como:

```text
5úmula 211 do STJ
RSE n. 7000171-3920237000000 (DF)
Rec. Esp. nº 1. 570.531 – CE
AgInt no RESP 21737l8 - SP
R.Esp. n° 1.45g.779-MA
```

Logo, espaços, pontuação irregular e trocas reconhecíveis entre letras e dígitos não tornam a referência incompleta por si sós.

### Jurisprudência incompleta

Os 32 exemplos incompletos têm ano, tribunal, classe e/ou relator, mas nenhum número que individualize o precedente:

```text
precedente do STF de 2026, da relatoria de CRISTIANO ZANIN
Reclamação do STF, de 2025, Rel. Min. CRISTIANO ZANIN
Agravo em Recurso Especial do STJ, de 2023, Rel. Min. Assusete Magalhães
Rcl de 2021, Rel. Min. Rosa Weber
acórdão do TSE julgado em 2020 sob relatoria de Edson Fachin
```

Portanto:

```text
classe + tribunal + ano + relator ≠ identificador único
```

O ano não deve ser confundido com o número do processo.

### Legislação completa

Todos os 28 exemplos legais têm um dispositivo numerado e um diploma identificável:

```text
art. 373, I, do CPC
art. 290 do Código Penal Militar
art. 75 da Lei Complementar nº 64/1990
art. 5º, LV, da Constituição Federal
art. 477 da Consolidação das Leis do Trabalho
```

Não há exemplos de legislação incompleta no gold. A regra para casos futuros é uma generalização conservadora: menção genérica à legislação ou artigo sem diploma identificável deve ser incompleta.

## Completude não é veracidade

No gold, 64 citações pesquisáveis são `inventada`. Portanto, um número inexistente, um artigo juridicamente incorreto ou uma súmula inventada continuam **completos** se oferecem uma chave de consulta específica.

```text
completude = há informação suficiente para buscar
veracidade  = a busca encontra correspondência canônica correta
```

Exemplos completos, mas rotulados como inventados no conjunto:

```text
Súmula 935 do STF
Súmula Vinculante 188
art. 303 da Constituição Federal
Tema 2.680 da repercussão geral
```

## Uso do contexto

O contexto pode completar partes que pertencem à mesma referência, especialmente quando uma quebra de linha separa a classe do número. Ele não deve fornecer o número de outra citação próxima. Todos os 32 casos incompletos continuam sem identificador próprio quando lidos no contexto original.

## Regra operacional

1. Identificar o tipo: jurisprudência ou lei.
2. Para jurisprudência, procurar número do processo/recurso, número de súmula ou número de tema.
3. Para lei, procurar dispositivo numerado e diploma identificável.
4. Tolerar degradação de OCR quando a intenção identificadora continuar reconhecível.
5. Não consultar existência nem plausibilidade nesta etapa.
6. Se não houver identificador pesquisável, retornar `completa=false` e `consulta=null`.
