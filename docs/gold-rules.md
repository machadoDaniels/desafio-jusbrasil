# Regras do gold

## 1. Evidência

Usar o trecho da citação e o TXT completo do documento.

## 2. Sem conhecimento externo

Não preencher dados ausentes do TXT com inferências jurídicas ou anotações enriquecidas.

## 3. Vínculo

Informação fora do trecho só vale se estiver claramente ligada à mesma citação, processo ou diploma.

## 4. Normalização obrigatória

Remover formatação, corrigir OCR inequívoco, normalizar CNJ, siglas jurídicas, relatores e cadeias recursais.

## 5. Jurisprudência

- Tribunal, UF e ano somente com evidência textual.
- `relator_norm` deve ser `null` ou um valor canônico presente em `data/relatores_padronizacao.json`.
- Não derivar tribunal ou ano dos segmentos internos do CNJ.
- CNJ, número de classe e número de registro devem estar explícitos e associados ao feito citado.
- Usar `sumula` somente com menção explícita a “Súmula”.

## 6. Legislação

- Normalizar diploma quando sigla ou nome estiver explícito.
- Preencher `numero_diploma` e `ano_diploma` somente quando o trecho ou o TXT trouxerem a identificação do mesmo diploma.
- Caso contrário, usar `null`.

## 7. Campos ausentes

Usar sempre `null`.
