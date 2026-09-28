# Modelos abertos para o pré-processamento do banco (RTX 4090)

Pesquisa de 2026-09-23 para substituir o Gemini (`gemini-3.8-flash`) por um modelo
aberto que rode numa RTX 4090 (24 GB) no `database_preprocessing`. Nada foi executado
ainda; esta é a shortlist e o plano de experimento.

## Perfil da tarefa

- Corpus: 1014 documentos — 996 acórdãos (média ~68k chars ≈ 17k tokens, máx ~150k
  chars), 13 dispositivos, 5 súmulas. Súmulas e dispositivos acertam 100% em todos os
  experimentos; o problema é o acórdão.
- Saída: JSON pequeno com enums fechados (21 classes, 27 UFs). O custo é quase todo
  prefill de contexto longo.
- O corte do texto importa (Gemini vs. `run-gemini-gold`, docs exatos de 1014):

  | head / tail (chars) | docs exatos | cadeia_recursal | numero_processo_cnj | uf |
  |---|---|---|---|---|
  | 500 / 1000 | 405 | 0.549 | 0.747 | 0.873 |
  | 5000 / 2000 | 577 | 0.706 | 0.893 | 0.912 |
  | 20000 / 5000 | 759 | 0.866 | 0.947 | 0.959 |

- Campos que exigem raciocínio sobre o documento inteiro (`cadeia_recursal`,
  distinguir o CNJ julgado dos citados) são onde modelos ≤32B tendem a falhar (HELMET).

## Shortlist

| # | Modelo | Por que testar |
|---|---|---|
| 1 | Gemma 4 12B QAT W4A16 (`google/gemma-4-12B-it-qat-w4a16-ct`) | Baseline já usado no projeto; atenção híbrida (sliding window), 256k de contexto |
| 2 | Qwen3.5-35B-A3B (W4A16) | MoE com ~3B ativos + Gated DeltaNet (≈3/4 das camadas sem KV cache): melhor perfil de velocidade. Risco: ~19–20 GB de pesos em 4 bits |
| 3 | Qwen3.6-27B (W4A16) | Teto de qualidade na 4090; Gated DeltaNet, 256k de contexto |
| 4 | Mistral Small 3.x 24B (AWQ) | Alternativa densa não-Qwen, 128k de contexto |
| 5 | Mistral NeMo 12B ou Qwen3 menor | Controle barato |

### Fila de testes: Gemma 4 quantizado pela Red Hat

Testar nas duas pipelines: pré-processamento do banco (`database_preprocessing`) e a
pipeline de citações (`configs/pipeline.yaml`: extractor, completeness, entities). Os
checkpoints já vêm quantizados; servir no vLLM sem `--quantization`. NVFP4 é nativo em
Blackwell (B200, GB10); FP8 é nativo em Hopper e Blackwell. `--quantization fp8` sobre o
BF16 falhou no GB10 (erro de kernel cutlass), e as imagens de vLLM do H100 não reconheciam
`gemma4_unified` até a atualização do nightly.

| Modelo | Pesos | Observação |
|---|---|---|
| `RedHatAI/gemma-4-12B-it-FP8-Dynamic` | 15.0 GB | Mesmo 12B do baseline, 8 bits: mede a perda do W4A16 |
| `RedHatAI/gemma-4-12B-it-NVFP4` | 10.3 GB | 12B em 4 bits NVFP4, alternativa ao QAT W4A16 |
| `RedHatAI/gemma-4-26B-A4B-it-NVFP4` | 16.4 GB | MoE com ~4B ativos: qualidade de 26B com custo próximo do 4B |

Descartados: Sabiá-4/Sabiazinho-3 (fechados, só API), Juru-7B (base, sem instruct,
contexto curto), Llama 3.1/3.2, Phi-4-mini (3.8B, PT não testado), GLM-4V-9B,
Qwen2.5-14B-1M (geração antiga, atenção cheia). Granite 4.1 8B fica como opcional
(sem evidência de PT nem de contexto longo).

Evidência de PT jurídico: no OAB-Bench, Qwen3.5-35B-A3B tirou 6.17/10, empatado com
Sabiazinho-3 (6.12). Não há benchmark público de extração estruturada de acórdãos em
PT; a decisão sai dos experimentos.

## Serving

- vLLM com W4A16/AWQ (kernels Marlin), `--enable-chunked-prefill`,
  `--kv-cache-dtype fp8` (Ada/sm_89 suporta FP8), xgrammar para `response_format`.
- Qwen: desligar thinking com `chat_template_kwargs={"enable_thinking": false}`. Exige
  ajuste em `agent.py`, que hoje só envia `top_k` via `extra_body`.

## Protocolo

- Todos os 1014 documentos com `text_start_char_limit: 20000` e
  `text_end_char_limit: 5000`.
- Avaliar com `scripts/evaluate_database_preprocessing.py` contra
  `outputs/database-preprocessing/run-gemini-gold`: docs exatos, acurácia por campo
  (foco em `cadeia_recursal`, `numero_processo_cnj`, `uf`) e docs/min.
- Meta: aproximar os 759 docs exatos do Gemini com o mesmo corte.

## Resultados (2026-09-24, DGX Spark GB10)

Sem 4090 disponível, os modelos rodaram no HPC Spark (1 job/usuário, 1 nó/job, então em
sequência) via vLLM, com corte head 10000 / tail 2000 chars, concorrência 32,
`temperature: 0` e thinking desligado nos Qwen. Limite: 2h para a base inteira.
Métrica: concordância com `run-gemini-gold` (1014 docs), sem `relator_norm` (determinístico;
`relatores_padronizacao.json` não existe mais no repo); docs que falharam contam como erro.
Saídas em `outputs/database-preprocessing/exp-*`.

| Modelo | Tempo | Docs exatos | CNJ | Nº classe | Nº registro | Classe | Cadeia | UF |
|---|---|---|---|---|---|---|---|---|
| Gemini, mesmo corte (referência) | – | 705 | 0.943 | 0.976 | 1.000 | 0.955 | 0.814 | 0.941 |
| Qwen3.6-27B (`cyankiwi/Qwen3.6-27B-AWQ-INT4`) | 106 min | 66 | 0.745 | 0.447 | 0.536 | 0.335 | 0.252 | 0.814 |
| Qwen3.5-35B-A3B (`Qwen/Qwen3.5-35B-A3B-GPTQ-Int4`) | 60 min | 20 | 0.721 | 0.382 | 0.289 | 0.403 | 0.121 | 0.780 |
| Mistral Small 3.2 24B (`Intel/...-int4-AutoRound`) | 109 min | 12 | 0.686 | 0.304 | 0.120 | 0.415 | 0.231 | 0.774 |
| Gemma 4 12B (`google/gemma-4-12B-it-qat-w4a16-ct`) | 97 min | 11 | 0.752 | 0.286 | 0.710 | 0.189 | 0.191 | 0.782 |
| Mistral NeMo 12B (`casperhansen/...-awq`) | >2h (717 docs) | 7 | – | – | – | – | – | – |

Mistral NeMo estourou o limite: gera saídas longas sem parar (o pipeline não define
`max_tokens`). Nos 699 acórdãos que processou, empata com o Qwen3.6-27B em classe e UF, mas
tem CNJ 0.31 contra 0.80.

Conclusões:

- Nenhum modelo aberto chega perto do Gemini. Súmulas e dispositivos saem certos; os erros
  estão nos acórdãos, em classe e cadeia recursal: os modelos não mapeiam nomes para siglas
  (ex.: "AG.REG." do STF → `AgRg`), tomam o recurso mais recente como classe principal,
  inventam "RE" e preenchem nº de registro onde não há.
- O gold também foi gerado pelo Gemini e tem ruído, então a tabela mede concordância com o
  Gemini, não acerto absoluto.
- Próximo passo mais barato: incluir no prompt um glossário de classes e siglas e uma regra
  explícita para a classe principal, testando com Qwen3.6-27B (melhor) e Qwen3.5-35B-A3B
  (mais rápido).
- Ajustes feitos para os experimentos: `pattern` de só dígitos nos campos numéricos
  (`contracts.py`), que eliminou retries por CNJ com pontuação, e `chat_template_kwargs`
  opcional (`agent.py`, `pipeline.py`).

## Resultados da fila Gemma 4 Red Hat (2026-09-24, 1× H100 no h100n2)

Servidos com vLLM 0.30.1rc1 (nightly) + transformers 5.10.1. NVFP4 no GB10 (Spark) gera
texto corrompido com o kernel FlashInferCutlassNvFp4 do vLLM 0.30.0; no H100 roda via
fallback. Referência: Gemma 4 12B QAT W4A16.

Pré-processamento do banco (1014 docs, `configs/database_preprocessing_{fp8,nvfp4,26b_nvfp4}.yaml`:
few-shot de 4 acórdãos, `extrair_verbatim: true`, corte 10000/1000, timeout 40 s). Concordância
com `run-gemini-gold` sem os docs do few-shot; docs sem resultado (timeout) contam como erro.

| Modelo | Tempo | Sem resultado | Docs exatos | CNJ | Nº classe | Nº registro | Classe | Cadeia | UF | Trechos localizados |
|---|---|---|---|---|---|---|---|---|---|---|
| 12B W4A16 few-shot (config anterior) | – | 0 | 228 | 0.748 | 0.860 | 0.916 | 0.709 | 0.427 | 0.850 | – |
| 12B FP8 (`RedHatAI/gemma-4-12B-it-FP8-Dynamic`) | ~47 min | 6 | 251 | 0.791 | 0.900 | 0.888 | 0.684 | 0.390 | 0.858 | 22.5% |
| 12B NVFP4 (`RedHatAI/gemma-4-12B-it-NVFP4`) | ~40 min | 3 | 219 | 0.774 | 0.845 | 0.874 | 0.664 | 0.362 | 0.851 | 21.2% |
| 26B-A4B NVFP4 (`RedHatAI/gemma-4-26B-A4B-it-NVFP4`) | ~26 min | 8 | 222 | 0.820 | 0.755 | 0.780 | 0.705 | 0.384 | 0.840 | 24.0% |
| Gemini com spans (referência) | 4 min | 0 | 721 | 0.960 | 0.982 | 1.000 | 0.933 | 0.833 | 0.948 | 99.9% |

Pipeline de citações (26 docs, `configs/pipeline_gemma4-*.yaml`). "Gold DB" usa o banco
enriquecido do gold na etapa 4 para isolar o modelo; "ponta a ponta" usa o banco
pré-processado pelo próprio modelo.

| Modelo | Tempo | Extração F1 | Completude F1 | Entidades F1 exata | Veracidade F1 | Oficial (gold DB) | Oficial (ponta a ponta) |
|---|---|---|---|---|---|---|---|
| 12B W4A16 (`run-010`) | – | 0.933 | 0.922 | 0.631 | 0.868 | 0.933 | 0.888 |
| 12B FP8 (`run-011`) | 3m20 | 0.957 | 0.952 | 0.649 | 0.904 | 0.961 | 0.888 |
| 12B NVFP4 (`run-012`) | 3m55 | 0.955 | 0.949 | 0.587 | 0.912 | 0.963 | 0.812 |
| 26B-A4B NVFP4 (`run-013`) | 2m42 | 0.943 | 0.932 | 0.564 | 0.867 | 0.891 | 0.852 |

Conclusões:

- O 12B FP8 é o melhor da fila nas duas pipelines: mais docs exatos no pré-processamento e
  empate com o NVFP4 na métrica oficial com gold DB.
- Os trechos literais funcionam mal nos modelos abertos (~22%): copiam o valor normalizado.
- O 26B-A4B não compensou: pior em nº classe/registro e nas citações, apesar de mais rápido.
- Timeouts de 40 s deixam 3–8 docs sem resultado; há loops de repetição (ex.: doc_0172).

### Qwen3.8-27B INT4 simulando RTX 4090 (2026-09-25)

`RedHatAI/Qwen3.8-27B-INT4` (W4A16) numa H100 limitada a `--gpu-memory-utilization 0.28`
(~22 GB), `--max-num-seqs 16`, thinking desligado. Cabe numa 4090: pesos 16,8 GiB, KV cache
de 45.296 tokens (2,76× o contexto de 16k); com 128 sequências falha por falta de blocos
Mamba (máximo 33). Configs: `configs/database_preprocessing_qwen3.8-27b-int4.yaml` e
`configs/pipeline_qwen3.8-27b-int4.yaml`.

| Pipeline | Tempo | Resultado |
|---|---|---|
| Pré-processamento | 20 min, 0 sem resultado | 203 docs exatos; CNJ 0.710, nº classe 0.786, nº registro 0.844, classe 0.678, cadeia 0.349, UF 0.793; trechos 24,4% |
| Citações (gold DB) | 12m50 | extração F1 0.677, completude 0.656, entidades 0.418, veracidade 0.619, oficial 0.650 |
| Citações (ponta a ponta) | – | oficial 0.600 |

Pior que todos os Gemma 4 nas duas pipelines. Na extração de citações devolve limites de
trecho diferentes do gold e às vezes altera o texto (ex.: maiúscula inicial), o que derruba
o casamento de spans; os prompts foram ajustados no Gemma.
