# Desafio Jusbrasil BRACIS 2026

Pipeline em quatro etapas para extrair e verificar citações jurídicas:

1. `extractor`: encontra citações e calcula os offsets localmente;
2. `completeness`: aplica uma regra específica por tipo e retorna somente se a citação é completa;
3. `entities`: extrai do trecho uma consulta estruturada alinhada ao banco enriquecido;
4. `veracity`: consulta primeiro as colunas estruturadas e usa FTS5 como fallback determinístico.

## Configuração

Toda a execução é configurada em [`pipeline.yaml`](pipeline.yaml). Cada etapa possui configuração independente de modelo, amostragem e concorrência.

```yaml
extractor:
  model: google/gemma-4-26B-A4B-it
  base_url: http://HOST:PORT/v1
  temperature: null
  top_p: null
  top_k: null
  reasoning_effort: null
  async_requests: true
  max_concurrency: 10
  debug: false
```

Todos os modelos são acessados por APIs OpenAI-compatible. Para vLLM, Ollama ou um gateway compatível, configure `base_url`; `null` usa o endpoint padrão da OpenAI.

### Modelo atual

O exemplo usa `google/gemma-4-26B-A4B-it`, um modelo mixture-of-experts com 26 bilhões de parâmetros totais e cerca de 4 bilhões ativos por token. O endpoint configurado atualmente anuncia `max_model_len: 32768` em `/v1/models`. O pipeline usa saída estruturada e, opcionalmente, reasoning.

O limite de 32.768 tokens é do servidor atual, não uma garantia portátil do modelo. Ele pode mudar conforme os argumentos usados para iniciar o vLLM e a memória disponível.

### Credencial

A chave é lida exclusivamente da variável `OPENAI_API_KEY`. Copie `.env.example` para `.env` e configure a chave quando necessário:

```bash
cp .env.example .env
```

Mesmo um endpoint local sem autenticação pode exigir um valor não vazio por compatibilidade com o SDK:

```dotenv
OPENAI_API_KEY=dummy
```

Não versione o arquivo `.env`.

## Parâmetros do modelo

| Campo | Padrão no código | Valores aceitos | Comportamento |
|---|---:|---|---|
| `model` | obrigatório | identificador textual | Deve coincidir com o ID publicado pelo servidor, por exemplo `google/gemma-4-26B-A4B-it`. |
| `base_url` | `null` | URL ou `null` | Para vLLM: `http://HOST:PORT/v1`. Para Ollama: `http://127.0.0.1:11434/v1`. `null` usa o endpoint padrão do cliente. |
| `temperature` | `null` | `0.0` a `2.0`, ou `null` | Controla aleatoriedade. `null` omite o parâmetro e preserva o default do servidor/modelo. |
| `top_p` | `null` | `0.0` a `1.0`, ou `null` | Amostragem nucleus. `null` omite o parâmetro. Em geral, ajuste `temperature` **ou** `top_p`, não ambos. |
| `top_k` | `null` | inteiro `>= 1`, ou `null` | Enviado ao vLLM dentro de `extra_body`. Quando `null`, `extra_body` não é enviado. |
| `reasoning_effort` | `null` | `none`, `minimal`, `low`, `medium`, `high`, `xhigh`, `max`, ou `null` | O suporte efetivo depende do provedor, da versão do servidor e do modelo. Para Gemma 4 no vLLM, prefira `none`, `low`, `medium` ou `high`. |
| `async_requests` | `false` | `true`, `false` | Habilita chamadas concorrentes quando a etapa é executada isoladamente. O orquestrador completo continua síncrono. |
| `max_concurrency` | `4` | inteiro `>= 1` | Número máximo de chamadas simultâneas nas execuções assíncronas. Não altera o batching interno do servidor vLLM. |
| `debug` | `false` | `true`, `false` | Na extração, preserva em `candidatos` as respostas sem correspondência textual, usando offsets nulos. Nas demais etapas, é apenas registrado no manifesto. |

`null` não significa enviar JSON `null`. Nas chamadas diretas, o pipeline usa `openai.omit` para `temperature`, `top_p` e `reasoning_effort`; para `top_k`, ele remove completamente `extra_body`. Assim, o servidor aplica seus próprios defaults.

Os defaults efetivos de amostragem do vLLM podem variar conforme sua versão, argumentos de inicialização, `generation_config.json` do modelo e chat template. Para execuções reproduzíveis, informe explicitamente os parâmetros desejados.

### Reasoning e `enable_thinking` no Gemma 4

No vLLM atual, não é necessário enviar simultaneamente `reasoning_effort` e `chat_template_kwargs.enable_thinking` para o Gemma 4. O servidor faz a conversão automaticamente:

| Configuração da requisição | Resultado no chat template |
|---|---|
| `reasoning_effort: low`, `medium` ou `high` | injeta `enable_thinking: true` |
| `reasoning_effort: none` | injeta `enable_thinking: false` |
| `reasoning_effort: null` | não injeta `enable_thinking`; preserva o comportamento configurado no servidor |

Para o Gemma 4, thinking é desativado por padrão, salvo se o servidor tiver sido iniciado com outro default. Portanto:

```yaml
# Reasoning desativado explicitamente
reasoning_effort: none

# Reasoning ativado
reasoning_effort: medium

# Não interferir; usar o default do servidor
reasoning_effort: null
```

Para este pipeline, uma configuração inicial razoável é manter reasoning desativado na extração e na completude, habilitando-o primeiro apenas na veracidade caso a melhoria de qualidade compense o aumento de latência e tokens.

O servidor vLLM também precisa estar configurado com o reasoning parser apropriado para separar reasoning e resposta final. Exemplo conceitual:

```bash
vllm serve google/gemma-4-26B-A4B-it \
  --reasoning-parser gemma4 \
  --host 0.0.0.0 \
  --port 8000
```

Um `enable_thinking` explícito em `chat_template_kwargs` prevalece sobre a ativação automática, mas o pipeline não o expõe porque seria redundante para o uso normal com `reasoning_effort`.

Referências:

- [vLLM — Reasoning Outputs](https://docs.vllm.ai/en/latest/features/reasoning_outputs/)
- [vLLM — Gemma 4 Usage Guide](https://docs.vllm.ai/projects/recipes/en/stable/Google/Gemma4.html)

## Execução

Pipeline completo, síncrono entre etapas:

```bash
uv run desafio-jusbrasil
```

Etapas isoladas:

```bash
uv run python -m desafio_jusbrasil.1_extractor
uv run python -m desafio_jusbrasil.2_completeness
uv run python -m desafio_jusbrasil.3_entities
uv run python -m desafio_jusbrasil.4_veracity
```

A etapa 4 possui duas alternativas independentes, ambas consumindo `03-entities` e gerando `04-veracity`:

```bash
# Busca somente pelas colunas estruturadas
uv run python -m desafio_jusbrasil.4_veracity

# Busca somente pelo índice FTS5
uv run python -m desafio_jusbrasil.4_veracity_fts
```

Todos os comandos aceitam um YAML alternativo, mantendo `pipeline.yaml` como padrão:

```bash
uv run desafio-jusbrasil --config pipeline-alternativo.yaml
uv run python -m desafio_jusbrasil.3_entities --config pipeline-alternativo.yaml
```

Extração e completude respeitam `async_requests`. `3_entities` usa somente chamadas assíncronas; `4_veracity` é determinística e não chama modelo. A ordem dos documentos e candidatos é preservada na saída.

### Pré-processamento do banco canônico

Para gerar uma cópia enriquecida sem alterar o SQLite original:

```bash
uv run python -m desafio_jusbrasil.database_preprocessing
uv run python -m desafio_jusbrasil.database_preprocessing --config configs/database_preprocessing-alternativo.yaml
```

Entrada, saída, diretório de auditoria, substituição do destino, parâmetros do agente e os prompts de sistema por natureza (`prompts`) ficam na seção `database_preprocessing` de `configs/database_preprocessing.yaml`. Para testar outro prompt, copie esse arquivo em `configs/` e passe com `--config`. O campo opcional `few_shot_path` aponta para um JSON com exemplos por natureza (`{"acordao": [{"entrada": ..., "saida": ...}]}`), enviados como pares usuário/assistente antes do documento; veja `configs/database_preprocessing_fewshot.yaml`, que usa `configs/few_shot_database_preprocessing.json`. Mudar os prompts invalida os checkpoints do `audit_dir`. Origem e destino nunca podem ser o mesmo arquivo. O processo não modifica a tabela FTS existente.

O run completo só materializa o banco se nenhum documento ficar em revisão. Para materializar com os resultados que já existem no `audit_dir`, sem chamar o modelo, use `--materializar`: documentos sem `resultado.json` ficam com as colunas novas nulas e são listados em `relatorio_materializacao.json`.

```bash
uv run python -m desafio_jusbrasil.database_preprocessing --materializar
```

Com `extrair_verbatim: true` na config, o agente também devolve, antes dos campos normalizados, o trecho literal do texto que sustenta cada campo. Os trechos ficam em `trechos` no `resultado.json` e não entram no banco. Para convertê-los em offsets do texto original:

```bash
uv run python scripts/gerar_spans_database_preprocessing.py outputs/database-preprocessing/<run>
```

O script grava `spans.jsonl` na pasta da run e imprime a taxa de trechos localizados por campo.

Para comparar uma run com o gold padrão `outputs/database-preprocessing/run-gemini-gold`:

```bash
uv run python scripts/evaluate_database_preprocessing.py \
  outputs/database-preprocessing/run-005
```

O relatório é salvo como `avaliacao.json` dentro da run avaliada. Use `--gold CAMINHO` para outra referência.

## Checkpoints e auditoria

Cada etapa grava um JSON por documento. A completude usa prompts separados para jurisprudência e legislação, retorna apenas `completa` e salva cada documento independentemente; resultados concluídos permanecem no disco mesmo se uma requisição posterior falhar. Em `03-entities`, o modelo recebe o trecho original de todas as citações, inclusive as incompletas, e extrai os campos totais ou parciais de busca; números processuais, natureza, súmula vinculante e relator normalizado recebem pós-processamento determinístico. `04-veracity` não chama LLM: consulta primeiro as colunas estruturadas do banco enriquecido e usa FTS5 como fallback, sempre em conexão SQLite somente leitura.

```text
outputs/<run>/01-extraction/<documento_id>/resultado.json
outputs/<run>/01-extraction/<documento_id>/0001.json
outputs/<run>/02-completeness/<documento_id>/resultado.json
outputs/<run>/02-completeness/<documento_id>/0001.json
outputs/<run>/03-entities/<documento_id>/resultado.json
outputs/<run>/03-entities/<documento_id>/0001.json
outputs/<run>/04-veracity/<documento_id>/resultado.json
outputs/<run>/04-veracity/<documento_id>/0001.json
outputs/<run>/predictions/<documento_id>.json
```

A extração primeiro procura o trecho literalmente; se isso falhar, tenta novamente tratando sequências de espaços, tabs e quebras de linha como equivalentes. Quando essa segunda busca encontra o trecho, os offsets são convertidos de volta para o texto original e `trecho` preserva inclusive suas quebras de linha.

O agente de entidades não recebe tools e não escreve SQL. Ele extrai campos estruturados de processo, súmula ou legislação. A etapa de veracidade transforma esses campos em uma consulta parametrizada e classifica o resultado: zero registros como `inventada`, um como `real` e mais de um como `incompleta`.

Com `extractor.debug: true`, candidatos ainda não localizados são preservados com `inicio` e `fim` iguais a `null`. Com o padrão `false`, eles não aparecem na lista processada de `candidatos`. Em ambos os modos, a resposta integral do modelo permanece nos arquivos numerados da pasta do documento para auditoria. Candidatos sem offsets nunca entram em `predictions`, pois a submissão exige posições numéricas.

Cada pasta de documento contém `resultado.json`, usado como checkpoint pela etapa seguinte, e arquivos numerados de auditoria. O checkpoint não duplica as chamadas nem o texto original; a completude relê o TXT de `input_dir` para montar o contexto. Em `03-entities`, cada arquivo numerado contém os campos extraídos e, ao final, `input` e `output` da chamada. Em `04-veracity`, cada arquivo numerado contém os campos recebidos, o SQL parametrizado, seus parâmetros, os registros e o resultado da classificação.

Credenciais nunca são incluídas nesses arquivos.

Cada diretório de etapa também contém `manifest.json`, que registra:

- etapa;
- data e hora UTC;
- formato do checkpoint;
- provider, modelo e endpoint;
- hiperparâmetros;
- modo assíncrono e concorrência.

Os avaliadores ignoram automaticamente o `manifest.json`.

Para avaliar as entidades estruturadas contra o gold:

```bash
uv run python scripts/evaluate_entities.py outputs/<run> \
  --gold outputs/gold/03-entities
```

O avaliador gera `03-entities/avaliacao.json`, um `resultado_eval.json` por documento e métricas de consulta exata por tipo, nível e campo. Em `cadeia_recursal`, a comparação ignora ordem e repetições.
