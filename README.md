# Jusbrasil BRACIS 2026 — Caça-Alucinações


Tudo roda offline em uma GPU de 24 GB de VRAM, com um único modelo aberto servido localmente pelo vLLM: [`google/gemma-4-12B-it-qat-w4a16-ct`](https://huggingface.co/google/gemma-4-12B-it-qat-w4a16-ct).

## Avaliação final: como executar

### Requisitos

- Linux com uma GPU NVIDIA de pelo menos 24 GB de VRAM (desenvolvido para uma RTX 4090), drivers NVIDIA recentes e o NVIDIA Container Toolkit.
- Docker. Nada mais é necessário no host: o vLLM, o Python e os pesos do modelo ficam na imagem.
- Acesso à internet **apenas para construir a imagem**, que baixa as dependências e os pesos do modelo. A execução não usa rede nem chama APIs externas.

### Preparação: build da imagem (com internet, antes da execução)

```bash
docker build -t desafio-jusbrasil .
```

O build é o único passo que usa a rede. A imagem é baseada em `vllm/vllm-openai:v0.29.0`, fixada por digest, com as dependências Python em versões fixas, e baixa para dentro dela os pesos do modelo em uma revisão fixa (`1d2c2d7f2466070e69d6fb3fd5ce9a7d75f2f6ee`), cerca de 32 GB no total. Com a imagem pronta, a execução abaixo roda sem internet; nenhum peso é baixado em tempo de execução.

### Execução (ponto de entrada único)

Na raiz do repositório, no host:

```bash
bash run.sh <caminho_db> <pasta_txt> <arquivo_saida>
```

`<caminho_db>` é uma base no formato original da amostra de desenvolvimento (`desafio1_bracis.db`), e `<pasta_txt>` contém um `<documento_id>.txt` por documento. A saída é um CSV no formato de submissão (`documento_id,citacoes`, com entradas `inicio,fim,classe,id_canonico,confianca` separadas por `|` e `-` para documentos sem citações), gerado por [`scripts/json_to_submission.py`](scripts/json_to_submission.py).

Se `<arquivo_saida>` for uma pasta, o CSV é gravado como `submission.csv` dentro dela.

O [`run.sh`](run.sh) usa a imagem já construída (`IMAGEM` sobrescreve a tag, padrão `desafio-jusbrasil:latest`; se ela não existir, o script para e pede o build) e a executa com `--network none`, montando a base, a pasta de TXT (somente leitura), a pasta de saída e `outputs/`. Dentro do container, [`scripts/run_in_container.sh`](scripts/run_in_container.sh) executa todas as etapas sem intervenção manual:

1. Sobe o vLLM em `127.0.0.1:8000` com a revisão fixa do modelo e espera até ele responder.
2. **Enriquece a base** recebida em uma cópia nova (`outputs/final/enriched.db`) com o código de `src/desafio_jusbrasil/database_preprocessing`. A base original nunca é modificada.
3. Roda as quatro etapas do pipeline sobre a pasta de TXT, consultando a cópia enriquecida.
4. Grava o CSV de submissão e derruba o servidor.

Checkpoints intermediários, auditorias e logs são gravados em `outputs/final/` no repositório, com posse do usuário do host: `run.log` (log de toda a execução, com horário, etapa e duração) e `vllm.log` (log do servidor). O container se chama `desafio-jusbrasil` (`CONTAINER` sobrescreve), então uma segunda execução simultânea é recusada pelo Docker e uma execução em andamento pode ser parada com `docker stop desafio-jusbrasil`. `GPUS` escolhe a GPU repassada ao Docker (padrão `all`; por exemplo, `GPUS='"device=0"'`).

A imagem também pode ser executada diretamente, sem o `run.sh`:

```bash
docker run --rm --gpus all --network none --ipc=host \
  -v /caminho/para/dados:/dados -v /caminho/para/saida:/saida \
  desafio-jusbrasil /dados/<base>.db /dados/<pasta_txt> /saida/submission.csv
```



### Configuração da execução

| Item | Valor | Onde |
| --- | --- | --- |
| Modelo | `google/gemma-4-12B-it-qat-w4a16-ct`, revisão `1d2c2d7…` | `Dockerfile`, `scripts/run_in_container.sh` |
| Servidor | vLLM `v0.29.0`, `--max-model-len 32768`, `--gpu-memory-utilization 0.92`, `--max-num-seqs 16`, `--reasoning-parser gemma4`, `--seed 0` | `scripts/run_in_container.sh` |
| Amostragem | `temperature: 0` em todas as chamadas ao modelo | `configs/final_*.yaml` |
| Concorrência | 8 requisições | `configs/final_*.yaml` |
| Config do pipeline | [`configs/final_pipeline.yaml`](configs/final_pipeline.yaml) | |
| Config do enriquecimento | [`configs/final_database_preprocessing.yaml`](configs/final_database_preprocessing.yaml) | |

Dentro do container, `MAX_MODEL_LEN`, `GPU_MEMORY_UTILIZATION`, `MAX_NUM_SEQS`, `PORTA` e `WORKDIR` podem ser sobrescritos por variáveis de ambiente.

### Reprodutibilidade

- Todas as chamadas ao modelo usam `temperature: 0` e o servidor roda com `--seed 0`. Não há amostragem.
- A revisão do modelo, a imagem do vLLM (tag e digest) e as dependências Python da imagem são fixas; para desenvolvimento local, as versões ficam no `uv.lock`.
- Sem caminhos absolutos, passos manuais ou arquivos fora do repositório: o único artefato extra, o dicionário de nomes de relatores `data/relatores_padronizacao.json`, é versionado, incluído na imagem e copiado ao lado da base enriquecida.
- Pequenas diferenças numéricas ainda podem vir dos kernels da GPU e do agrupamento de requisições no vLLM.


## Abordagem

```mermaid
flowchart LR
    Z[.db original] --> Y[0. Enriquecimento da base: LLM]
    A[TXT original] --> B[1. Extração: LLM + localização local do trecho]
    B --> C[2. Completude: LLM]
    C --> D[3. NER: extratores de campos em paralelo]
    D --> E[4. Veracidade: consultas SQLite determinísticas]
    Y --> E
    E --> F[Predições e CSV de submissão]
```

### 0. Enriquecimento da base

A base canônica guarda a maior parte dos identificadores apenas dentro do texto dos documentos. O enriquecimento copia a base e acrescenta colunas estruturadas que a etapa 4 consegue consultar:

- **Acórdãos:** número CNJ do processo julgado, número sequencial com a classe do tribunal, número de registro no tribunal, classe processual principal, cadeia recursal (`cadeia_recursal`) e UF.
- **Súmulas:** número e se é vinculante (`vinculante`).
- **Dispositivos legais:** diploma normalizado, número do diploma e artigo.
- `relator_norm`: normalização determinística do nome do relator com `data/relatores_padronizacao.json`.

Cada documento é enviado uma vez ao modelo, com prompt por tipo, exemplos few-shot (`configs/few_shot_database_preprocessing.json`) e um JSON schema com vocabulários fechados. Os acórdãos são cortados nos primeiros 10.000 e nos últimos 2.000 caracteres, onde esses metadados costumam aparecer. Campos numéricos guardam só dígitos, e os números CNJ são validados contra o texto. Se um documento falhar depois das novas tentativas, a base é materializada mesmo assim e esse documento fica com as colunas nulas (o script de execução recorre a `--materializar`).

### 1. Extração

O LLM retorna o texto e o tipo da citação. O Python calcula os offsets; o modelo nunca os gera.

O extrator primeiro procura o texto retornado de forma literal. Se não encontrar, tenta uma correspondência com espaços em branco normalizados e mapeia o resultado de volta para o documento original. Uma citação localizada guarda o recorte original `texto[inicio:fim]`, preservando a formatação. Os offsets são posições de caracteres Unicode, com fim exclusivo.

Documentos muito grandes são divididos em blocos sobrepostos, com orçamento de tokens e tratamento de erros de capacidade. Com `tokenizer_path: /tokenize`, a contagem de tokens vem do servidor vLLM. Os offsets de cada bloco são convertidos para offsets do documento, e trechos duplicados são unidos.

Candidatos não localizados nunca entram na submissão final.

### 2. Completude

Um LLM retorna `completa: true` ou `false`, usando a citação e ±300 caracteres de contexto. Ele não consulta a base nem decide se uma citação é inventada.

Os prompts exigem uma referência numerada pesquisável para jurisprudência (número do processo, súmula ou tema) e, para legislação, um dispositivo numerado com um diploma identificável. O contexto pode juntar partes da mesma referência, mas não pode fornecer identificadores de outra citação.

A chamada também pede os log-probabilities dos tokens. A probabilidade de a citação estar incompleta é lida do token `true`/`false` da resposta e normalizada entre os dois valores. É a confiança usada para `incompleta` (ver [Confiança](#confiança)).

### 3. NER em paralelo

Cada extrator recebe o texto original da citação e retorna só os campos que lhe cabem. Todas as chamadas usam o mesmo modelo e rodam de forma independente, sob um limite de concorrência compartilhado.

| Extrator | Campos |
| --- | --- |
| Natureza | `natureza`, `numero_sumula`, `sumula_vinculante` |
| Identificadores | `numero_processo_cnj`, `numero_classe_tribunal`, `numero_registro_tribunal` |
| Classe processual | `classe_processual`, `cadeia_recursal` |
| Tribunal | `tribunal` |
| UF | `uf` |
| Ano | `ano` |
| Relator | `relator` |
| Diploma e artigo | `diploma`, `numero_artigo` |
| Número do diploma | `numero_diploma` |

A jurisprudência usa os sete primeiros extratores; a legislação usa os dois últimos. O coordenador une os campos e aplica a normalização determinística, incluindo o tratamento do CNJ e a normalização do nome do relator em `relator_norm`.

Cada prompt de sistema traz regras de evidência compartilhadas, instruções específicas do campo, exemplos sintéticos e o JSON schema da resposta. Os valores permitidos, incluindo as 27 UFs, vêm dos contratos Python. Cada resposta é limitada a 512 tokens; requisições com falha são repetidas e auditadas, nunca convertidas em campos vazios em silêncio.

### 4. Veracidade

O verificador executa consultas SQLite parametrizadas e somente leitura sobre as colunas enriquecidas. Ele não chama nenhum LLM.

Quando a busca estruturada não é possível ou não encontra registros, o verificador recorre ao índice FTS5 da base (`documentos_fts`), buscando no texto dos documentos os números extraídos (processo, súmula, artigo e diploma), em `4_veracity_fts.py`. Esse fallback cobre campos que o enriquecimento deixou de preencher; os registros que ele encontra seguem as mesmas regras da tabela abaixo.

| Condição | Resultado | Confiança |
| --- | --- | --- |
| `completa: false`, ou saída de entidades indisponível | `incompleta` (sem consulta) | Probabilidade da etapa 2 |
| Os campos extraídos não formam uma consulta suportada | `incompleta` | Probabilidade da etapa 2 |
| Nem a consulta estruturada nem o FTS encontram registro | `inventada` | 0,75 |
| Um único registro canônico corresponde | `real` com o seu ID canônico | 1,0 |
| Vários registros restam depois de desambiguar por tribunal, UF, ano e relator | `real` com o menor ID entre eles | 0,0 |

### Confiança

A métrica oficial soma um bônus baseado no Brier sobre as citações pareadas, em que o alvo é 1 quando a classe prevista (e, para `real`, o ID canônico) está correta. Por isso a confiança estima se a **classe final** está certa, e não se o trecho foi bem extraído, e é definida onde essa classe é decidida:

- `real` e `inventada`: constantes definidas pela etapa 4 conforme o resultado da consulta. Na amostra de desenvolvimento, `real` com um único registro sempre esteve correta, e cerca de 75% das `inventada` estavam corretas.
- `incompleta`: a probabilidade, dada pela etapa 2, de a citação estar incompleta, a partir dos log-probabilities do modelo. Numa comparação na amostra de desenvolvimento, ela pontuou o mesmo que pedir ao modelo que escrevesse um valor de confiança, que foi sempre 1,0 e não trazia informação.
- Quando a etapa 2 não retorna probabilidade, a confiança é omitida (`-`), o que tira a citação do termo de Brier em vez de penalizá-la.

A materialização usa apenas a confiança da etapa 4.


### Executando as etapas separadamente

```bash
uv run python -m desafio_jusbrasil.database_preprocessing --config configs/final_database_preprocessing.yaml
uv run desafio-jusbrasil --config configs/final_pipeline.yaml
```

A CLI de enriquecimento aceita `--input`, `--output` e `--audit-dir`, e `--materializar` monta a base a partir dos resultados existentes, sem chamar o modelo. A CLI do pipeline aceita `--input-dir`, `--workdir` e `--database`. As etapas também podem rodar uma de cada vez, cada uma consumindo os checkpoints da anterior:

```bash
uv run python -m desafio_jusbrasil.1_extractor --config configs/final_pipeline.yaml
uv run python -m desafio_jusbrasil.2_completeness --config configs/final_pipeline.yaml
uv run python -m desafio_jusbrasil.3_entities --config configs/final_pipeline.yaml
uv run python -m desafio_jusbrasil.4_veracity --config configs/final_pipeline.yaml
```

Use um `workdir` novo para cada experimento. O último comando também exporta os JSONs finais de predição.

### Avaliação

Gere referências locais por etapa a partir das anotações de desenvolvimento publicadas:

```bash
uv run scripts/generate_stage_golds.py --output outputs/gold
```

As referências de entidades também usam normalização local e correções revisadas; são anotações de desenvolvimento, não um benchmark independente.

```bash
uv run scripts/evaluate_extraction.py outputs/my-run --gold outputs/gold
uv run scripts/evaluate_completeness.py outputs/my-run --gold outputs/gold
uv run scripts/evaluate_entities.py outputs/my-run --gold outputs/gold
uv run scripts/evaluate_veracity.py outputs/my-run --gold outputs/gold
uv run scripts/clear_outputs.py outputs/my-run --gold outputs/gold
uv run python scripts/evaluate_database_preprocessing.py outputs/database-preprocessing/<run>
```

O `clear_outputs.py` lê os resultados salvos, sem chamar modelos, e grava relatórios legíveis em `outputs/clear_outputs/<data-e-hora>/`.

Para pontuar uma execução com a métrica oficial da competição:

```bash
uv run python scripts/json_to_submission.py outputs/my-run/predictions outputs/my-run/submission.csv
uv run python scripts/evaluate.py outputs/my-run/submission.csv
```

Esses comandos criam e pontuam arquivos locais; não enviam nada ao Kaggle.

### Resultados de desenvolvimento

Nos 26 documentos de desenvolvimento, com `google/gemma-4-12B-it-qat-w4a16-ct`, temperatura 0 e a base enriquecida pelo mesmo modelo, a etapa 4 atual (com o fallback FTS) obteve **0,923** na métrica oficial, contra 0,907 sem o fallback. As etapas 1 a 3 vieram de uma execução anterior às regras de confiança atuais, então a confiança das `incompleta` não entrou nessa medida.

Os prompts usam demonstrações sintéticas, mas o corpus de desenvolvimento também orientou a escrita dos prompts. Esses números são resultados de desenvolvimento, não um benchmark em dados não vistos. Os artefatos de execução em `outputs/` são locais e ficam fora do Git.

### Checkpoints e auditorias

Cada etapa grava `<etapa>/<documento_id>/resultado.json` e arquivos de auditoria numerados em `01-extraction`, `02-completeness`, `03-entities` e `04-veracity`; os JSONs finais ficam em `predictions/`. As auditorias registram as requisições, respostas, tentativas e tempos das chamadas ao modelo, e o SQL, os parâmetros, os registros encontrados e a classificação de cada verificação. Os manifestos de cada etapa registram a configuração. Os avaliadores ignoram os manifestos e os arquivos de auditoria numerados.

### Limitações conhecidas

- Citações fragmentadas não são reparadas nem expandidas, e jurisprudência não é buscada só por tribunal, ano e relator; essas referências ficam `incompleta`.
- Quando vários registros correspondem, o menor ID é uma escolha determinística, não uma desambiguação.
- O dicionário de nomes de relatores foi construído a partir da base de desenvolvimento; nomes fora dele não são normalizados.
- O enriquecimento com o modelo de 12B concorda menos com os metadados de referência do que modelos maiores; campos que ele perde podem transformar citações reais em `inventada`.
