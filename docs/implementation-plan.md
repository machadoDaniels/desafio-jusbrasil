# Plano de implementação

## Objetivo

Implementar um pipeline simples, sequencial e reiniciável:

```text
TXT
  → 01-extraction/
  → 02-completeness/
  → 03-veracity/
  → predictions/*.json
  → submission.csv
```

Cada etapa deve:

1. poder ser executada pelo seu próprio arquivo;
2. ler apenas arquivos locais;
3. salvar imediatamente sua saída em um JSON por documento;
4. permitir retomar o pipeline sem repetir etapas anteriores;
5. validar entrada e saída com Pydantic;
6. expor seu comportamento por um `Protocol` pequeno;
7. decompor leitura, inferência, validação e escrita em métodos ou funções independentes.

A arquitetura segue `ideia.excalidraw` e `class-diagram.mmd`, priorizando simplicidade.

## Estrutura de código

```text
src/desafio_jusbrasil/
├── contracts.py
├── extractor.py
├── completeness.py
├── veracity.py
└── orchestrator.py
```

Não haverá `run.py`. Os quatro arquivos de execução terão seu próprio `main()` e poderão ser executados com `python -m`. `contracts.py` conterá os contratos e a leitura da configuração YAML.

## Configuração

Todos os entrypoints leem diretamente o arquivo `pipeline.yaml` da raiz:

```yaml
input_dir: desafio-jusbrasil-bracis-2026/txt
workdir: outputs/run-001
database: desafio-jusbrasil-bracis-2026/desafio1_bracis.db

extractor:
  provider: openai
  model: gpt-4.1-mini
  base_url: null
  temperature: 0
  reasoning_effort: null
  async_requests: true
  max_concurrency: 4

completeness:
  provider: openai
  model: gpt-4.1-mini
  base_url: null
  temperature: 0
  reasoning_effort: null
  async_requests: true
  max_concurrency: 4

veracity:
  provider: openai
  model: gpt-4.1-mini
  base_url: null
  temperature: 0
  reasoning_effort: null
  async_requests: true
  max_concurrency: 4
```

A chave OpenAI não fica no YAML; ela vem da variável de ambiente `OPENAI_API_KEY`.

## Estrutura dos artefatos

Todas as etapas escrevem no diretório de trabalho definido por `workdir` no YAML:

```text
outputs/run-001/
├── 01-extraction/
├── 02-completeness/
├── 03-veracity/
├── predictions/
│   ├── gen_n1_001.json
│   ├── gen_n1_002.json
│   └── ...
└── submission.csv
```

O diretório de trabalho é explícito para não misturar dados gerados com os TXT originais. “Localmente” significa que a comunicação entre etapas acontece por esses arquivos, não por estado em memória ou por um serviço externo.

## Regra para os checkpoints

- Cada documento possui um arquivo `<documento_id>.json`.
- Cada arquivo é um JSON válido e independente, escrito em UTF-8.
- Os arquivos são lidos em ordem de `documento_id`.
- Cada etapa valida o JSON recebido antes de processá-lo.

Manter um documento por arquivo simplifica retomada, inspeção e processamento isolado.

---

# 0. `contracts.py`

## Responsabilidade

Centralizar todos os contratos compartilhados e carregar a configuração YAML, sem lógica de negócio, chamadas de LLM ou consultas SQL.

## Organização interna

```text
contracts.py
├── configuração
│   └── PipelineConfig
├── enums
│   ├── TipoCitacao
│   └── Classificacao
├── modelos básicos
│   ├── LoteCandidatos
│   ├── CandidatoCitacao
│   ├── ConsultaCanonica
│   ├── RegistroCanonico
│   ├── ResultadoCompletude
│   ├── ResultadoVeracidade
│   ├── Resolucao
│   └── Predicao
├── checkpoints
│   ├── DocumentoExtraido
│   ├── CandidatoAnalisado
│   ├── DocumentoCompletude
│   ├── CandidatoClassificado
│   ├── DocumentoClassificado
│   └── DocumentoPredito
└── interfaces
    ├── ExtratorCandidatos
    ├── ClassificadorCompletude
    └── ClassificadorVeracidade
```

## Regras

- Todos os modelos persistidos herdam de `pydantic.BaseModel`.
- As interfaces usam `typing.Protocol`, sem herança obrigatória nas implementações.
- Validações estruturais ficam nos próprios modelos Pydantic.
- Regras que dependem de LLM e SQLite ficam fora deste arquivo.
- A única operação de filesystem permitida é `PipelineConfig.from_yaml()`.
- `contracts.py` não importa nenhum dos outros módulos do projeto.
- `extractor.py`, `completeness.py`, `veracity.py` e `orchestrator.py` importam contratos somente deste arquivo.

As interfaces ficam juntas ao final do arquivo:

```python
class ExtratorCandidatos(Protocol):
    def extrair(self, texto: str) -> list[CandidatoCitacao]:
        ...


class ClassificadorCompletude(Protocol):
    def classificar(
        self,
        candidato: CandidatoCitacao,
        contexto: str,
    ) -> ResultadoCompletude:
        ...


class ClassificadorVeracidade(Protocol):
    def classificar(
        self,
        consulta: ConsultaCanonica,
    ) -> ResultadoVeracidade:
        ...
```

Os trechos de contratos apresentados nas seções seguintes representam definições que pertencem a `contracts.py`, não aos arquivos das etapas.

---

# 1. `extractor.py`

## Responsabilidade

Ler os documentos TXT, extrair candidatos a citações e salvar `01-extraction/`.

## Entrada

```text
desafio-jusbrasil-bracis-2026/txt/*.txt
```

## Saída

```text
<workdir>/01-extraction/
```

## Contratos utilizados

Definidos em `contracts.py`:

```python
class TipoCitacao(StrEnum):
    JURISPRUDENCIA = "jurisprudencia"
    LEI = "lei"


class CandidatoCitacaoRequest(BaseModel):
    tipo: TipoCitacao
    trecho: str
    confianca_extracao: float | None = None


class CandidatoCitacao(CandidatoCitacaoRequest):
    inicio: int
    fim: int


class DocumentoExtraido(BaseModel):
    documento_id: str
    texto: str
    candidatos: list[CandidatoCitacao]
```

O texto é mantido no checkpoint para que as etapas posteriores sejam independentes da pasta original.

## Implementação concreta

Implementa estruturalmente o `Protocol` `ExtratorCandidatos`, definido em `contracts.py`:

```python
class AgenteExtrator:
    def extrair(self, texto: str) -> list[CandidatoCitacao]:
        candidatos = self._consultar_modelo(texto)
        candidatos_com_spans = self._adicionar_spans(texto, candidatos)
        return self._remover_duplicatas(candidatos_com_spans)

    def _consultar_modelo(self, texto: str) -> list[CandidatoCitacaoRequest]:
        ...

    def _adicionar_spans(
        self,
        texto: str,
        candidatos: list[CandidatoCitacaoRequest],
    ) -> list[CandidatoCitacao]:
        ...

    def _remover_duplicatas(
        self,
        candidatos: list[CandidatoCitacao],
    ) -> list[CandidatoCitacao]:
        ...
```

`ExtratorCandidatos` descreve somente o comportamento consumido pelo restante do sistema. `AgenteExtrator` o satisfaz estruturalmente, sem herança obrigatória.

## Implementação

- Usar diretamente `OpenAI.chat.completions.parse(...)` para obter candidatos estruturados com Pydantic.
- Manter `extrair()` e `extrair_async()` na mesma classe, compartilhando o processamento local.
- Cada etapa disponibiliza funções síncrona e assíncrona, selecionadas por seu bloco no YAML quando executada independentemente.
- O orquestrador permanece totalmente síncrono e ignora `async_requests`.
- Limitar chamadas concorrentes com `max_concurrency`.
- Pedir ao modelo somente o trecho literal, o tipo e a confiança opcional.
- Não expor `inicio` nem `fim` no schema enviado ao modelo.
- Localizar todas as ocorrências exatas do trecho e calcular os offsets no código local.
- Validar:
  - `0 <= inicio < fim <= len(texto)`;
  - `texto[inicio:fim] == trecho`;
  - ausência de duplicatas com IoU maior ou igual a `0.5`.
- Ordenar candidatos por `inicio`.
- Não modificar o texto original.

## CLI independente

```bash
uv run python -m desafio_jusbrasil.extractor
```

A etapa lê `input_dir` e grava `workdir/01-extraction/`.

## Critério de aceite

- Todos os TXT produzem exatamente um arquivo JSON.
- Cada trecho corresponde exatamente ao span no texto salvo.
- O arquivo pode ser lido e validado sem acesso aos TXT originais.

---

# 2. `completeness.py`

## Responsabilidade

Ler candidatos extraídos, decidir se cada citação permite uma consulta e salvar `02-completeness/`.

## Entrada

```text
<workdir>/01-extraction/
```

## Saída

```text
<workdir>/02-completeness/
```

## Contratos utilizados

Definidos em `contracts.py`:

```python
class ConsultaCanonica(BaseModel):
    tipo: TipoCitacao
    classe_processual: str | None = None
    numero: str | None = None
    tribunal: str | None = None
    uf: str | None = None
    ano: int | None = None
    relator: str | None = None
    dispositivo: str | None = None


class ResultadoCompletude(BaseModel):
    completa: bool
    consulta: ConsultaCanonica | None = None
    justificativa: str
    confianca: float | None = None


class CandidatoAnalisado(BaseModel):
    candidato: CandidatoCitacao
    completude: ResultadoCompletude


class DocumentoCompletude(BaseModel):
    documento_id: str
    texto: str
    candidatos: list[CandidatoAnalisado]
```

## Implementação concreta

Implementa estruturalmente o `Protocol` `ClassificadorCompletude`, definido em `contracts.py`:

```python
class AgenteCompletude:
    def classificar(
        self,
        candidato: CandidatoCitacao,
        contexto: str,
    ) -> ResultadoCompletude:
        return self._consultar_modelo(candidato, contexto)

    def _consultar_modelo(
        self,
        candidato: CandidatoCitacao,
        contexto: str,
    ) -> ResultadoCompletude:
        ...
```

## Implementação

- Ler `DocumentoExtraido` linha por linha.
- Para cada candidato, obter uma janela simples:

```python
contexto = texto[max(0, inicio - 300):min(len(texto), fim + 300)]
```

- Usar diretamente `OpenAI.chat.completions.parse(...)` com `ResultadoCompletude`.
- Extrair somente campos sustentados pelo trecho ou pelo contexto.
- Manter `consulta=None` quando `completa=False`.
- Preservar candidato, texto e `documento_id` no checkpoint seguinte.

## CLI independente

```bash
uv run python -m desafio_jusbrasil.completeness
```

A etapa lê `workdir/01-extraction/` e grava `workdir/02-completeness/`.

## Critério de aceite

- O arquivo pode ser gerado sem executar novamente o extrator.
- Cada candidato de entrada aparece exatamente uma vez na saída.
- `completa=True` exige uma consulta.
- Campos ausentes não são inventados.

---

# 3. `veracity.py`

## Responsabilidade

Ler o resultado de completude, consultar a SQLite e salvar as classificações em `03-veracity/`.

## Entrada

```text
<workdir>/02-completeness/
```

## Saída

```text
<workdir>/03-veracity/
```

## Contratos utilizados

Definidos em `contracts.py`:

```python
class Classificacao(StrEnum):
    REAL = "real"
    INVENTADA = "inventada"
    INCOMPLETA = "incompleta"


class RegistroCanonico(BaseModel):
    documento_id: str
    id_canonico: int
    tribunal: str | None = None
    ano: int | None = None
    relator: str | None = None
    natureza: str
    tipo: str
    texto: str


class ResultadoVeracidade(BaseModel):
    classificacao: Classificacao
    id_canonico: int | None = None
    justificativa: str
    confianca: float | None = None


class CandidatoClassificado(BaseModel):
    candidato: CandidatoCitacao
    completude: ResultadoCompletude
    veracidade: ResultadoVeracidade


class DocumentoClassificado(BaseModel):
    documento_id: str
    texto: str
    candidatos: list[CandidatoClassificado]
```

## Implementação concreta

Implementa estruturalmente o `Protocol` `ClassificadorVeracidade`, definido em `contracts.py`:

```python
class AgenteVeracidade:
    def __init__(self, modelo: BaseChatModel, database: Path) -> None:
        self._database = database
        ferramenta = StructuredTool.from_function(
            func=self._buscar_para_agente,
            args_schema=ConsultaCanonica,
        )
        self._agent = create_agent(
            model=modelo,
            tools=[ferramenta],
            response_format=ResultadoVeracidade,
        )

    def classificar(
        self,
        consulta: ConsultaCanonica,
    ) -> ResultadoVeracidade:
        ...

    def buscar(
        self,
        consulta: ConsultaCanonica,
    ) -> list[RegistroCanonico]:
        termos = self._montar_consulta_fts(consulta)
        return self._executar_consulta(termos, consulta.tipo)

    def _buscar_para_agente(self, ...) -> str:
        ...

    def _montar_consulta_fts(self, consulta: ConsultaCanonica) -> str:
        ...

    def _executar_consulta(
        self,
        termos: str,
        tipo: TipoCitacao,
    ) -> list[RegistroCanonico]:
        ...
```

A interface expõe apenas `classificar()`. `buscar()` permanece público por ser útil para diagnóstico e testes, mas é detalhe da implementação concreta.

## Implementação

A busca usa `sqlite3`, mas a decisão é feita por um agente LangChain com uma única ferramenta: `buscar_base_canonica`. Para `provider: gemini`, usa-se a integração nativa `ChatGoogleGenerativeAI`, que preserva thought signatures durante o loop de ferramentas; os demais endpoints OpenAI-compatible usam `ChatOpenAI`.

Para cada candidato:

- se `completa=False`, produzir diretamente `incompleta`, sem invocar o agente;
- se `completa=True`, o agente é obrigado a chamar a ferramenta SQLite;
- o agente recebe no máximo dez registros e decide entre `real`, `inventada` e `incompleta`;
- a resposta é validada como `ResultadoVeracidade`;
- uma execução sem chamada da ferramenta falha explicitamente.

Para `real`, `id_canonico` deve ser `documentos.id`, não `documento_id`.

## CLI independente

```bash
uv run python -m desafio_jusbrasil.veracity
```

A etapa lê `workdir/02-completeness/` e grava `workdir/03-veracity/`.

## Critério de aceite

- A etapa usa um agente LangChain e uma única ferramenta SQLite.
- A etapa roda sem executar extração ou completude novamente.
- O agente não pode responder sem consultar a ferramenta.
- Toda predição `real` contém um `id_canonico` válido.

---

# 4. `orchestrator.py`

## Responsabilidade

Oferecer duas operações:

1. executar o pipeline completo, chamando as funções públicas das três etapas;
2. materializar os JSONs finais a partir de `03-veracity/`.

O orquestrador não deve duplicar a lógica interna dos agentes.

## Saída final

```text
<workdir>/predictions/<documento_id>.json
```

## Contratos utilizados

Definidos em `contracts.py`:

```python
class Resolucao(BaseModel):
    id_canonico: int | None = None


class Predicao(BaseModel):
    inicio: int
    fim: int
    trecho: str
    tipo: TipoCitacao
    classificacao: Classificacao
    resolucao: Resolucao | None = None
    confianca: float | None = None


class DocumentoPredito(BaseModel):
    documento_id: str
    citacoes: list[Predicao]
```

A estrutura deve seguir o contrato consumido por `desafio-jusbrasil-bracis-2026/json_to_submission.py`.

## Classe

```python
class Orquestrador:
    def __init__(
        self,
        extrator: ExtratorCandidatos,
        completude: ClassificadorCompletude,
        veracidade: ClassificadorVeracidade,
    ) -> None:
        ...

    def executar(self, pasta_txt: Path, workdir: Path) -> None:
        executar_extracao(...)
        executar_completude(...)
        executar_veracidade(...)
        self.materializar(...)

    def materializar(
        self,
        entrada: Path,
        pasta_saida: Path,
    ) -> None:
        ...

    def _ler_classificados(
        self,
        entrada: Path,
    ) -> Iterator[DocumentoClassificado]:
        ...

    def _montar_predicao(
        self,
        documento: DocumentoClassificado,
    ) -> DocumentoPredito:
        ...
```

Os métodos públicos representam casos de uso. Os métodos privados isolam coordenação, transformação e I/O sem criar classes adicionais.

## Execução completa

```bash
uv run python -m desafio_jusbrasil.orchestrator
```

Essa operação cria, em ordem:

```text
01-extraction/
02-completeness/
03-veracity/
predictions/*.json
```

## Materialização

`veracity.materializar()` permite recriar os JSONs finais diretamente de `03-veracity/`, sem repetir as etapas anteriores.

## Conversão e avaliação local

```bash
uv run scripts/evaluate.py
```

O script lê `outputs/run-001/submission.csv`, agrega o `goldenset.csv` e executa a métrica oficial. Também aceita outro CSV explicitamente:

```bash
uv run scripts/evaluate.py outputs/oracle-submission.csv
```

A conversão dos JSONs continua separada:

```bash
uv run python desafio-jusbrasil-bracis-2026/json_to_submission.py \
  outputs/run-001/predictions \
  outputs/run-001/submission.csv
```

Os checkpoints gold são regenerados do `goldenset_offsets.csv` com:

```bash
uv run scripts/generate_stage_golds.py
```

Cada checkpoint também possui um avaliador independente:

```bash
uv run scripts/evaluate_extraction.py
uv run scripts/evaluate_completeness.py
uv run scripts/evaluate_veracity.py
```

---

# API interna mínima

Cada módulo executável deve expor uma função de arquivo além da classe concreta. As interfaces usadas nas assinaturas vêm de `contracts.py`:

```python
# extractor.py
def executar_extracao(
    input_dir: Path,
    output_file: Path,
    extrator: ExtratorCandidatos,
) -> None: ...

# completeness.py
def executar_completude(
    input_file: Path,
    output_file: Path,
    classificador: ClassificadorCompletude,
) -> None: ...

# veracity.py
def executar_veracidade(
    input_file: Path,
    output_file: Path,
    classificador: ClassificadorVeracidade,
) -> None: ...
```

A função de arquivo cuida apenas de ler, iterar e escrever. A classe cuida da regra de uma unidade. O `main()` monta a implementação concreta e chama a função de arquivo. O orquestrador faz a mesma composição diretamente, sem iniciar subprocessos.

Todos os módulos terminam com:

```python
if __name__ == "__main__":
    main()
```

---

# Dependências

Adicionar ao `pyproject.toml`:

```toml
dependencies = [
    "kaggle>=2.2.4",
    "langchain>=1.0",
    "langchain-openai>=1.0",
    "openai>=2.0",
    "pydantic>=2.0",
    "pyyaml>=6.0",
]
```

Não adicionar ORM, framework de CLI ou biblioteca específica para checkpoints. `json`, `pathlib`, `sqlite3` e PyYAML são suficientes.

A chave deve vir do ambiente. Para usar Gemini, configure `base_url` no YAML e forneça a chave Gemini em `OPENAI_API_KEY`:

```bash
OPENAI_API_KEY="$(security find-generic-password \
  -s agent/openai-api-key -w)" \
uv run python -m desafio_jusbrasil.orchestrator
```

---

# Ordem de implementação

## Etapa 1 — Contratos centrais

Implementar `contracts.py` com enums, modelos Pydantic, modelos dos checkpoints e `Protocol`s.

**Aceite:**

- o módulo importa sem depender dos quatro módulos executáveis;
- exemplos válidos de cada checkpoint passam na validação;
- spans inválidos e predições `real` sem `id_canonico` são rejeitados;
- implementações fake satisfazem os `Protocol`s no type checker.

## Etapa 2 — Utilitários locais de JSON

Implementar pequenas funções privadas para ler e escrever um JSON por documento com validação Pydantic.

Não criar um sexto arquivo de utilidades; manter as poucas linhas necessárias em cada módulo.

**Aceite:** cada documento gera um checkpoint JSON independente.

## Etapa 3 — Extração isolada

Implementar `extractor.py`, sua CLI e `01-extraction/`.

**Aceite:** todos os TXT produzem registros válidos e spans exatos.

## Etapa 4 — Completude isolada

Implementar `completeness.py`, sua CLI e `02-completeness/`.

**Aceite:** a etapa funciona usando apenas os JSONs da extração.

## Etapa 5 — Veracidade isolada

Implementar `veracity.py`, sua CLI e `03-veracity/`.

**Aceite:** a etapa funciona usando apenas os JSONs de completude e a SQLite.

## Etapa 6 — Materialização

Implementar em `veracity.py` a conversão de `03-veracity/` para os JSONs finais.

**Aceite:** `json_to_submission.py` aceita a pasta produzida.

## Etapa 7 — Orquestração completa

Implementar o subcomando `run` chamando as três funções de arquivo e a materialização.

**Aceite:** os 26 TXT geram:

- 26 arquivos JSON em cada checkpoint;
- 26 JSONs finais;
- 26 linhas no CSV de submissão;
- um CSV aceito pela métrica local.

## Etapa 8 — Baseline

Executar a métrica e registrar:

- score final;
- score N1;
- score N2;
- F1 por classe;
- erros de span;
- erros de classificação;
- erros de resolução.

Na primeira versão, omitir `confianca` da predição final. Confiança não calibrada pode piorar o Brier score.

---

# Testes mínimos

Um único arquivo futuro, `tests/test_pipeline.py`, é suficiente.

1. `contracts.py` não possui dependência dos módulos executáveis.
2. Fakes simples satisfazem cada `Protocol`.
3. Pydantic rejeita spans inválidos.
4. `trecho == texto[inicio:fim]`.
5. `real` exige `id_canonico`.
6. O extrator gera JSONs legíveis pela completude.
7. A completude gera JSONs legíveis pela veracidade.
8. A ferramenta SQLite do agente de veracidade recupera registros canônicos.
9. Zero resultados vira `inventada`.
10. Ambiguidade vira `incompleta`.
11. Um checkpoint existente permite retomar da etapa seguinte.
12. `03-veracity/` gera JSONs aceitos pelo conversor.
13. Os 26 documentos geram uma submissão aceita pela métrica.

---

# Fora do escopo inicial

Não implementar ainda:

- `ABC` e hierarquias de herança;
- mais de um arquivo de contratos ou interfaces;
- `AgentExecutor` ou ReAct;
- memória de agente;
- seleção dinâmica de ferramentas;
- banco vetorial ou embeddings;
- ORM;
- processamento paralelo;
- cache além dos próprios checkpoints;
- retries elaborados;
- calibração aprendida;
- configuração YAML;
- prompts em arquivos separados;
- serviço web ou fila externa.

A primeira entrega termina quando cada etapa roda isoladamente e o pipeline completo produz um CSV válido a partir dos quatro módulos executáveis e do arquivo central de contratos.
