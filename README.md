# Linguagem Simples Graph

Workflow para analisar, simplificar e avaliar textos português brasileiro. O sistema gera três versões de cada documento, direcionadas a públicos diferentes, e revisa automaticamente versões rejeitadas por um avaliador, utilizando modelos de linguagem.

## Como o workflow funciona

Para cada documento do corpus, o sistema executa estas etapas:

1. **Análise linguística e de segurança**: identifica dificuldades de compreensão, informações que devem ser preservadas e conteúdo que não pode ser simplificado. Caso o texto de entrada não possa ser simplificado por conter um conteúdo com discurso de ódio, o sistema retornará o motivo.
2. **Simplificação em paralelo**: gera 4 versões de simplificação para cada documento.
3. **Avaliação**: verifica preservação semântica, legibilidade, coerência e adequação ao público.
4. **Revisão**: uma versão rejeitada volta ao simplificador com o feedback do avaliador, até atingir aprovação ou o limite configurado.
5. **Persistência**: salva resultados, tentativas, uso de tokens, manifesto e metadados da execução.

Os ramos fornecidos nas configurações de exemplo são:

| Ramo | Público-alvo | Intensidade |
|---|---|---|
| `simple` | Estudantes, acadêmicos e profissionais da área | Leve |
| `moderate` | Jornalistas e profissionais de comunicação | Moderada |
| `aggressive` | Público geral | Forte |

Público-alvo e intensidade são configurações independentes e podem ser alterados sem modificar os nós do workflow.

## Requisitos

- Python 3.12 ou superior
- [uv](https://docs.astral.sh/uv/)
- Credencial do provider escolhido, quando a execução usar uma API externa
- NILC-Metrix, somente para o cálculo opcional das métricas externas

Instale as dependências com:

```sh
uv sync
```

## Estrutura do projeto

```text
core/
  schemas.py                 Contratos Pydantic versionados
  simplificacao.py           Construção e execução do LangGraph
  graph.png                  Visualização do workflow
  workflow/state.py          Estado interno do LangGraph
config.py                    Carregamento YAML, manifesto e retomada
configs/
  openrouter.yaml            Configuração para OpenRouter
  hpc.yaml                   Modelo de configuração para ambiente HPC
prompts/
  analisador.txt
  simplificador.txt
  avaliador.txt
input/
  corpus.json
tests/                       Testes sem consumo de API
main.py                      Interface de linha de comando
compute_metrics.py           Cálculo de métricas sobre os resultados
```

## Credenciais

As credenciais devem ser fornecidas pelo ambiente. Elas não são aceitas nos arquivos YAML, não aparecem na configuração efetiva e não são gravadas no manifesto ou nos logs.

Exemplo de `.env` para OpenRouter:

```dotenv
OPENROUTER_API_KEY=seu_token
```

O arquivo `.env` é ignorado pelo Git. Para carregá-lo com `uv`:

```sh
uv run --env-file .env main.py --config configs/openrouter.yaml
```

## Configuração

Os arquivos YAML definem:

- caminho do corpus;
- campos usados como identificador e texto;
- diretório de execução;
- nomes dos arquivos de resultado;
- limite de revisões;
- público-alvo e intensidade de cada ramo;
- caminhos dos prompts;
- modelo, provider, URL, timeout, parâmetros de geração e tentativas técnicas de cada papel.

Há três papéis de modelo:

- `analyzer`: análise linguística e de segurança;
- `simplifier`: geração e revisão das versões simplificadas;
- `evaluator`: auditoria da qualidade das versões.

Exemplo para OpenRouter:

```yaml
models:
  analyzer:
    name: qwen/qwen3-32b
    provider: openrouter
    base_url: https://openrouter.ai/api/v1
    api_key_env: OPENROUTER_API_KEY
    request_timeout_seconds: 60
    technical_retries: 2
    retry_backoff_seconds: 0.5
    temperature: 0.1
    max_tokens: 4096
    top_p: 1.0
    supports_structured_output: false
  simplifier:
    name: qwen/qwen3-32b
    provider: openrouter
    base_url: https://openrouter.ai/api/v1
    api_key_env: OPENROUTER_API_KEY
    request_timeout_seconds: 60
    technical_retries: 2
    retry_backoff_seconds: 0.5
    temperature: 0.3
    max_tokens: 4096
    top_p: 1.0
    supports_structured_output: false
  evaluator:
    name: qwen/qwen3-32b
    provider: openrouter
    base_url: https://openrouter.ai/api/v1
    api_key_env: OPENROUTER_API_KEY
    request_timeout_seconds: 60
    technical_retries: 2
    retry_backoff_seconds: 0.5
    temperature: 0.0
    max_tokens: 4096
    top_p: 1.0
    supports_structured_output: false
```

`api_key_env` contém somente o nome da variável de ambiente, nunca o valor da credencial. Chaves desconhecidas e valores inválidos são rejeitados antes de qualquer chamada ao modelo.

### Providers de modelo

O workflow não instancia modelos diretamente. Uma fábrica cria um cliente para cada papel antes do processamento do corpus, e todos os nós usam a mesma interface normalizada.

Providers disponíveis:

- `openrouter`: usa a API OpenAI-compatible do OpenRouter e exige uma credencial indicada por `api_key_env`;
- `openai_compatible`: usa um endpoint local ou remoto compatível com a API OpenAI; a autenticação é opcional;
- `simulated`: fornece respostas programadas e falhas controladas somente para testes.

Para um endpoint local, use por exemplo:

```yaml
models:
  analyzer:
    name: modelo-local
    provider: openai_compatible
    base_url: http://localhost:8000/v1
    api_key_env: null
    request_timeout_seconds: 120
    technical_retries: 1
    retry_backoff_seconds: 1.0
    temperature: 0.1
    max_tokens: 4096
    top_p: 1.0
    supports_structured_output: false
```

Repita a configuração para `simplifier` e `evaluator`, ajustando modelo e parâmetros quando necessário.

`technical_retries` controla somente repetições de falhas temporárias do endpoint, como timeout, indisponibilidade e limite de taxa. Falhas de autenticação ou configuração não são repetidas. As tentativas internas do SDK permanecem desativadas para evitar multiplicação de chamadas.

`workflow.max_revisions` tem outra finalidade: controla quantas versões um ramo pode gerar quando o avaliador rejeita a qualidade da simplificação.

Quando `supports_structured_output` é `true`, o adaptador solicita saída estruturada nativa ao backend. Quando ela está desabilitada ou indisponível, o sistema faz parsing do JSON textual e valida o resultado com o mesmo schema Pydantic.

### Resolução de caminhos

Caminhos relativos no YAML e na CLI são resolvidos a partir da raiz do projeto. O resultado não depende do diretório a partir do qual o comando foi iniciado.

### Precedência

A configuração efetiva usa esta precedência, da maior para a menor:

1. opções da CLI;
2. variáveis de ambiente `LSG_*`;
3. arquivo YAML informado por `--config`.

### Variáveis de ambiente

Variáveis gerais:

- `LSG_MODEL`
- `LSG_CORPUS_PATH`
- `LSG_EXECUTION_DIR`
- `LSG_MAX_REVISIONS`
- `LSG_ANALYZER_PROMPT`
- `LSG_SIMPLIFIER_PROMPT`
- `LSG_EVALUATOR_PROMPT`

Variáveis por papel:

- `LSG_ANALYZER_MODEL`
- `LSG_SIMPLIFIER_MODEL`
- `LSG_EVALUATOR_MODEL`
- `LSG_<PAPEL>_PROVIDER`
- `LSG_<PAPEL>_BASE_URL`
- `LSG_<PAPEL>_API_KEY_ENV`
- `LSG_<PAPEL>_TIMEOUT_SECONDS`
- `LSG_<PAPEL>_TEMPERATURE`
- `LSG_<PAPEL>_TECHNICAL_RETRIES`
- `LSG_<PAPEL>_RETRY_BACKOFF_SECONDS`
- `LSG_<PAPEL>_MAX_TOKENS`
- `LSG_<PAPEL>_TOP_P`
- `LSG_<PAPEL>_SUPPORTS_STRUCTURED_OUTPUT`

`LSG_<PAPEL>_MAX_RETRIES` continua aceito como alias legado de `LSG_<PAPEL>_TECHNICAL_RETRIES`.

Variáveis de público:

- `LSG_SIMPLE_TARGET_AUDIENCE`
- `LSG_MODERATE_TARGET_AUDIENCE`
- `LSG_AGGRESSIVE_TARGET_AUDIENCE`

## Execução

### Usar os modelos definidos no YAML

```sh
uv run --env-file .env main.py --config configs/openrouter.yaml
```

### Sobrescrever o modelo dos três papéis

O argumento `--model` aplica o mesmo modelo ao analisador, simplificador e avaliador:

```sh
uv run --env-file .env main.py \
  --config configs/openrouter.yaml \
  --model qwen/qwen3-32b
```

### Usar modelos diferentes por papel

Defina os modelos diretamente no YAML ou use variáveis de ambiente:

```sh
LSG_ANALYZER_MODEL=modelo-analisador \
LSG_SIMPLIFIER_MODEL=modelo-simplificador \
LSG_EVALUATOR_MODEL=modelo-avaliador \
uv run --env-file .env main.py --config configs/openrouter.yaml
```

### Outras opções

```sh
uv run main.py --help
```

As opções incluem:

- `--config`: arquivo YAML;
- `--model`: modelo para os três papéis;
- `--corpus`: caminho alternativo para o corpus;
- `--output-dir`: diretório-base das execuções;
- `--max-revisions`: limite de tentativas por ramo;
- `--run-id`: identificador de uma execução;
- `--resume`: retomada de uma execução existente.

## Corpus

Cada item deve conter pelo menos um identificador e um texto. Os nomes desses campos são definidos no YAML.

Exemplo com os campos padrão:

```json
[
  {
    "id": "DOC-001",
    "juridico": "Texto que será simplificado."
  }
]
```

O sistema preserva o `id` como `document_id` e calcula um hash SHA-256 do conteúdo. Entradas malformadas falham antes da inicialização dos modelos.

## Resultados e manifesto

Cada execução recebe um `run_id` e cria o diretório:

```text
<execution.directory>/<run_id>/
```

Arquivos gerados:

- `manifest.json`: configuração efetiva, commit Git, hash do corpus, hashes dos prompts, versões dos schemas e dependências;
- `results.jsonl`: resultado validado de cada documento;
- `metadata_per_text.csv`: tempo, tokens e chamadas por documento;
- `summary.json`: totais acumulados da execução.

Cada ramo registra:

- público-alvo e intensidade;
- status de execução;
- veredito de qualidade;
- motivo de término;
- texto final;
- tentativas;
- identificadores das chamadas;
- feedbacks e erros estruturados.

Uma execução técnica pode terminar corretamente e ainda produzir uma versão rejeitada por qualidade ao atingir o limite de revisões.

Os adaptadores normalizam o conteúdo, o identificador da requisição, o motivo de término e o uso de tokens. Quando o backend não informa uso, os campos correspondentes são persistidos como `null`; eles não são convertidos em zero medido. O CSV deixa esses campos vazios e o resumo mantém `null` se qualquer documento tiver uso desconhecido.

## Retomada

Use o mesmo `run_id` e a mesma configuração efetiva:

```sh
uv run --env-file .env main.py \
  --config configs/openrouter.yaml \
  --run-id UUID_DA_EXECUCAO \
  --resume
```

Repita na retomada os overrides usados na execução inicial, incluindo `--model`, quando aplicável.

A retomada é recusada se houver mudança incompatível em:

- configuração efetiva;
- corpus;
- prompts;
- versões dos schemas.

Documentos já presentes em `results.jsonl` não são processados novamente. O hash de cada documento também é conferido.

## Testes

A suíte usa o provider simulado e implementações locais falsas dos endpoints. Ela não acessa APIs externas.

Execute:

```sh
uv run pytest
```

Os testes verificam:

- importação das dependências de runtime;
- validação dos schemas;
- precedência da configuração;
- rejeição de chaves desconhecidas;
- ausência de credenciais no manifesto;
- compatibilidade de retomada;
- fábrica de clientes, parâmetros explícitos e modelos por papel;
- retries apenas para falhas transitórias;
- falhas de autenticação sem repetição;
- parsing e validação de saída estruturada;
- execução dos mesmos nós com OpenRouter, endpoint local e simulador;
- preservação de uso de tokens desconhecido;
- aprovação e limite de revisões;
- conclusão parcial;
- bloqueio de conteúdo antes da simplificação;
- leitura de resultados JSONL pelas métricas;
- disponibilidade das opções da CLI.

## Métricas

`compute_metrics.py` aceita listas JSON legadas e o formato JSONL produzido pelo workflow. Para computar as métricas, é necessária a instalação do pacote de métricas [NILC-Metrix](https://doi.org/10.1007/s10579-023-09693-w), disponível [neste repositório do GitHub](https://github.com/sidleal/nilcmetrix).

Exemplo:

```sh
uv run compute_metrics.py \
  --input output/runs/UUID_DA_EXECUCAO/results.jsonl \
  --output output/runs/UUID_DA_EXECUCAO/metrics.jsonl \
  --nilc-metrix-folder /caminho/para/nilc-metrix
```

Consulte os argumentos disponíveis:

```sh
uv run compute_metrics.py --help
```
