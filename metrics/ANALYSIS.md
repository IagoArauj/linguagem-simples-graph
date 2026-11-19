# Análise em lote das execuções

`metrics.analyze` transforma os arquivos já calculados em tabelas e gráficos PNG, sem chamar APIs, executar NILC-Metrix ou carregar modelos BERT. Reproduz as famílias de análises do notebook `Análise.ipynb`, corrigindo o pareamento por posição e a confusão entre métricas humanas e de IA.

## Instalação

```sh
uv sync --extra analysis --python 3.12
uv run --extra analysis --python 3.12 python -m metrics.analyze --help
```

O extra `analysis` instala pandas, NumPy, Matplotlib e SciPy. Não exige o extra `metrics` nem PyTorch. Para manter ambos no mesmo ambiente, use `uv sync --extra analysis --extra metrics --python 3.12`.

## Execução

```sh
uv run --extra analysis --python 3.12 python -m metrics.analyze \
  --run-dir \
    output/porsimplessent-runs/73734703-0884-4d5f-9a9e-7f29e07ad877 \
    output/porsimplessent-runs/6eea9265-437d-4e9c-bc16-dcd0a94cbcb8
```

`--run-dir` aceita uma ou várias pastas. Use caminhos explícitos para incluir somente as execuções desejadas. O nome exibido é o modelo do simplificador registrado no manifesto. Se o mesmo modelo aparecer em várias runs, o `run_id` diferencia os rótulos; as observações nunca são agregadas silenciosamente entre runs.

Opções:

- `--output-dir output/analyses/comparacao`: destino alternativo, necessariamente uma pasta **nova** dentro de `output/`. Sem essa opção, cria `output/analyses/<timestamp-UTC>_<id>/`.
- `--corpus caminho/corpus.jsonl`: substitui o caminho do manifesto, mas o hash deve coincidir.
- `--corpus-metrics caminho/metrics.jsonl`: substitui o arquivo de métricas do corpus.
- `--approved-only`: filtra candidatas IA por aprovação nas métricas textuais. O padrão inclui todo texto final disponível, inclusive rejeitado ou pertencente a um ramo com falha. O relatório operacional sempre inclui todos os documentos e branches, independentemente desse filtro.
- `--no-plots`: gera somente tabelas e testes estatísticos, mais rapidamente.

Os gráficos são gerados em modo não interativo (Agg), sem abrir janelas. A barra de progresso avança por conjunto: um combinado e um por run. Uma etapa pode levar minutos, especialmente na primeira construção do cache de fontes. Nada é recalculado nos experimentos; os arquivos de origem permanecem intactos.

## Fontes e associação

Cada run fornece `manifest.json` e o arquivo apontado por `effective_config.execution.results_filename`. Os nomes dos campos de corpus e o caminho são lidos do manifesto. Todas as runs da comparação devem usar o mesmo hash de corpus e os mesmos campos de ID/texto.

Fontes opcionais:

| Local | Arquivo | Uso |
|---|---|---|
| Pasta do corpus | `metrics.jsonl` | Métricas NILC do original e humanos |
| Pasta do corpus | `human_bertscore.csv` | BERTScore humano |
| Cada run | `metrics.jsonl` | Métricas NILC da IA |
| Cada run | `semantic_metrics.csv` | BERTScore, ROUGE-L e SARI |
| Cada run | Nome de metadados indicado no manifesto | Tempo, tokens e chamadas |

Comprimentos em caracteres são obtidos dos textos do corpus/resultados, não de índices do DataFrame. Metadados operacionais aparecem com versão `execution`, por documento, sem replicação nos três ramos.

O pareamento usa `document_id`/ID do corpus. Registros NILC legados contendo apenas `item_index` usam a ordem do arquivo original explicitamente, com aviso no manifesto. Retries NILC usam o último registro `success`/`partial`; um erro posterior não apaga uma medição válida. Entradas semanticamente incompatíveis (hash, identidade, configuração BERTScore) causam erro em vez de mistura silenciosa. Quando o formato NILC não registra hash/configuração do cálculo, a rastreabilidade é limitada ao ID e ao SHA-256 do arquivo usado.

Fontes opcionais ausentes geram avisos; análises que dependem delas são omitidas. Nenhuma ausência é convertida em zero. Humanos e original entram uma única vez na análise conjunta.

## Saída

```text
output/analyses/<timestamp>_<id>/
  manifest.json
  values.csv
  means.csv
  means.md
  execution_report.md
  execution_*.csv
  deltas.csv
  consistency.csv
  friedman.csv
  wilcoxon_holm.csv
  combined/
    *.png
  runs/
    001/*.png
    002/*.png
```

- `manifest.json`: criação/fim em UTC, status (`running`, `completed`, `failed`), pastas de entrada, arquivos e SHA-256, modelos por papel, opções, versões das dependências, hashes do código de análise, avisos e lista dos artefatos. Informa também qual run corresponde a cada pasta numerada. Uma interrupção forçada do processo pode deixar status `running`; nesse caso os artefatos são parciais.
- `values.csv`: observações em formato longo para auditoria (`run_id`, `run_name`, `document_id`, `version`, `metric`, `value`).
- `means.md`: tabelas Markdown por métrica com modelo, run, nível, **n, média, mediana e desvio-padrão**; grupos sem observações aparecem com n=0 e traços.
- `means.csv`: resumo também inclui quartis. Se houver uma única observação, o desvio-padrão amostral fica ausente.
- `deltas.csv`: resumo pareado da variação simplificado − original; a média do original usa exatamente os IDs disponíveis no par.
- `consistency.csv`: contagens e percentuais de mudanças favoráveis/desfavoráveis/sem mudança, conforme hipóteses direcionais declaradas no manifesto.
- `friedman.csv` e `wilcoxon_holm.csv`: inferência exploratória por run para Flesch e Gunning-Fog (`gunning_fox`).

### Recursos e falhas de execução

Toda nova execução de **`metrics.analyze`** também gera `execution_report.md`, com tabelas por run e uma comparação conjunta:

- Médias de `llm_calls`, `input_tokens`, `output_tokens`, `total_tokens`, `elapsed_time`, `retry_wait_seconds` e `elapsed_without_backoff`, com **n, ausentes, cobertura e soma conhecida**. Recursos são por documento, não replicados por branch.
- Status e motivos finais dos documentos, incluindo `partial` e `failed`.
- Status, aprovação/rejeição, motivos finais e limite de revisões por branch, com quantidade esperada, observada e ausente nos documentos registrados.
- Eventos operacionais por estágio e categoria heurística: rate limit, parâmetro não suportado, autenticação, timeout, resposta inválida e demais falhas. Mensagens brutas não são expostas nas tabelas.
- Rankings por contagem e taxa de erros, empates explícitos e conclusões descritivas. Falhas do analisador propagadas aos ramos são separadas de erros próprios de cada branch.

Os CSVs permitem auditar e reutilizar as tabelas:

| Arquivo | Conteúdo |
|---|---|
| `execution_resources.csv` | Recursos por documento e origem de cada valor |
| `execution_resource_summary.csv` | Médias, cobertura e somas conhecidas por run/recurso |
| `execution_documents.csv` | Status e motivos por documento |
| `execution_document_summary.csv` | Contagens por status/motivo do documento |
| `execution_branches.csv` | Resultados, tentativas e falhas por documento/branch |
| `execution_branch_summary.csv` | Resumo por branch, contagens e taxas |
| `execution_branch_reasons.csv` | Motivos finais por branch |
| `execution_failures.csv` | Eventos e impactos por documento, branch e tentativa |
| `execution_failure_summary.csv` | Eventos agregados por escopo, estágio e categoria |

**Interpretação:** `revision_limit` é um término sem aprovação após o limite de tentativas, não necessariamente erro técnico. Rejeições de qualidade também são separadas. Falhas finais e falhas de tentativas não devem ser somadas: podem representar o mesmo evento. `event_count` indica o que entra na contagem de eventos. A identificação de estágio/categoria e de propagação do analisador é heurística, baseada nos campos persistidos; não comprova a causa-raiz.

`llm_calls` representa as invocações lógicas registradas pelo fluxo: retries internos do provider podem não estar incluídos. Os tempos estão em segundos; `retry_wait_seconds` só desconta backoff sem outra chamada ativa. Os valores históricos ausentes **não são zero**, e não se estima o tempo sem backoff de runs antigas. Quando ambos os componentes de tokens são conhecidos, `total_tokens` ausente pode ser obtido pela soma exata de entrada e saída.

Os recursos usam `results.usage` prioritariamente e o CSV de metadados como fallback; conflitos numéricos geram avisos. Somente observações válidas entram na média. Uma soma com cobertura incompleta não é o consumo total real. Comparações não são pareadas e não demonstram superioridade estatística nem causalidade. Denominadores de branches esperadas abrangem os documentos presentes nos resultados, não documentos do corpus ainda não executados.

### Gerar apenas o relatório operacional (runs antigas ou novas)

O comando independente usa apenas a biblioteca padrão Python. Não precisa do corpus, de métricas NILC/semânticas ou de dependências para gráficos. Lê os nomes de arquivos no manifesto da run.

```sh
uv run python -m metrics.analyze_execution \
  --run-dir \
    output/porsimplessent-runs/73734703-0884-4d5f-9a9e-7f29e07ad877 \
    output/porsimplessent-runs/6eea9265-437d-4e9c-bc16-dcd0a94cbcb8
```

Para reutilizar a seleção de runs de uma análise já feita:

```sh
uv run python -m metrics.analyze_execution \
  --analysis-dir output/analyses/20260918T175626_eabde156
```

A saída fica em **uma nova pasta** `output/analyses/execution_<timestamp-UTC>_<id>/`. Use `--output-dir` para escolher outra pasta nova dentro de `output/`. Não altera a análise anterior nem os arquivos das runs; não recalcula métricas ou gráficos. `--run-dir` e `--analysis-dir` são alternativas mutuamente exclusivas.

O novo `manifest.json` registra data, pastas, modelos, arquivos e hashes, código utilizado e avisos. Com `--analysis-dir`, registra também o manifesto anterior e seu hash. O relatório lê os **arquivos atuais das runs**, não os agregados antigos; se uma fonte mudou desde a análise anterior, há aviso. Caso as pastas tenham sido movidas, use `--run-dir` com os caminhos atuais. Comparações operacionais entre corpus diferentes são permitidas com aviso, sem abrir o corpus, e exigem cautela adicional.

### Gráficos

Conforme as métricas disponíveis:

- Boxplots de comprimento dos textos.
- Boxplots e violinos de métricas NILC e semânticas.
- Trajetórias original → simples → moderada → agressiva com mediana e intervalo interquartil.
- Radar comparativo de medianas e radares individuais por run/versão com IQR.
- Boxplots e violinos dos deltas em relação ao original.
- Dispersões delta Flesch × delta LSA adjacente e delta Gunning-Fog × delta LSA adjacente.

As 13 métricas NILC selecionadas são as do notebook: `hard_conjunctions_ratio`, `long_sentence_ratio`, `medium_long_sentence_ratio`, `simple_word_ratio`, `lsa_adj_mean`, `lsa_adj_std`, `lsa_givenness_mean`, `lsa_givenness_std`, `cross_entropy`, `brunet`, `dalechall_adapted`, `flesch`, `gunning_fox`. Outras métricas semânticas disponíveis também entram nas tabelas/gráficos.

## Critérios de interpretação

- As médias descritivas usam todos os valores finitos disponíveis por run/versão/métrica. **n pode variar entre modelos**: diferenças de médias não implicam comparação pareada ou superioridade estatística. Use `values.csv` para selecionar uma coorte comum quando necessário.
- Os deltas usam pares por ID; trajetórias e Friedman exigem as quatro versões completas por métrica/run. Friedman exige pelo menos três documentos. Insuficiência ou ausência de variação é sinalizada, sem estatísticas inventadas.
- Wilcoxon bilateral só é executado após Friedman com p < 0,05. Holm corrige os seis pares dentro de cada run/métrica; não há correção entre todas as runs/métricas, nem teste automático de superioridade entre modelos. Kendall W descreve o efeito global.
- O radar usa normalização conjunta p5/p95 de cada conjunto, com recorte 0–1. Eixos constantes finitos recebem 0,5; ausentes permanecem ausentes. **Maior raio não significa melhor qualidade**. Compare modelos no radar combinado, não entre radares de pastas com escalas diferentes.
- Direções como menor entropia/maior coesão são hipóteses exploratórias, não qualidade universal. Brunet não recebe direção automática. Os percentuais usam o total de pares válidos, não o número teórico de documentos × métricas.
- Singleton/constante em violinplots vira ponto, evitando falhas de densidade.
- A análise não corrige truncamento BERTScore, erros semânticos ou vieses dos avaliadores: apenas resume os arquivos fornecidos.
