# Métricas semânticas de simplificação

O módulo `metrics.semantic` calcula métricas localmente, sem API de geração de texto. Ele lê `manifest.json` e o arquivo de resultados indicado no manifesto de cada execução. O corpus, os campos de ID/texto e o nome do arquivo de resultados não são fixados no código.

## Instalação e execução

Na raiz do projeto:

```sh
uv sync --extra metrics
uv run --extra metrics python -m metrics.semantic --help
uv run --extra metrics python -m metrics.semantic \
  --run-dir output/porsimplessent-runs/6eea9265-437d-4e9c-bc16-dcd0a94cbcb8
```

### macOS Intel (x86_64)

Use **Python 3.12** para o extra `metrics`. PyTorch 2.2.2 é a última versão com wheels para macOS Intel e não oferece wheels para Python 3.13 ou superior. O projeto fixa essa versão, NumPy < 2 e Transformers < 4.45 somente nessa plataforma; as demais plataformas não recebem essas restrições.

```sh
uv sync --extra metrics --python 3.12
uv run --extra metrics --python 3.12 python -m metrics.semantic --help
```

Use `--device cpu` (padrão) no Mac Intel. A resolução do lockfile exige compatibilidade com macOS Intel/Python 3.12. Como essa plataforma depende de versões antigas, carregue apenas pesos de origem confiável.

Informe uma ou mais pastas em `--run-dir`. Cada pasta deve conter um manifesto versão 1.0 e resultados do workflow versão 1.0. O nome do modelo do experimento não interfere na execução destas métricas.

O corpus é obtido de `effective_config.corpus.path`. Se a execução foi copiada de outra máquina, forneça `--corpus input/PorSimplesSent/corpus.jsonl`: seu SHA-256 ainda deve coincidir com o manifesto. IDs inteiros do corpus são associados aos `document_id` textuais. Documentos duplicados, IDs ausentes, hashes divergentes e resultados de outro `run_id` são rejeitados antes da inferência.

Por padrão, as referências humanas são `natural_text` e `strong_text`; use `--natural-field` e `--strong-field` para outros nomes.

## Baselines SARI do corpus humano

O módulo independente `metrics.human_sari` calcula quatro comparações, sempre com fonte Original: cópia do Original contra Natural; cópia contra Strong; Natural contra Strong; Strong contra Natural. Reutiliza a função SARI dos experimentos, sem carregar BERT, acessar APIs ou exigir extras.

```sh
uv run python -m metrics.human_sari -i input/PorSimplesSent/corpus.jsonl
```

Cria `human_sari.md` (médias e desvio-padrão amostral), `human_sari.csv` (escores por documento) e `human_sari_manifest.json` (timestamp, comparações e hashes do corpus/código), na pasta do corpus. Use `--output-dir` para outro destino ou `--overwrite` para recalcular arquivos existentes. O corpus original nunca é alterado.

A entrada deve conter `production_id`, `original_text`, `natural_text` e `strong_text`. Campos ausentes/vazios, IDs duplicados e JSON inválido interrompem o cálculo, sem substituir ausências por zero. As médias atribuem peso igual a cada documento, na escala 0–100. SARI não é simétrico; o valor humano cruzado não é teto de qualidade. Para comparar com IA, use a mesma referência e os mesmos IDs do CSV, especialmente quando houver falhas nas runs.

## Comparações

| Métrica | Fonte | Candidata | Referência |
|---|---|---|---|
| BERTScore humano | — | natural / strong | original |
| BERTScore IA | — | simplificação de cada ramo | original |
| ROUGE-L natural | — | simplificação IA | natural |
| ROUGE-L strong | — | simplificação IA | strong |
| SARI natural | original | simplificação IA | natural |
| SARI strong | original | simplificação IA | strong |
| SARI multi-reference | original | simplificação IA | natural e strong juntas |

SARI não é apenas similaridade entre dois textos: mede operações de manter, adicionar e excluir n-gramas em relação ao original. O SARI com duas referências não é a média dos dois SARIs individuais.

### BERTScore

- Modelo padrão: `xlm-roberta-base`, camada 9, `lang=pt`.
- Retorna precisão, recall e F1 sem reescala por baseline, com `idf=False`.
- `--model` e `--num-layers` permitem outra variante XLM-R, inclusive um diretório local de pesos. Ao trocar a variante, configure a camada explicitamente.
- `--device cpu` é o padrão; `cuda` e `mps` são opcionais e dependem do hardware/PyTorch.
- A primeira execução baixa os pesos do Hugging Face e requer rede e espaço em disco; os textos são processados localmente. Para operação offline, use pesos previamente disponíveis e `HF_HUB_OFFLINE=1`.
- O BERTScore padrão trunca textos que excedem a janela do tokenizer/modelo (aproximadamente 512 tokens no XLM-R). Nesses casos, o resultado **não avalia o documento inteiro**. Não é feita divisão automática em sentenças nem média de segmentos, pois isso definiria outra métrica. Considere esse limite ao interpretar textos longos.
- O modelo é carregado apenas se houver algum par ainda não calculado e é reutilizado entre as execuções da mesma CLI. Cada chamada avalia um par, para limitar memória e persistir o cache incrementalmente.

### ROUGE-L e SARI

Implementações locais em `text_similarity.py`, sem stemmer inglês. A tokenização usa Unicode NFC, minúsculas e separação de palavras/pontuação; acentos são preservados. ROUGE-L usa LCS e retorna P/R/F1 em 0–1, não ROUGE-Lsum. SARI retorna 0–100, com n-gramas 1–4, KEEP-F1, ADD-F1 e DELETE-precision, seguindo a variante Xu/Evaluate com convenção 0/0=1. A tokenização é diferente de 13a: não compare diretamente com números produzidos usando outro tokenizer.

## Arquivos produzidos e retomada

Na pasta do corpus:

- `human_bertscore.csv`: uma linha por documento/referência humana; nunca contém dados de experimentos.
- `bertscore_cache.sqlite3`: cache compartilhado de pares original/candidata. A chave inclui os textos e a configuração do scorer (modelo, camada, device, versões e parâmetros). Nenhum texto é armazenado no banco, apenas hashes e escores.

Na pasta de **cada** experimento:

- `semantic_metrics.csv`: uma linha por documento/ramo, com BERTScore, ROUGE-L contra cada humano e os três valores SARI. Inclui `run_id`, `document_id`, `branch_id`, hashes, configuração do scorer e status de execução/qualidade originais.

Repetir o comando não recalcula BERTScore para pares já concluídos, inclusive humanos ou candidatas idênticas entre experimentos. O cache é salvo após cada par; os CSVs são substituídos atomicamente quando completos. Uma mudança no texto ou na configuração cria uma nova chave, sem reutilização indevida. Para pesos locais, mantenha o diretório imutável ou troque seu caminho ao substituir os pesos. O cache não detecta substituições de pesos sob o mesmo nome/caminho.

Ramos sem `final_text` recebem `metric_status=skipped_no_text`, com escores vazios (não zero). Textos disponíveis são medidos mesmo quando rejeitados ou pertencentes a um ramo com falha; os status originais permitem filtragem posterior. `metric_status=success` indica cálculo concluído, não aprovação da simplificação.

Os CSVs refletem os resultados presentes no momento da leitura. Se o experimento ainda estiver em execução, aguarde o fim para evitar ler uma linha JSONL incompleta. Execute os jobs de métricas sequencialmente quando compartilham a mesma pasta de corpus/cache.

## Testes sem modelos ou rede

```sh
uv run pytest tests/test_semantic_metrics.py tests/test_text_similarity.py
```

O scorer BERT é substituído por uma implementação falsa. Os testes verificam alinhamento pelo manifesto, rejeição de hashes incompatíveis, retomada/cache entre experimentos, ausência de textos e exemplos numéricos de ROUGE-L/SARI. Não validam a inferência real de XLM-R.
