# SARI — baselines de cópia e simplificações humanas

Corpus: `corpus.jsonl`. Documentos: **154**.

Média aritmética dos escores individuais por documento, com peso igual. Escala de 0 a 100; uma referência por comparação. Não é agregação de n-gramas do corpus.

| Experimento | Source | Prediction | Reference | n | SARI médio | Desvio-padrão amostral | O que mede |
|---|---|---|---|---:|---:|---:|---|
| Copy baseline | Original | Original | Natural | 154 | 58.6495 | 3.2486 | Não simplificar, contra a referência Natural |
| Copy baseline Strong | Original | Original | Strong | 154 | 55.1055 | 2.3662 | Não simplificar, contra a referência Strong |
| Human Natural → Strong | Original | Natural | Strong | 154 | 85.0834 | 6.3286 | Operações da Natural em relação às operações da Strong |
| Human Strong → Natural | Original | Strong | Natural | 154 | 77.1135 | 11.9370 | Operações da Strong em relação às operações da Natural |

## Método e interpretação

- Mesma função `metrics.text_similarity.sari` usada nas simplificações de IA; variante SARIsent/Evaluate documentada no código, com n-gramas de ordens 1 a 4.
- Tokenização: Unicode NFC, minúsculas e regex `\w+|[^\w\s]`, mantendo pontuação. Não usa stemming ou a tokenização inglesa padrão.
- A convenção de divisão por zero é 1. Por isso, uma candidata que apenas copia o original não precisa receber SARI zero.
- SARI avalia operações de manter, adicionar e remover em relação ao original. Não é uma medida simétrica: Natural → Strong pode diferir de Strong → Natural.
- A referência é uma simplificação possível, não uma verdade única. Esses valores não são um teto humano nem comprovam preservação semântica ou superioridade entre níveis.
- Compare com a IA apenas usando a mesma implementação, tokenização, referência e conjunto de documentos. Se houver falhas nas runs, use o CSV para parear os IDs.
- `human_sari.csv` contém os valores sem arredondamento para cada documento/experimento. O manifesto registra hashes do corpus e do código. Nenhuma API ou modelo é utilizado.
