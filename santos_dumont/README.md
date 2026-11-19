# Simplificação textual no Santos Dumont

Este diretório executa **somente o pipeline `main.py`**, usando Qwen3-32B servido localmente pelo vLLM. Não inicia frontend, DeerFlow, métricas, análises ou APIs externas de geração.

## Arquitetura

- `prepare_env.sh`: instala o extra `hpc` em armazenamento compartilhado, em um nó Linux com internet, antes de submeter o job.
- `submit.sbatch`: solicita 4 nós exclusivos em `sequana_gpu`, 4 GPUs e 16 CPUs por nó, 64 GB **por nó**, por 1 hora. Inicia uma task por nó com `srun`.
- `node.sh`: rank 0 inicia Ray head, espera as GPUs dos quatro nós, inicia vLLM, espera `/v1/models` e executa o pipeline **uma única vez**. Os demais ranks mantêm Ray workers. O nó do pipeline também participa da inferência; não se reserva um nó apenas para cliente.
- `wait_ready.py`: probes com prazo para rede, membership Ray e modelo HTTP.
- `porsimplessent.yaml`: corpus PorSimplesSent, três branches, analisador/avaliador com JSON Schema e simplificador textual, todos no endpoint local OpenAI-compatible.

Padrão: **TP=4, PP=4** (16 GPUs), favorecendo tensor parallel dentro de um nó e pipeline parallel entre nós. Não é a mesma topologia do exemplo TP=16/PP=1; ambas precisam ser avaliadas nas GPUs/rede e versão de vLLM disponíveis. A verificação exige `TP_SIZE * PP_SIZE = nós * 4`.

O contexto padrão é 32.768, sem forçar YaRN ou extensão para 131.072. A entrada e a saída de cada chamada precisam caber no contexto. Usa `--enforce-eager` e utilização de memória GPU 0,85. Não há function calling neste fluxo, portanto não são necessários `--tool-call-parser`/`--enable-auto-tool-choice`.

O vLLM **0.8.5.post1** fixado no extra `hpc` não oferece `qwen3` entre os valores aceitos por `--reasoning-parser` (apenas `deepseek_r1` e `granite` nessa instalação). Por isso, o argumento é omitido. `REASONING_PARSER` existe como override opcional, mas não deve ser definido nessa versão. Ative `qwen3` somente depois de atualizar e validar vLLM; o parser separa conteúdo de raciocínio na API, mas não é necessário para iniciar o modelo.

## Preparação (antes de solicitar GPUs)

Raiz esperada:

```text
/scratch/ufsj/hpc4agents-br/iago.oliveira3/linguagem-simples-graph
```

O `.venv` nessa raiz deve estar preparado, acessível em **todos os nós** e conter:

- dependências do projeto e **Python 3.12**;
- `ray`, `vllm` e PyTorch/CUDA compatíveis entre si e com as GPUs/drivers do cluster;
- `uv` disponível no PATH.

### Instalar uma vez, no nó de login com internet

```bash
cd /scratch/ufsj/hpc4agents-br/iago.oliveira3/linguagem-simples-graph
bash santos_dumont/prepare_env.sh
```

O script valida Linux x86_64, carrega os mesmos módulos do job e executa:

```bash
uv sync --locked --extra hpc --no-dev --python 3.12
uv pip check --python .venv/bin/python
```

O extra **`hpc`** no `pyproject.toml` instala vLLM **0.8.5.post1**, Ray **2.43.0** (incluindo dependências de serviços) e Transformers **4.51.3** conforme o `uv.lock`. O vLLM exige PyTorch **2.6.0**, torchaudio e torchvision correspondentes; eles são instalados transitivamente. Ray 2.44 não é usado porque a própria distribuição de vLLM o exclui. Esse conjunto tem suporte a Qwen3, mas precisa de validação de execução com os drivers e GPUs do cluster: o módulo CUDA sozinho não garante compatibilidade binária.

As versões antigas foram escolhidas como conjunto explícito e conservador, não como as versões mais recentes. Uma atualização deve ser feita no extra e no lockfile, seguida de nova preparação e teste no cluster. Não há atualização automática de versões pelo job.

A preparação baixa muitos GB de wheels NVIDIA/PyTorch e pode levar tempo. Precisa de espaço no scratch, acesso ao PyPI e permissão para instalar pacotes no nó de preparação. Ela não exige uma GPU para resolver/instalar e conferir metadados, mas **não testa CUDA nem inferência**. O script recusa execução dentro de um job Slurm.

O ambiente fica em `.venv`, o cache UV em `.uv-cache` e, se necessário, o Python baixado pelo UV em `.uv-python`, todos na raiz compartilhada. Não copie uma `.venv` do macOS: o ambiente deve ser criado no Linux, com arquitetura, glibc e bibliotecas compatíveis com os nós de computação. O interpretador referenciado pela `.venv` também deve estar acessível nos nós.

**Não prepare/sincronize o ambiente enquanto um job estiver usando essa `.venv`.** `uv sync` faz sincronização exata e pode remover pacotes não selecionados. Para manter o serving, use sempre `--extra hpc` ao atualizar esse ambiente.

O extra `hpc` é restrito a Linux x86_64/Python 3.12; em outras plataformas os marcadores excluem os pacotes GPU, por isso o script valida o sistema antes de instalar. `hpc` e `metrics` são declarados mutuamente exclusivos no UV, pois o stack BERTScore legado do macOS Intel exige outro Transformers/PyTorch. Não use `--all-extras`; use outro ambiente para BERTScore. O extra `analysis` pode ser combinado com `hpc` se necessário, embora não seja usado pelo job.

### Nos nós de computação: nenhuma instalação

O job usa `uv run --no-sync --offline --extra hpc`: não sincroniza, instala ou baixa Python. Define também `UV_OFFLINE`, `UV_NO_SYNC`, `UV_PYTHON_DOWNLOADS=never`, `HF_HUB_OFFLINE` e `TRANSFORMERS_OFFLINE`, e desliga telemetria de Hugging Face/vLLM/Ray. Se faltar um pacote ou arquivo de modelo, a execução falha em vez de tentar baixá-lo. A comunicação interna Ray/vLLM entre os nós continua necessária.

O download do **modelo não faz parte de `uv sync`**. Prepare pesos, configuração e tokenizer completos no caminho local compartilhado antes de `sbatch`. Se nem o nó de login tiver internet, use um nó Linux compatível com acesso e armazenamento compartilhado autorizado pelo cluster para executar a preparação; não tente instalar durante a alocação GPU.

Os módulos `cuda/12.6_sequana` e `python` são carregados antes da ativação do ambiente. Revise esses nomes se os módulos do cluster mudarem. São necessários Bash 4+, `scontrol`, `srun` e `timeout` (Linux).

Os pesos/tokenizer devem estar completos em armazenamento compartilhado. Por padrão reutilizamos o caminho do modelo do job fornecido:

```text
/scratch/ufsj/hpc4agents-br/iago.oliveira3/agents/Qwen3-32B
```

**Confirme esse caminho**; use `MODEL_PATH` para apontar para outra cópia local. A nova pasta de scripts não precisa conter os pesos.

## Submissão

No terminal do cluster, a partir da raiz do projeto:

```bash
cd /scratch/ufsj/hpc4agents-br/iago.oliveira3/linguagem-simples-graph
sbatch santos_dumont/submit.sbatch
```

Não precisa executar `chmod`: Slurm lê o batch e `node.sh` é iniciado via Bash.

Override do modelo local:

```bash
MODEL_PATH=/scratch/ufsj/hpc4agents-br/iago.oliveira3/agents/Qwen3-32B \
  sbatch santos_dumont/submit.sbatch
```

Para usar a topologia do job anterior:

```bash
TP_SIZE=16 PP_SIZE=1 sbatch santos_dumont/submit.sbatch
```

Para mais tempo de alocação, conforme os limites da fila:

```bash
sbatch --time=04:00:00 santos_dumont/submit.sbatch
```

Carregamento do modelo e execução compartilham esse tempo total. `STARTUP_TIMEOUT` vale 1.200 segundos **por etapa de espera**, não substitui o walltime do Slurm. Uma hora pode não ser suficiente para carregar o modelo e processar todo o corpus com revisões.

Argumentos adicionais são encaminhados ao `main.py`, por exemplo:

```bash
sbatch santos_dumont/submit.sbatch --max-revisions 2
```

O nome do modelo dos três agentes é forçado para `SERVED_MODEL_NAME`, por padrão `Qwen/Qwen3-32B`. Não use nomes de modelo diferentes por agente contra este servidor único. Revise variáveis `LSG_*` herdadas: elas têm precedência sobre o YAML (exceto opções CLI), e podem alterar corpus, limites ou providers inadvertidamente.

## Resultados e logs

```text
slurm-JOBID.out                          # no diretório de submissão
output/slurm/JOBID/
  serving.json                          # configuração de serving e versões GPU
  node-0.log ... node-3.log
  ray-0.log ... ray-3.log
  vllm-0.log
  pipeline-0.log                         # contém run_id do pipeline
output/hpc-porsimplessent-runs/RUN_ID/
  manifest.json
  results.jsonl
  metadata_per_text.csv
  summary.json
```

Os IDs do job e da run são diferentes. O manifesto normal do projeto continua sendo gerado com configuração, corpus, prompts e versões. Os logs/arquivos de controle não são compartilhados entre jobs. Um diretório `output/slurm/JOBID` preexistente é recusado; uma requeue automática do mesmo job não é suportada — submeta outro job.

`success` indica que `main.py` terminou com código zero, **não que todas as simplificações foram aprovadas**. Verifique os status de `results.jsonl` e `summary.json`.

## Retomada

Após walltime ou interrupção, submeta um novo job com o `run_id` impresso no log anterior:

```bash
sbatch santos_dumont/submit.sbatch --run-id UUID_DA_RUN --resume
```

Mantenha os mesmos parâmetros do pipeline, inclusive overrides como `--max-revisions`. `http://127.0.0.1:8000/v1` permanece igual mesmo em outro nó, evitando divergência de configuração causada pelo IP dinâmico. A chave é transitória e não entra no manifesto. O job ainda precisa recarregar Ray/vLLM. A retomada preserva o comportamento do projeto: documentos já registrados, inclusive falhas, são ignorados; não é retry automático de documentos falhos.

## Rede e segurança

A API vLLM escuta somente em **127.0.0.1 no head**; o pipeline roda ali. Uma chave aleatória é gerada por job em `VLLM_API_KEY`, sem segredo fixo no código ou argumento de linha de comando. Os serviços internos Ray precisam comunicar-se entre nós em rede privada/confiável; a autenticação HTTP não protege Ray. Não exponha as portas do Ray à internet.

O IPv4 de cada nó é obtido do hostname alocado. Se o DNS escolher a rede de gerenciamento, defina `SD_NETWORK_INTERFACE` com o nome real da interface de dados do cluster (igual em todos os nós); o head publica seu IP no diretório do job. **Não exporte o mesmo `VLLM_HOST_IP` para todos os nós**. A escolha de interface NCCL/InfiniBand pode exigir `NCCL_SOCKET_IFNAME`, `GLOO_SOCKET_IFNAME` ou ajustes específicos definidos pelos administradores; nenhum nome de interface é presumido aqui.

O job reserva nós exclusivos porque o cleanup executa `ray stop --force` localmente. Não retire `--exclusive` nem inicie outro cluster Ray nesses nós. Em falha, `srun --kill-on-bad-exit=1` encerra as outras tasks; traps encerram os processos locais. No sucesso, os workers confirmam o término do pipeline antes do head sair. SIGKILL/walltime extremo podem impedir traps; o Slurm continua responsável pelo encerramento dos processos da alocação.

## Validação

Testes locais cobrem probes mockados, topologia e sintaxe Bash. **Não validam execução real de Slurm, NCCL, Ray ou vLLM**. Antes da execução completa, confirme compatibilidade da versão instalada de vLLM com Qwen3, pipeline parallel, `VLLM_API_KEY` e `response_format` JSON Schema (usado por analisador/avaliador). Comece com um corpus pequeno via `--corpus` e uma nova run para esse teste.
