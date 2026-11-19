#!/bin/bash -l
# Execute no nó de login/preparação Linux com internet, nunca dentro do sbatch.
set -euo pipefail
ROOT="${ROOT:-/scratch/ufsj/hpc4agents-br/iago.oliveira3/linguagem-simples-graph}"
cd "$ROOT"
if [[ "$(uname -s)" != Linux || "$(uname -m)" != x86_64 ]]; then
    echo 'O extra hpc requer Linux x86_64; prepare o ambiente no cluster, não no macOS.' >&2
    exit 1
fi
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo 'Prepare o ambiente fora de uma alocação Slurm, antes de submeter o job.' >&2
    exit 1
fi
module load cuda/12.6_sequana
module load python
command -v uv >/dev/null || { echo 'Disponibilize uv no PATH antes da preparação.' >&2; exit 1; }
export UV_PROJECT_ENVIRONMENT="$ROOT/.venv"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$ROOT/.uv-cache}"
# Se UV precisar baixar Python, instale em armazenamento compartilhado também.
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$ROOT/.uv-python}"
# --locked recusa mudanças silenciosas de versões; não atualiza uv.lock.
uv sync --locked --extra hpc --no-dev --python 3.12
uv pip check --python "$ROOT/.venv/bin/python"
"$ROOT/.venv/bin/python" - <<'PY'
import importlib.metadata
import sys
assert sys.version_info[:2] == (3, 12), 'O ambiente hpc requer Python 3.12'
for name in ('vllm', 'ray', 'torch', 'transformers', 'langgraph', 'langchain-openai'):
    print(f'{name}=={importlib.metadata.version(name)}')
PY
for executable in ray vllm; do
    [[ -x "$ROOT/.venv/bin/$executable" ]] || { echo "Executável ausente: $executable" >&2; exit 1; }
done
printf '\nAmbiente pronto em %s/.venv. Confirme os pesos locais e submeta com sbatch.\n' "$ROOT"
