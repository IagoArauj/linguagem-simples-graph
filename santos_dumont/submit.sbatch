#!/bin/bash -l
#SBATCH --job-name=lsg-qwen3
#SBATCH --partition=sequana_gpu
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --exclusive
#SBATCH --output=slurm-%j.out

# Submit from the project root so Slurm opens slurm-JOBID.out there.
# No installs here: first run bash santos_dumont/prepare_env.sh on a login node.
set -euo pipefail
export ROOT="${ROOT:-/scratch/ufsj/hpc4agents-br/iago.oliveira3/linguagem-simples-graph}"
cd "$ROOT"
module load cuda/12.6_sequana
module load python
source .venv/bin/activate
export UV_PROJECT_ENVIRONMENT="$ROOT/.venv"
export UV_OFFLINE=1 UV_NO_SYNC=1 UV_PYTHON_DOWNLOADS=never
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export VLLM_NO_USAGE_STATS=1
command -v uv >/dev/null || { echo 'uv is not available in this job environment' >&2; exit 1; }
for executable in python ray vllm; do
    [[ -x "$ROOT/.venv/bin/$executable" ]] || { echo "Missing .venv/bin/$executable; run bash santos_dumont/prepare_env.sh before sbatch" >&2; exit 1; }
done

export MODEL_PATH="${MODEL_PATH:-/scratch/ufsj/hpc4agents-br/iago.oliveira3/agents/Qwen3-32B}"
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen/Qwen3-32B}"
export TP_SIZE="${TP_SIZE:-4}" PP_SIZE="${PP_SIZE:-4}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
# vLLM 0.8.5.post1 não oferece o parser qwen3. Deixe vazio nessa versão.
export REASONING_PARSER="${REASONING_PARSER:-}"
export STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-1200}"
export RAY_PORT="${RAY_PORT:-6379}"
export GPUS_PER_NODE=4 CPUS_PER_NODE=16
export EXPECTED_NODES="${SLURM_JOB_NUM_NODES:?Run using sbatch}"
[[ "$STARTUP_TIMEOUT" =~ ^[1-9][0-9]*$ ]] || { echo 'STARTUP_TIMEOUT must be a positive integer' >&2; exit 1; }
[[ -d "$MODEL_PATH" ]] || { echo 'MODEL_PATH must be an existing local model directory' >&2; exit 1; }
[[ -f santos_dumont/porsimplessent.yaml ]] || { echo 'Create santos_dumont/porsimplessent.yaml before submitting' >&2; exit 1; }
python santos_dumont/wait_ready.py layout --nodes "$EXPECTED_NODES" --gpus "$GPUS_PER_NODE" --tp "$TP_SIZE" --pp "$PP_SIZE"

# Create once, before srun exports the environment. Never echo the key or use CLI auth.
if [[ -z "${VLLM_API_KEY:-}" ]]; then
    VLLM_API_KEY="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
fi
export VLLM_API_KEY
export RAY_USAGE_STATS_ENABLED=0
export JOB_DIR="$ROOT/output/slurm/${SLURM_JOB_ID:?}"
umask 077
mkdir -p "$ROOT/output/slurm"
mkdir "$JOB_DIR"  # Refuse stale success markers, including on a requeued job.
# Only this exclusive allocation may use ray stop during cleanup.
export SD_EXCLUSIVE_JOB="$SLURM_JOB_ID"
python - <<'PY' >"$JOB_DIR/serving.json"
import importlib.metadata
import json
import os
import sys
from datetime import datetime, timezone
keys = ("SLURM_JOB_ID", "SLURM_JOB_NODELIST", "MODEL_PATH", "SERVED_MODEL_NAME",
        "TP_SIZE", "PP_SIZE", "MAX_MODEL_LEN", "REASONING_PARSER", "STARTUP_TIMEOUT")
json.dump({"created_at": datetime.now(timezone.utc).isoformat(),
           "settings": {key: os.environ.get(key) for key in keys},
           "versions": {name: importlib.metadata.version(name) for name in ("vllm", "ray", "torch")}},
          sys.stdout, indent=2)
PY

# Resolve the allocated rank-0 hostname, not an arbitrary hostname -I address.
mapfile -t hosts < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
export HEAD_HOST="${hosts[0]}"
# When selecting a non-DNS interface, rank 0 resolves HEAD_IP and publishes it.
# An explicit HEAD_IP must match that node's VLLM_HOST_IP.
export HEAD_IP="${HEAD_IP:-}"
srun --export=ALL --nodes="$EXPECTED_NODES" --ntasks="$EXPECTED_NODES" \
    --ntasks-per-node=1 --kill-on-bad-exit=1 \
    bash -l "$ROOT/santos_dumont/node.sh" "$@"
