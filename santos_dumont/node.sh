#!/bin/bash -l
# Launched only by submit.sbatch: one task per exclusively allocated node.
set -euo pipefail
set -m  # Each background command gets a process group for descendant cleanup.
: "${ROOT:?}" "${JOB_DIR:?}" "${SLURM_PROCID:?}" "${STARTUP_TIMEOUT:?}"
[[ "${SD_EXCLUSIVE_JOB:-}" == "${SLURM_JOB_ID:?}" ]] || { echo 'Requires exclusive submit.sbatch allocation' >&2; exit 1; }
cd "$ROOT"
module load cuda/12.6_sequana
module load python
source .venv/bin/activate
rank="$SLURM_PROCID"
exec >>"$JOB_DIR/node-$rank.log" 2>&1
ray_pid= vllm_pid= task_pid=
ray_started=0
cleanup() {
    status=$?
    trap - EXIT TERM INT
    for pid in "$task_pid" "$vllm_pid" "$ray_pid"; do
        if [[ -n "$pid" ]]; then kill -TERM -- "-$pid" 2>/dev/null || true; fi
    done
    # Safe only because submit.sbatch reserves entire nodes; never use pkill.
    if (( ray_started )); then timeout 30 ray stop --force >>"$JOB_DIR/ray-$rank.log" 2>&1 || true; fi
    for pid in "$task_pid" "$vllm_pid" "$ray_pid"; do
        if [[ -n "$pid" ]]; then
            kill -KILL -- "-$pid" 2>/dev/null || true
            wait "$pid" 2>/dev/null || true
        fi
    done
    exit "$status"
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
fail() { echo "rank $rank: $*" >&2; exit 1; }
services_alive() {
    kill -0 "$ray_pid" 2>/dev/null || fail 'Ray exited unexpectedly'
    if [[ -n "$vllm_pid" ]]; then kill -0 "$vllm_pid" 2>/dev/null || fail 'vLLM exited unexpectedly'; fi
}
# Protect even blocking ray.init/HTTP calls with an outer wall-clock deadline.
run_guarded() {
    local limit="$1" start="$SECONDS" status=0
    shift
    "$@" &
    task_pid=$!
    while kill -0 "$task_pid" 2>/dev/null; do
        services_alive
        if (( limit > 0 && SECONDS - start >= limit )); then fail 'Readiness timed out'; fi
        sleep 1
    done
    wait "$task_pid" || status=$?
    task_pid=
    return "$status"
}

# Slurm's node name must resolve on the Ray network. Override SD_NETWORK_INTERFACE
# (all nodes) or VLLM_HOST_IP (per node) if DNS uses a management network.
node_host="${SLURMD_NODENAME:?}"
if (( rank == 0 )); then node_host="${HEAD_HOST:?}"; fi
export VLLM_HOST_IP="${VLLM_HOST_IP:-$(python santos_dumont/wait_ready.py ip "$node_host")}"
[[ -n "$VLLM_HOST_IP" ]] || fail 'Could not resolve node IPv4; configure SD_NETWORK_INTERFACE'
if (( rank == 0 )); then
    HEAD_IP="${HEAD_IP:-$VLLM_HOST_IP}"
    [[ "$HEAD_IP" == "$VLLM_HOST_IP" ]] || fail 'HEAD_IP must match rank 0 VLLM_HOST_IP'
    printf '%s\n' "$HEAD_IP" >"$JOB_DIR/head-ip.tmp"
    mv "$JOB_DIR/head-ip.tmp" "$JOB_DIR/head-ip"
else
    deadline=$((SECONDS + STARTUP_TIMEOUT))
    until [[ -f "$JOB_DIR/head-ip" ]]; do
        (( SECONDS < deadline )) || fail 'Timed out waiting for head address'
        sleep 1
    done
    read -r HEAD_IP <"$JOB_DIR/head-ip"
    timeout "$STARTUP_TIMEOUT" python santos_dumont/wait_ready.py tcp \
        --host "$HEAD_IP" --port "$RAY_PORT" --timeout "$STARTUP_TIMEOUT"
fi
export RAY_ADDRESS="$HEAD_IP:$RAY_PORT"
# Local, job-specific Ray session directory; short enough for Unix sockets.
ray_tmp="/tmp/lsg-ray-${SLURM_JOB_ID}-${rank}"
ray_args=(--node-ip-address "$VLLM_HOST_IP" --num-gpus "$GPUS_PER_NODE"
    --num-cpus "$CPUS_PER_NODE" --disable-usage-stats --block)
ray_started=1
if (( rank == 0 )); then
    ray start --head --port "$RAY_PORT" --include-dashboard=false \
        --temp-dir "$ray_tmp" "${ray_args[@]}" >"$JOB_DIR/ray-$rank.log" 2>&1 &
    ray_pid=$!
    run_guarded "$STARTUP_TIMEOUT" python santos_dumont/wait_ready.py tcp \
        --host "$HEAD_IP" --port "$RAY_PORT" --timeout "$STARTUP_TIMEOUT"
    run_guarded "$STARTUP_TIMEOUT" python santos_dumont/wait_ready.py ray \
        --address "$RAY_ADDRESS" --nodes "$EXPECTED_NODES" --gpus "$GPUS_PER_NODE" \
        --timeout "$STARTUP_TIMEOUT"
    touch "$JOB_DIR/cluster.ready"
    # vLLM 0.8.5.post1 não aceita `qwen3` como reasoning parser. O argumento
    # fica ausente por padrão e só deve ser configurado após atualizar/testar.
    reasoning_args=()
    if [[ -n "${REASONING_PARSER:-}" ]]; then
        reasoning_args=(--reasoning-parser "$REASONING_PARSER")
    fi
    # vLLM supports VLLM_API_KEY environment auth; do not expose it in argv.
    vllm serve "$MODEL_PATH" --served-model-name "$SERVED_MODEL_NAME" \
        --host 127.0.0.1 --port 8000 --distributed-executor-backend ray \
        --tensor-parallel-size "$TP_SIZE" --pipeline-parallel-size "$PP_SIZE" \
        --max-model-len "$MAX_MODEL_LEN" "${reasoning_args[@]}" \
        --enforce-eager --gpu-memory-utilization 0.85 >"$JOB_DIR/vllm-$rank.log" 2>&1 &
    vllm_pid=$!
    run_guarded "$STARTUP_TIMEOUT" python santos_dumont/wait_ready.py http \
        --url http://127.0.0.1:8000/v1/models --model "$SERVED_MODEL_NAME" \
        --timeout "$STARTUP_TIMEOUT"
    export LSG_ANALYZER_BASE_URL=http://127.0.0.1:8000/v1
    export LSG_SIMPLIFIER_BASE_URL=http://127.0.0.1:8000/v1
    export LSG_EVALUATOR_BASE_URL=http://127.0.0.1:8000/v1
    run_guarded 0 uv run --no-sync --offline --extra hpc python main.py \
        --config santos_dumont/porsimplessent.yaml "$@" --model "$SERVED_MODEL_NAME" \
        >"$JOB_DIR/pipeline-$rank.log" 2>&1
    touch "$JOB_DIR/success"
    # Workers see success before testing Ray liveness and acknowledge it before
    # exiting. Keep the head alive until every worker has crossed this barrier.
    deadline=$((SECONDS + STARTUP_TIMEOUT))
    for ((worker = 1; worker < EXPECTED_NODES; worker++)); do
        until [[ -f "$JOB_DIR/ack-$worker" ]]; do
            (( SECONDS < deadline )) || fail 'Worker success acknowledgement timed out'
            # Acknowledged workers may already stop their local Ray, so vLLM
            # need not remain healthy after the pipeline has completed.
            kill -0 "$ray_pid" 2>/dev/null || fail 'Ray head exited before worker acknowledgements'
            sleep 1
        done
    done
else
    deadline=$((SECONDS + STARTUP_TIMEOUT))
    while :; do
        ray start --address "$RAY_ADDRESS" "${ray_args[@]}" >>"$JOB_DIR/ray-$rank.log" 2>&1 &
        ray_pid=$!
        while kill -0 "$ray_pid" 2>/dev/null; do
            if [[ -f "$JOB_DIR/success" ]]; then touch "$JOB_DIR/ack-$rank"; exit 0; fi
            if [[ ! -f "$JOB_DIR/cluster.ready" ]] && (( SECONDS >= deadline )); then
                fail 'Timed out joining Ray cluster'
            fi
            sleep 1
        done
        if [[ -f "$JOB_DIR/success" ]]; then touch "$JOB_DIR/ack-$rank"; exit 0; fi
        wait "$ray_pid" || true
        # Retry only during startup; reclaim this exclusive node's failed Ray
        # attempt so a partial connection cannot leave stale local daemons.
        [[ ! -f "$JOB_DIR/cluster.ready" ]] || fail 'Ray worker exited during execution'
        (( SECONDS < deadline )) || fail 'Ray worker connection retries exhausted'
        timeout 30 ray stop --force >>"$JOB_DIR/ray-$rank.log" 2>&1 || true
        sleep 2
    done
fi
