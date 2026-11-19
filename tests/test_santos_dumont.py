import io
import json
from pathlib import Path
import socket
import subprocess
import urllib.error

import pytest

from santos_dumont import wait_ready


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "santos_dumont"


def member(ip, gpus=4, alive=True):
    return {"Alive": alive, "NodeManagerAddress": ip, "Resources": {"GPU": gpus}}


def test_ray_membership_requires_all_alive_gpu_nodes():
    nodes = [member(f"10.0.0.{i}") for i in range(4)]
    assert wait_ready.ray_ready(nodes, 4, 4)
    assert wait_ready.ray_ready(nodes + [member("old", alive=False)], 4, 4)
    assert not wait_ready.ray_ready(nodes[:-1], 4, 4)
    assert not wait_ready.ray_ready(nodes + [member("extra")], 4, 4)
    assert not wait_ready.ray_ready(nodes[:-1] + [member("10.0.0.3", gpus=0)], 4, 4)
    assert not wait_ready.ray_ready(nodes[:-1] + [member("10.0.0.3", gpus=8)], 4, 4)
    assert not wait_ready.ray_ready(nodes[:-1] + [member("10.0.0.0")], 4, 4)
    assert not wait_ready.ray_ready(nodes[:-1] + [member("10.0.0.3", alive=False)], 4, 4)


def test_ray_cli_uses_mocked_membership_and_disconnects(monkeypatch):
    import sys
    from types import SimpleNamespace

    calls = []
    monkeypatch.setitem(sys.modules, "ray", SimpleNamespace(
        init=lambda **kwargs: calls.append(kwargs),
        nodes=lambda: [member(f"10.0.0.{i}") for i in range(4)],
        shutdown=lambda: calls.append("shutdown"),
    ))
    monkeypatch.setattr(sys, "argv", [
        "wait_ready.py", "ray", "--address", "10.0.0.0:6379", "--nodes", "4", "--gpus", "4",
    ])
    wait_ready.main()
    assert calls == [{"address": "10.0.0.0:6379", "logging_level": "ERROR"}, "shutdown"]


@pytest.mark.parametrize("payload,ready", [
    ({"data": [{"id": "Qwen/Qwen3-32B"}]}, True),
    ({"data": [{"id": "other"}]}, False),
    ({"data": []}, False),
    ({"data": None}, False),
    ({"data": 42}, False),
    ({"data": {"id": "Qwen/Qwen3-32B"}}, False),
    ({"data": [None]}, False),
    ({}, False),
    ([], False),
])
def test_http_model_and_auth(monkeypatch, payload, ready):
    def open_request(request, timeout):
        assert request.full_url == "http://127.0.0.1:8000/v1/models"
        assert request.get_method() == "GET"
        assert request.get_header("Authorization") == "Bearer test-secret"
        assert timeout == 5
        return io.BytesIO(json.dumps(payload).encode())

    monkeypatch.setattr(wait_ready.urllib.request, "urlopen", open_request)
    assert wait_ready.model_ready(
        "http://127.0.0.1:8000/v1/models", "Qwen/Qwen3-32B", "test-secret"
    ) is ready


@pytest.mark.parametrize("error", [
    urllib.error.URLError("not ready"),
    urllib.error.HTTPError("http://localhost", 401, "Unauthorized", {}, None),
    TimeoutError("slow response"),
])
def test_http_errors_not_ready(monkeypatch, error):
    def unavailable(*args, **kwargs):
        raise error

    monkeypatch.setattr(wait_ready.urllib.request, "urlopen", unavailable)
    assert not wait_ready.model_ready("http://localhost/v1/models", "model", "secret")


def test_invalid_http_json_not_ready(monkeypatch):
    monkeypatch.setattr(wait_ready.urllib.request, "urlopen", lambda *a, **kw: io.BytesIO(b"not json"))
    assert not wait_ready.model_ready("http://localhost/v1/models", "model", "secret")


def test_bounded_wait_without_real_sleep():
    now = [0.0]
    sleeps = []

    def sleep(seconds):
        sleeps.append(seconds)
        now[0] += seconds

    with pytest.raises(TimeoutError):
        wait_ready.wait_until(lambda: False, 5, clock=lambda: now[0], sleep=sleep)
    assert sleeps == [2, 2, 1]
    results = iter([False, True])
    wait_ready.wait_until(lambda: next(results), 5, clock=lambda: now[0], sleep=sleep)
    assert now[0] == 7


@pytest.mark.parametrize("layout", [(4, 4, 4, 4), (4, 4, 8, 2), (2, 4, 4, 2)])
def test_valid_layout(layout):
    wait_ready.validate_layout(*layout)


@pytest.mark.parametrize("layout", [(4, 4, 4, 1), (4, 4, 0, 4), (0, 4, 4, 4), (4, 4, -4, -4)])
def test_invalid_layout(layout):
    with pytest.raises(ValueError):
        wait_ready.validate_layout(*layout)


def test_ip_uses_allocated_hostname_ipv4(monkeypatch):
    def resolve(host, port, family, kind):
        assert (host, port, family, kind) == ("allocated-node", None, socket.AF_INET, socket.SOCK_STREAM)
        return [(family, kind, 6, "", (ip, 0)) for ip in ["127.0.0.1", "10.10.0.4"]]

    monkeypatch.setattr(socket, "getaddrinfo", resolve)
    assert wait_ready.node_ip("allocated-node") == "10.10.0.4"


def test_loopback_resolution_is_rejected(monkeypatch):
    monkeypatch.setattr(socket, "getaddrinfo", lambda *a: [(2, 1, 6, "", ("127.0.0.1", 0))])
    with pytest.raises(ValueError, match="SD_NETWORK_INTERFACE"):
        wait_ready.node_ip("allocated-node")


def test_tcp_probe_mocked(monkeypatch):
    def connect(address, timeout):
        assert address == ("head", 6379)
        assert timeout == 5
        return io.BytesIO()

    monkeypatch.setattr(socket, "create_connection", connect)
    assert wait_ready.tcp_ready("head", 6379)

    def refused(*args, **kwargs):
        raise ConnectionRefusedError

    monkeypatch.setattr(socket, "create_connection", refused)
    assert not wait_ready.tcp_ready("head", 6379)


@pytest.mark.parametrize("script", ["submit.sbatch", "node.sh", "prepare_env.sh"])
def test_bash_syntax(script):
    subprocess.run(["bash", "-n", str(SCRIPTS / script)], check=True, timeout=10)


def test_pipeline_yaml_loads_with_local_models():
    from config import load_effective_config

    config, _ = load_effective_config(SCRIPTS / "porsimplessent.yaml", environ={})
    run_config = config.to_run_config()
    from core.schemas import ModelRole

    for role in ModelRole:
        model = run_config.model(role)
        assert model.provider.value == "openai_compatible"
        assert model.name == "Qwen/Qwen3-32B"
        assert model.base_url == "http://127.0.0.1:8000/v1"
        assert model.api_key_env == "VLLM_API_KEY"
        assert model.supports_structured_output == (role.value != "simplifier")
    assert config.corpus.path.name == "corpus.jsonl"
    assert config.corpus.id_field == "production_id"
    assert config.corpus.text_field == "original_text"


def test_submission_and_pipeline_contracts():
    submit = (SCRIPTS / "submit.sbatch").read_text()
    node = (SCRIPTS / "node.sh").read_text()
    for directive in [
        "--partition=sequana_gpu", "--nodes=4", "--ntasks-per-node=1", "--gres=gpu:4",
        "--cpus-per-task=16", "--mem=64G", "--time=01:00:00", "--exclusive", "--output=slurm-%j.out",
    ]:
        assert f"#SBATCH {directive}" in submit
    for script in (submit, node):
        assert script.index("module load cuda/12.6_sequana") < script.index("source .venv/bin/activate")
        assert script.index("module load python") < script.index("source .venv/bin/activate")
        assert "--api-key" not in script
        assert "multi-node-serving.sh" not in script
        assert "deerflow" not in script
    assert "--kill-on-bad-exit=1" in submit
    assert 'node.sh" "$@"' in submit
    assert "uv run --no-sync --offline --extra hpc python main.py" in node
    assert '--config santos_dumont/porsimplessent.yaml "$@" --model "$SERVED_MODEL_NAME"' in node
        assert '--reasoning-parser "$REASONING_PARSER"' in node
        assert "--reasoning-parser qwen3" not in node
        assert 'REASONING_PARSER="${REASONING_PARSER:-}"' in submit
    for role in ("ANALYZER", "SIMPLIFIER", "EVALUATOR"):
        assert f"LSG_{role}_BASE_URL=http://127.0.0.1:8000/v1" in node
    assert "--host 127.0.0.1 --port 8000" in node
    assert 'MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"' in submit
    assert node.index('touch "$JOB_DIR/success"') > node.index("uv run --no-sync")
    assert 'touch "$JOB_DIR/ack-$rank"; exit 0' in node
    assert 'trap cleanup EXIT' in node
    barrier = node.split('for ((worker = 1; worker < EXPECTED_NODES; worker++)); do', 1)[1].split('\nelse\n', 1)[0]
    assert 'services_alive' not in barrier
