"""Contratos offline; não instala pacotes CUDA nem exige acesso a GPU."""
from pathlib import Path
import tomllib

from packaging.markers import default_environment
from packaging.requirements import Requirement

ROOT = Path(__file__).resolve().parents[1]


def test_hpc_dependencies_are_platform_scoped_and_locked():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    requirements = [Requirement(s) for s in project["project"]["optional-dependencies"]["hpc"]]
    assert {r.name for r in requirements} == {"vllm", "ray", "transformers"}
    linux = {**default_environment(), "sys_platform": "linux", "platform_machine": "x86_64", "python_version": "3.12", "python_full_version": "3.12.13"}
    assert all(r.marker.evaluate(linux) for r in requirements)
    assert not any(r.marker.evaluate({**linux, "sys_platform": "darwin"}) for r in requirements)
    assert not any(r.marker.evaluate({**linux, "python_version": "3.13", "python_full_version": "3.13.0"}) for r in requirements)
    lock = tomllib.loads((ROOT / "uv.lock").read_text())
    for req in requirements:
        assert any(p["name"] == req.name and req.specifier.contains(p["version"]) for p in lock["package"])
    assert [{"extra": "hpc"}, {"extra": "metrics"}] in project["tool"]["uv"]["conflicts"]


def test_online_preparation_and_offline_job():
    prep = (ROOT / "santos_dumont/prepare_env.sh").read_text()
    submit = (ROOT / "santos_dumont/submit.sbatch").read_text()
    node = (ROOT / "santos_dumont/node.sh").read_text()
    assert "uv sync --locked --extra hpc --no-dev --python 3.12" in prep
    assert 'UV_PROJECT_ENVIRONMENT="$ROOT/.venv"' in prep
    assert "UV_PYTHON_INSTALL_DIR" in prep
    assert "SLURM_JOB_ID" in prep
    assert "uv pip check" in prep
    for job in (submit, node):
        assert "uv sync" not in job
        assert "pip install" not in job
    for flag in ("UV_OFFLINE=1", "UV_NO_SYNC=1", "UV_PYTHON_DOWNLOADS=never", "HF_HUB_OFFLINE=1", "TRANSFORMERS_OFFLINE=1"):
        assert flag in submit
    assert "--no-sync --offline --extra hpc" in node
