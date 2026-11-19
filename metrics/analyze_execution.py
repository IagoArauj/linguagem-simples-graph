"""Relatórios operacionais independentes: python -m metrics.analyze_execution."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
from uuid import uuid4

from metrics.execution_report import write_execution_report

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def create_execution_analysis(
    run_dirs: list[Path] | None = None, *, analysis_dir: Path | None = None,
    output_dir: Path | None = None,
) -> Path:
    """Cria uma análise nova sem modificar runs ou relatórios anteriores."""
    if bool(run_dirs) == (analysis_dir is not None):
        raise ValueError("Informe run_dirs ou analysis_dir, exclusivamente.")
    parent = None
    previous = None
    if analysis_dir is not None:
        path = Path(analysis_dir).resolve() / "manifest.json"
        content = path.read_bytes()
        previous = json.loads(content)
        recorded_runs = previous.get("runs")
        if not isinstance(recorded_runs, list) or not recorded_runs:
            raise ValueError("Manifesto da análise sem runs; informe --run-dir.")
        run_dirs = []
        for run in recorded_runs:
            directory = run.get("directory") if isinstance(run, dict) else None
            if not isinstance(directory, str) or not directory.strip():
                raise ValueError("Run sem directory no manifesto; informe --run-dir.")
            directory = Path(directory)
            run_dirs.append(directory if directory.is_absolute() else PROJECT_ROOT / directory)
        parent = {"path": str(path), "sha256": hashlib.sha256(content).hexdigest()}
    assert run_dirs
    created = datetime.now(timezone.utc)
    root = (PROJECT_ROOT / "output").resolve()
    output = (output_dir or root / "analyses" / f"execution_{created:%Y%m%dT%H%M%S}_{uuid4().hex[:8]}").resolve()
    if not output.is_relative_to(root) or output == root:
        raise ValueError("A pasta de análise deve ficar dentro de output/.")
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "analysis_schema_version": "1.0", "analysis_type": "execution",
        "created_at": created.isoformat(), "status": "running",
        "requested_run_directories": [str(Path(p).resolve()) for p in run_dirs],
        "parent_analysis": parent, "runs": [], "source_files": [], "warnings": [],
        "options": {"all_execution_outcomes": True},
        "aggregation": "available finite nonnegative observations per document; branch outcomes and recorded failure events separately",
        "dependencies": {"python": platform.python_version()},
        "analysis_code": {
            name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ("analyze_execution.py", "execution_report.py")
        },
    }

    def save_manifest():
        (output / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    save_manifest()
    try:
        manifest.update(write_execution_report(run_dirs, output))
        if parent is not None:
            manifest["source_files"].append(parent)
            manifest["warnings"].append(
                "Runs selecionadas pelo manifesto anterior; estatísticas recalculadas dos arquivos atuais, não dos agregados antigos.")
            old_sources = {s["path"]: s["sha256"] for s in previous.get("source_files", [])}
            for source in manifest["source_files"]:
                if source["path"] in old_sources and source["sha256"] != old_sources[source["path"]]:
                    manifest["warnings"].append(f"Fonte alterada desde a análise anterior: {source['path']}")
        manifest["status"] = "completed"
    except BaseException:
        manifest["status"] = "failed"
        raise
    finally:
        manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
        manifest["artifacts"] = sorted(p.name for p in output.iterdir() if p.is_file() and p.name != "manifest.json")
        save_manifest()
    for warning in manifest["warnings"]:
        print(f"Aviso: {warning}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tabelas de recursos, status e falhas, sem corpus, gráficos, NILC ou inferência.")
    sources = parser.add_mutually_exclusive_group(required=True)
    sources.add_argument("--run-dir", type=Path, nargs="+", help="Pastas das runs com manifest.json e results.")
    sources.add_argument("--analysis-dir", type=Path, help="Análise anterior: reutiliza as pastas das runs registradas no manifesto.")
    parser.add_argument("--output-dir", type=Path, help="Nova pasta em output/; padrão: output/analyses/execution_<timestamp>_<id>.")
    args = parser.parse_args()
    try:
        output = create_execution_analysis(args.run_dir, analysis_dir=args.analysis_dir, output_dir=args.output_dir)
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(1, f"Erro: {exc}\n")
    print(f"Análise operacional salva em: {output}")


if __name__ == "__main__":
    main()
