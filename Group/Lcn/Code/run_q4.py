"""Enumerate, independently verify, and export Q4 over the frozen Q3 schedule."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time

sys.dont_write_bytecode = True
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "lcn_q4_matplotlib"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from q3.data import load_data
from q3.physics import Terrain
from q3.verify import verify as verify_q3
from q4.model import solve
from q4.verify import verify


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8", newline="\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    lcn = Path(__file__).resolve().parent.parent
    parser.add_argument("--q3", type=Path, default=lcn / "Results/Q3/solution.json")
    parser.add_argument("--output", type=Path, default=lcn / "Results/Q4")
    parser.add_argument("--verify-only", action="store_true",
                        help="Verify saved Q4 and output fingerprints without rewriting files")
    args = parser.parse_args()
    begun = time.perf_counter()
    started = datetime.now(timezone.utc).isoformat()
    data = load_data()
    root = Path(data["repo_root"])
    q3_path, output = args.q3.resolve(), args.output.resolve()
    if not output.is_relative_to(root):
        parser.error("Output must stay inside this repository")
    q3 = json.loads(q3_path.read_text(encoding="utf-8"))
    if q3.get("source_sha256") != data["source_sha256"]:
        raise RuntimeError("Q3 input hashes differ from the current original attachments")
    inputs = [Path(p) for p in data["source_paths"].values()]
    inputs.extend([q3_path, root / "D题/结果提交模板.xlsx"])
    input_hashes = {p.relative_to(root).as_posix(): sha(p) for p in inputs}
    code_dir = Path(__file__).resolve().parent
    code_files = [Path(__file__), *sorted((code_dir / "q3").glob("*.py")),
                  *sorted((code_dir / "q4").glob("*.py"))]
    code_hashes = {p.relative_to(root).as_posix(): sha(p) for p in code_files}

    print("Rechecking frozen Q3 flight, deadline, energy and continuous communication constraints...", flush=True)
    q3_audit = verify_q3(q3, data, Terrain(data["dem_path"]))
    if not q3_audit["passed"]:
        raise RuntimeError(f"Frozen Q3 failed validation: {q3_audit['errors']}")

    if args.verify_only:
        result = json.loads((output / "solution.json").read_text(encoding="utf-8"))
        manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        if result["source"]["input_sha256"] != input_hashes:
            raise RuntimeError("Saved Q4 was computed from different inputs")
        for item in manifest["outputs"]:
            if sha(root / item["path"]) != item["sha256"]:
                raise RuntimeError(f"Saved output fingerprint differs: {item['path']}")
        for path, digest in manifest["code_sha256"].items():
            if sha(root / path) != digest:
                raise RuntimeError(f"Recorded implementation has changed: {path}")
        audit = verify(result, q3, data)
        print(json.dumps(audit, ensure_ascii=False, indent=2))
        if not audit["passed"]:
            raise SystemExit(1)
        return

    print("Enumerating every nonempty two- and three-group partition...", flush=True)
    result = solve(q3, data)
    result["source"] = {
        "q3_solution": q3_path.relative_to(root).as_posix(),
        "q3_solution_sha256": sha(q3_path),
        "input_sha256": input_hashes,
        "q3_metrics": q3_audit["metrics"],
        "q3_communication": q3_audit["communication"],
    }
    audit = verify(result, q3, data)
    if not audit["passed"]:
        raise RuntimeError(f"Q4 independent validation failed: {audit['errors']}")
    print(f"Independent Q4 audit passed {audit['check_count']} checks.", flush=True)
    for path, digest in {**input_hashes, **code_hashes}.items():
        if sha(root / path) != digest:
            raise RuntimeError(f"Input or implementation changed during execution: {path}")

    from q4.report import export_report

    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "solution.json", result)
    write_json(output / "validation.json", audit)
    export_report(result, q3, data, output)
    for path, digest in {**input_hashes, **code_hashes}.items():
        if sha(root / path) != digest:
            raise RuntimeError(f"Input or implementation changed during reporting: {path}")
    versions = [f"Python {platform.python_version()}"]
    versions.extend(f"{p} {importlib.metadata.version(p)}" for p in ["numpy", "matplotlib", "openpyxl", "rasterio"])
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip())
    manifest = {
        "schema_version": 1,
        "claim_id": "Q4_FIXED_Q3_NO_RELAY_DUPLICATION_PARTITION_OPTIMUM",
        "repository": {"commit": commit, "dirty": dirty},
        "command": subprocess.list2cmdline([sys.executable, *sys.argv]),
        "environment": {"software": versions, "hardware": platform.platform()},
        "mathematics": {
            "assertion_tested": "Exhaustive canonical partitions and attainable minimum per-type resources under frozen Q3 tasks, no relay duplication and exchangeable same-type identities.",
            "coefficient_domain": "Exact integer assignments and counts; inherited IEEE-754 float64 times with strict half-open comparisons and no overlap tolerance.",
            "conventions": "m,s,kg,kWh; resource occupancy includes full recharge and relay turnaround; workload excludes these post-return periods.",
            "inputs": list(input_hashes),
            "bounds": {"service_areas": 15, "atoms": len(result["atoms"]), "groups": [2, 3],
                       "enumerated": {k: p["enumerated"] for k, p in result["partitions"].items()},
                       "fixed_q3": True, "relay_duplication": False, "entity_reassignment": True},
            "non_claims": ["No optimum over alternative Q3 schedules or duplicated relay sorties",
                           "No guarantee of real weather, propagation or subpixel terrain"],
        },
        "randomness": {"used": False, "generator": "none", "seed": None},
        "run": {"started_at": started, "runtime_seconds": time.perf_counter() - begun, "exit_status": 0},
        "outputs": [{"path": p.relative_to(root).as_posix(), "sha256": sha(p)}
                    for p in sorted(output.iterdir()) if p.is_file() and p.name != "manifest.json"],
        "checks": ["Fresh Q3 flight and continuous radio audit", "Independent dependency components",
                   "Complete labelled-assignment cross-check of canonical enumeration",
                   "Independent active-at-start interval peaks", "Disjoint entity allocations with recharge",
                   "Resource gaps, workload CV, lexicographic optimum and Pareto frontier",
                   "Inputs and implementation unchanged during execution"],
        "result": (f"Implementation and all {sum(p['enumerated'] for p in result['partitions'].values())} "
                   "partitions verified for the frozen supplied Q3 instance; minimum total "
                   "type-specific shortages: " + "; ".join(
                       f"K={k}: {p['selected']['shortage_total']} units"
                       for k, p in result["partitions"].items())),
        "residual_risks": ["Task-arrangement interpretation permits relabelling exchangeable same-type entities",
                           "Q3 physical assumptions are inherited; finite partition optimality does not improve the underlying Q3 heuristic"],
        "input_sha256": input_hashes,
        "code_sha256": code_hashes,
    }
    write_json(output / "manifest.json", manifest)
    for k, part in result["partitions"].items():
        selected = part["selected"]
        print(f"K={k}: {part['enumerated']} partitions; selected {selected['id']}; "
              f"resources={selected['total_resources']}, shortage={selected['shortage_total']}, "
              f"CV={selected['work_cv']:.6f}")
    print(f"PASS: {output}; elapsed {time.perf_counter()-begun:.2f} s")


if __name__ == "__main__":
    main()
