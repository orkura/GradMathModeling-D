"""Solve, independently verify, and export transport-only problem 2."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time

sys.dont_write_bytecode = True
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "lcn_q2_matplotlib"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from q3.data import load_data
from q3.physics import Terrain
from q2.solver import solve
from q2.verify import verify


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8", newline="\n")


def main() -> None:
    lcn = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--starts", type=int, default=48)
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument("--max-stops", type=int, default=3)
    parser.add_argument("--neighbors", type=int, default=4)
    parser.add_argument("--refine-seconds", type=float, default=120.0)
    parser.add_argument("--output", type=Path, default=lcn / "Results/Q2")
    parser.add_argument("--verify-only", action="store_true",
                        help="Read and audit the saved result and its provenance without rewriting files")
    args = parser.parse_args()
    if (args.starts < 1 or not 1 <= args.max_stops <= 4 or not 0 <= args.neighbors <= 6
            or not math.isfinite(args.refine_seconds) or args.refine_seconds < 0):
        parser.error("Require starts>=1, 1<=max-stops<=4, 0<=neighbors<=6 and finite nonnegative refinement time")
    begun, started = time.perf_counter(), datetime.now(timezone.utc).isoformat()
    data = load_data()
    root, output = Path(data["repo_root"]), args.output.resolve()
    if not output.is_relative_to(root):
        parser.error("Output must stay inside this repository")
    input_files = [Path(p) for p in data["source_paths"].values()]
    input_files.append(root / "D题/结果提交模板.xlsx")
    input_hashes = {p.relative_to(root).as_posix(): sha(p) for p in input_files}
    code_dir = Path(__file__).resolve().parent
    code_files = [Path(__file__), *sorted((code_dir / "q2").glob("*.py")),
                  *[code_dir / "q3" / name for name in ("data.py", "physics.py", "solver.py")]]
    code_hashes = {p.relative_to(root).as_posix(): sha(p) for p in code_files}
    terrain = Terrain(data["dem_path"])

    if args.verify_only:
        result = json.loads((output / "solution.json").read_text(encoding="utf-8"))
        manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        if result["input_sha256"] != input_hashes:
            raise RuntimeError("Saved Q2 was generated from different input files")
        for item in manifest["outputs"]:
            if sha(root / item["path"]) != item["sha256"]:
                raise RuntimeError(f"Saved output fingerprint changed: {item['path']}")
        for path, digest in manifest["code_sha256"].items():
            if sha(root / path) != digest:
                raise RuntimeError(f"Recorded implementation changed: {path}")
        audit = verify(result, data, terrain)
        print(json.dumps({k: audit[k] for k in ("passed", "check_count", "error_count", "errors", "metrics")},
                         ensure_ascii=False, indent=2))
        if not audit["passed"]:
            raise SystemExit(1)
        return

    config = {key: getattr(args, key) for key in ("starts", "seed", "max_stops", "neighbors", "refine_seconds")}
    print(f"Q2: {len(data['boxes'])} boxes, {data['totals']['hard_deadline_boxes']} hard deadlines; "
          "transport-only batching, routing and resource scheduling.", flush=True)
    result = solve(data, terrain, config)
    result["source_sha256"] = data["source_sha256"]
    result["input_sha256"] = input_hashes
    result["run_config"] = config
    result["assumptions"] = [
        "Shared data and transport physics code are reused; no Q3/Q4 solution, communication constraint or relay task enters Q2.",
        "Local affine WGS84 metric coordinates and all touched original DEM pixels; cruise altitude=max(DEM maximum+50m, endpoint work altitudes).",
        "Horizontal energy=usable battery energy*distance/load-dependent range; climb energy=(empty mass+remaining payload)*g*climb/(efficiency*3.6e6).",
        "No extra descent or transport handover energy; supplied descent efficiency is zero and handover power is unspecified.",
        "Preparation plus per-box loading occupies airframe and battery; all boxes unloaded at a stop finish after its handover time.",
        "Same-type batteries may switch airframes; each starts full and must recharge fully before reuse; charging is parallel without unspecified port limits.",
        "Medical due dates and first-delivery deadlines are hard constraints; other due dates enter weighted tardiness.",
        "Bounded candidate routes and greedy batch prefixes, followed by integer-second fixed-structure CP-SAT scheduling; no global completion-time, energy or sortie optimum claim.",
    ]
    audit = verify(result, data, terrain)
    if not audit["passed"]:
        raise RuntimeError(f"Independent Q2 audit failed: {audit['errors']}")
    result["validation"] = audit
    for path, digest in {**input_hashes, **code_hashes}.items():
        if sha(root / path) != digest:
            raise RuntimeError(f"Input or implementation changed during execution: {path}")

    from q2.report import write_report

    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "solution.json", result)
    write_json(output / "validation.json", audit)
    write_json(output / "search_history.json", result["search_history"])
    write_report(result, data, terrain, output)
    # Verify the serialized scientific artifact, not only the in-memory object.
    serialized = json.loads((output / "solution.json").read_text(encoding="utf-8"))
    if not verify(serialized, data, terrain)["passed"]:
        raise RuntimeError("Serialized Q2 solution failed independent verification")
    for path, digest in {**input_hashes, **code_hashes}.items():
        if sha(root / path) != digest:
            raise RuntimeError(f"Input or implementation changed during export: {path}")
    manifest = {
        "schema_version": 1,
        "claim_id": "Q2_SUPPLIED_INSTANCE_TRANSPORT_FEASIBLE_SCHEDULE",
        "repository": {
            "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
            "dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip()),
        },
        "command": subprocess.list2cmdline([sys.executable, *sys.argv]),
        "environment": {"software": [f"Python {platform.python_version()}", *[
            f"{p} {importlib.metadata.version(p)}" for p in ("numpy", "rasterio", "matplotlib", "openpyxl", "ortools")]],
            "hardware": platform.platform()},
        "mathematics": {
            "assertion_tested": "Each supplied box is delivered once, respecting payload, volume, return SOC, hard deadlines and actual airframe/battery inventory with full recharge.",
            "coefficient_domain": "IEEE-754 float64 physical calculation; time tolerance 1e-5s, energy tolerance 1e-6kWh; strict half-open resource comparisons. Conservative integer-second CP-SAT scheduling.",
            "conventions": "m,s,kg,kWh; shared appendix-2 supplemental energy formula; local affine WGS84 and original piecewise-constant DEM; no radio constraints.",
            "inputs": list(input_hashes), "bounds": result["search_bounds"],
            "non_claims": ["No global joint optimum of batching, routing, energy and makespan",
                           "No radio feasibility guarantee in Q2", "No real-weather or subpixel terrain certificate"],
        },
        "randomness": {"used": True, "generator": "Python random.Random; single-worker CP-SAT", "seed": args.seed},
        "run": {"started_at": started, "runtime_seconds": time.perf_counter() - begun, "exit_status": 0},
        "outputs": [{"path": p.relative_to(root).as_posix(), "sha256": sha(p)}
                    for p in sorted(output.iterdir()) if p.is_file() and p.name != "manifest.json"],
        "checks": ["Independent per-leg load, DEM, time, energy and SOC reconstruction",
                   "Exact box coverage and medical/first-delivery deadlines",
                   "Same-type entity inventory and nonoverlap including recharge",
                   "Summary metrics independently recomputed", "JSON round-trip audit",
                   "Inputs and implementation unchanged during execution"],
        "result": f"Feasible supplied transport instance verified by {audit['check_count']} independent checks with zero errors; bounded search, not a full-problem optimum proof.",
        "residual_risks": ["Greedy packing and bounded route family",
                           "CP-SAT wall-clock limits can change incumbents across environments",
                           "Shared physical assumptions inherit the stated energy and DEM conventions",
                           "Legacy manifest v1 lacks machine-enforced resource caps"],
        "input_sha256": input_hashes, "code_sha256": code_hashes,
    }
    write_json(output / "manifest.json", manifest)
    print(json.dumps(result["metrics"], ensure_ascii=False, indent=2), flush=True)
    print(f"PASS: {audit['check_count']} checks; {output}; elapsed {time.perf_counter()-begun:.2f}s", flush=True)


if __name__ == "__main__":
    main()
