"""T00 aggregator: raw_records.jsonl -> baseline/raw_runs.jsonl, stage_samples.jsonl.

Run after all harness invocations:

    python benchmarks/ultrafast/aggregate_runs.py \
        --records /Users/alansynn/Workspace/solweig_ultrafast_artifacts/raw_records.jsonl \
        --out-dir benchmarks/ultrafast/baseline
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    records = [
        json.loads(line)
        for line in Path(args.records).read_text().splitlines() if line.strip()
    ]
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    now = datetime.now(timezone.utc).isoformat()
    with open(out_dir / "raw_runs.jsonl", "w") as fh:
        for r in records:
            fh.write(json.dumps({**r, "recorded_at_utc": now}) + "\n")

    stage_keys = [
        "admission_s", "forcing_s", "compose_s", "read_window_s",
        "svf_seconds", "time_loop_seconds", "solve_s", "total_s",
    ]
    with open(out_dir / "stage_samples.jsonl", "w") as fh:
        for r in records:
            t = r.get("timings_s", {})
            fh.write(json.dumps({
                "run_id": r["run_id"], "mode": r["mode"], "case": r.get("case"),
                "phase": r["phase"], "rep": r["rep"], "recorded_at_utc": now,
                **{k: t.get(k) for k in stage_keys if k in t},
            }) + "\n")

    by_run: dict[str, list[dict]] = {}
    for r in records:
        by_run.setdefault(r["run_id"], []).append(r)
    summary = {}
    for run_id, runs in by_run.items():
        samples = [r for r in runs if r["phase"] == "sample"]
        stage_acc: dict[str, list[float]] = {}
        for r in samples:
            for k, v in r.get("timings_s", {}).items():
                stage_acc.setdefault(k, []).append(v)
        summary[run_id] = {
            "n_samples": len(samples),
            "n_warmup": sum(1 for r in runs if r["phase"] == "warmup"),
            "per_stage_s": {
                k: {"min": min(v), "max": max(v),
                    "mean": sum(v) / len(v)}
                for k, v in stage_acc.items()
            },
        }
    (out_dir / "stage_summary.json").write_text(json.dumps({
        "recorded_at_utc": now, "per_run": summary,
    }, indent=2))
    print(f"aggregated {len(records)} records into raw_runs.jsonl / "
          f"stage_samples.jsonl / stage_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
