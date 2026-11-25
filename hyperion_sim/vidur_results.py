from __future__ import annotations

import csv
from pathlib import Path
import math
import json
from pathlib import Path
import argparse


def load_request_metrics(path: Path) -> list[dict[str, str]]:
    """Read Vidur's one-row-per-request metrics CSV."""
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))

def column_as_floats(
    rows: list[dict[str, str]],
    column: str,
) -> list[float]:
    """Convert one Vidur CSV column from text into float values."""
    return [float(row[column]) for row in rows if row[column] != ""]

def percentile(values: list[float], percent: float) -> float:
    """Return the nearest-rank percentile from a non-empty list."""
    if not values:
        raise ValueError("Cannot calculate a percentile from an empty list")

    if not 0 < percent <= 100:
        raise ValueError("percent must be between 0 and 100")

    ordered = sorted(values)
    index = math.ceil(len(ordered) * percent / 100) - 1
    return ordered[index]


def build_profile(rows: list[dict[str, str]], config: dict) -> dict:
    ttft = column_as_floats(rows, "prefill_e2e_time")
    e2e = column_as_floats(rows, "request_e2e_time")
    tpot = column_as_floats(
        rows, "decode_time_execution_plus_preemption_normalized"
    )
    scheduling_delay = column_as_floats(rows, "request_scheduling_delay")

    cluster = config["cluster_config"]
    replica = cluster["replica_config"]
    request_generator = config["request_generator_config"]
    interval_config = request_generator["interval_generator_config"]

    return {
        "model": replica["model_name"],
        "gpu_profile": replica["device"],
        "replicas": cluster["num_replicas"],
        "qps": interval_config["qps"],
        "request_count": len(rows),
        "prompt_tokens": int(rows[0]["request_num_prefill_tokens"]),
        "output_tokens": int(rows[0]["request_num_decode_tokens"]),
        "configuration": {
            "pipeline_parallel": cluster["replica_config"].get("num_pipeline_stages", 1),
            "tensor_parallel": cluster["replica_config"].get("tensor_parallel_size", 1),
            "scheduler": cluster.get("replica_scheduler_config", {}).get("name", "unknown"),
            "chunk_size": cluster.get("replica_scheduler_config", {}).get("chunk_size"),
            "batch_size": cluster.get("replica_scheduler_config", {}).get("batch_size_cap"),
        },
        "latency": {
            "ttft_p50_ms": round(percentile(ttft, 50) * 1000, 2),
            "ttft_p90_ms": round(percentile(ttft, 90) * 1000, 2),
            "ttft_p95_ms": round(percentile(ttft, 95) * 1000, 2),
            "e2e_p50_seconds": round(percentile(e2e, 50), 3),
            "e2e_p95_seconds": round(percentile(e2e, 95), 3),
            "tpot_p50_ms": round(percentile(tpot, 50) * 1000, 2),
            "tpot_p95_ms": round(percentile(tpot, 95) * 1000, 2),
            "tpot_p99_ms": round(percentile(tpot, 99) * 1000, 2),
            "scheduling_delay_p95_ms": round(
                percentile(scheduling_delay, 95) * 1000, 2
            ),
        },
    }

def latest_run_dir(root: Path = Path("simulator_output")) -> Path:
    run_dirs = [
        path
        for path in root.iterdir()
        if path.is_dir() and (path / "request_metrics.csv").is_file()
    ]

    if not run_dirs:
        raise FileNotFoundError(
            f"No Vidur request_metrics.csv files found under {root}"
        )

    # Vidur folder names begin with sortable timestamps.
    return max(run_dirs, key=lambda path: path.name)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir",
        type=Path,
        help="Vidur run folder to process; defaults to the newest run",
    )
    args = parser.parse_args()

    run_dir = args.run_dir or latest_run_dir()
    metrics_path = run_dir / "request_metrics.csv"
    rows = load_request_metrics(metrics_path)

    ttft = column_as_floats(rows, "prefill_e2e_time")
    e2e = column_as_floats(rows, "request_e2e_time")
    tpot = column_as_floats(
        rows, "decode_time_execution_plus_preemption_normalized"
    )
    scheduling_delay = column_as_floats(rows, "request_scheduling_delay")

    print(f"Loaded {len(rows)} requests")
    print(f"TTFT p50: {percentile(ttft, 50) * 1000:.2f} ms")
    print(f"TTFT p95: {percentile(ttft, 95) * 1000:.2f} ms")
    print(f"E2E p50: {percentile(e2e, 50):.2f} sec")
    print(f"E2E p95: {percentile(e2e, 95):.2f} sec")
    print(f"TPOT p50: {percentile(tpot, 50) * 1000:.2f} ms/token")
    print(f"TPOT p95: {percentile(tpot, 95) * 1000:.2f} ms/token")
    print(
        "Scheduling delay p95: "
        f"{percentile(scheduling_delay, 95) * 1000:.2f} ms"
    )

    config_path = metrics_path.parent / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))

    profile = build_profile(rows, config)
    profile_path = metrics_path.parent / "hyperion_profile.json"
    profile_path.write_text(
        json.dumps(profile, indent=2),
        encoding="utf-8",
    )

    print(f"Saved profile to {profile_path}")
    print(json.dumps(profile, indent=2))



