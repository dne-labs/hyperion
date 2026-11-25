from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_profile(run_dir: Path) -> dict:
    profile_path = run_dir / "hyperion_profile.json"

    if not profile_path.is_file():
        raise FileNotFoundError(f"Profile not found: {profile_path}")

    return json.loads(profile_path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare two saved Vidur performance profiles."
    )
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    args = parser.parse_args()

    baseline = load_profile(args.baseline)
    candidate = load_profile(args.candidate)

    # The workloads must match, or the comparison would be unfair.
    workload_fields = (
        "model",
        "gpu_profile",
        "qps",
        "request_count",
        "prompt_tokens",
        "output_tokens",
    )

    mismatched_fields = [
        field
        for field in workload_fields
        if baseline[field] != candidate[field]
    ]

    if mismatched_fields:
        parser.error(
            "Profiles have different workloads: "
            + ", ".join(mismatched_fields)
        )

    metrics = (
        ("TTFT p95", "ttft_p95_ms", "ms"),
        ("E2E p95", "e2e_p95_seconds", "sec"),
        ("TPOT p95", "tpot_p95_ms", "ms/token"),
        ("Scheduling delay p95", "scheduling_delay_p95_ms", "ms"),
    )

    print(
        f"Comparing {baseline['replicas']} replica(s) "
        f"against {candidate['replicas']} replica(s)"
    )
    print(
        f"Same workload: {baseline['model']}, {baseline['qps']} QPS, "
        f"{baseline['request_count']} requests"
    )
    print()

    for label, metric_key, unit in metrics:
        before = baseline["latency"][metric_key]
        after = candidate["latency"][metric_key]

        if before == 0:
            print(f"{label}: cannot calculate change from zero")
            continue

        reduction = (before - after) / before * 100
        print(
            f"{label}: {before:.2f} → {after:.2f} {unit} "
            f"({reduction:.1f}% lower)"
        )


if __name__ == "__main__":
    main()