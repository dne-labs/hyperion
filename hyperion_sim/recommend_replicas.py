"""Recommend the smallest replica count that meets latency targets.

Run with:
    python -m hyperion_sim.recommend_replicas RUN_DIR_1 RUN_DIR_2 ...
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


WORKLOAD_FIELDS = (
    "model",
    "gpu_profile",
    "qps",
    "request_count",
    "prompt_tokens",
    "output_tokens",
)
LATENCY_LIMITS = {
    "ttft_p95_ms": ("TTFT p95", "ms"),
    "e2e_p95_seconds": ("E2E p95", "sec"),
    "scheduling_delay_p95_ms": ("Scheduling delay p95", "ms"),
}


def load_profile(run_dir: Path) -> dict[str, Any]:
    profile_path = run_dir / "hyperion_profile.json"
    if not profile_path.is_file():
        raise FileNotFoundError(f"Profile not found: {profile_path}")

    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    required_fields = (*WORKLOAD_FIELDS, "replicas", "latency")
    missing = [field for field in required_fields if field not in profile]
    if missing:
        raise ValueError(f"{profile_path} is missing fields: {', '.join(missing)}")

    missing_latency = [
        key for key in LATENCY_LIMITS if key not in profile["latency"]
    ]
    if missing_latency:
        raise ValueError(
            f"{profile_path} is missing latency fields: "
            + ", ".join(missing_latency)
        )

    return profile


def check_profile(
    profile: dict[str, Any],
    max_ttft_p95_ms: float,
    max_e2e_p95_seconds: float,
    max_scheduling_delay_p95_ms: float,
) -> list[str]:
    limits = {
        "ttft_p95_ms": max_ttft_p95_ms,
        "e2e_p95_seconds": max_e2e_p95_seconds,
        "scheduling_delay_p95_ms": max_scheduling_delay_p95_ms,
    }
    return [
        f"{LATENCY_LIMITS[key][0]} {profile['latency'][key]:.2f} "
        f"> {limit:.2f} {LATENCY_LIMITS[key][1]}"
        for key, limit in limits.items()
        if profile["latency"][key] > limit
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Choose the fewest simulated replicas that satisfy all latency "
            "targets for the supplied Vidur profiles."
        )
    )
    parser.add_argument(
        "run_dirs",
        nargs="+",
        type=Path,
        metavar="RUN_DIR",
        help="Vidur run directories containing hyperion_profile.json",
    )
    parser.add_argument(
        "--max-ttft-p95-ms",
        type=float,
        default=600,
        help="Maximum allowed p95 time to first token in milliseconds (default: 600)",
    )
    parser.add_argument(
        "--max-e2e-p95-seconds",
        type=float,
        default=10,
        help="Maximum allowed p95 end-to-end latency in seconds (default: 10)",
    )
    parser.add_argument(
        "--max-scheduling-delay-p95-ms",
        type=float,
        default=400,
        help="Maximum allowed p95 scheduling delay in milliseconds (default: 400)",
    )
    args = parser.parse_args()

    limits = (
        args.max_ttft_p95_ms,
        args.max_e2e_p95_seconds,
        args.max_scheduling_delay_p95_ms,
    )
    if any(limit <= 0 for limit in limits):
        parser.error("latency targets must be greater than zero")

    try:
        profiles = [(run_dir, load_profile(run_dir)) for run_dir in args.run_dirs]
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))

    reference = profiles[0][1]
    for run_dir, profile in profiles[1:]:
        mismatches = [
            field
            for field in WORKLOAD_FIELDS
            if profile[field] != reference[field]
        ]
        if mismatches:
            parser.error(
                f"{run_dir} has a different workload: "
                + ", ".join(mismatches)
            )

    profiles.sort(key=lambda item: item[1]["replicas"])
    print(
        "Targets: "
        f"TTFT p95 <= {args.max_ttft_p95_ms:g} ms, "
        f"E2E p95 <= {args.max_e2e_p95_seconds:g} sec, "
        "scheduling delay p95 <= "
        f"{args.max_scheduling_delay_p95_ms:g} ms"
    )
    print(
        f"Workload: {reference['model']}, {reference['qps']} QPS, "
        f"{reference['request_count']} requests"
    )
    print()

    passing: list[tuple[Path, dict[str, Any]]] = []
    for run_dir, profile in profiles:
        failures = check_profile(
            profile,
            args.max_ttft_p95_ms,
            args.max_e2e_p95_seconds,
            args.max_scheduling_delay_p95_ms,
        )
        status = "PASS" if not failures else "FAIL"
        latency = profile["latency"]
        print(
            f"[{status}] {profile['replicas']} replica(s): "
            f"TTFT {latency['ttft_p95_ms']:.2f} ms, "
            f"E2E {latency['e2e_p95_seconds']:.2f} sec, "
            f"scheduling {latency['scheduling_delay_p95_ms']:.2f} ms"
        )
        for failure in failures:
            print(f"       {failure}")
        if not failures:
            passing.append((run_dir, profile))

    print()
    if passing:
        run_dir, profile = passing[0]
        print(
            f"Recommendation: {profile['replicas']} replica(s) "
            f"(profile: {run_dir / 'hyperion_profile.json'})"
        )
        print(
            "This is the fewest tested replicas that meets the supplied "
            "targets."
        )
    else:
        print("No supplied profile meets all latency targets.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
