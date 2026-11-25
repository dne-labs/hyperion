"""Launch a reproducible Vidur study; requires the separate Python 3.10 Vidur environment."""
from __future__ import annotations
import argparse
import json
import subprocess
import sys
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--qps", type=float, default=4.0)
    p.add_argument("--requests", type=int, default=128)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    command = [sys.executable, "-m", "vidur.main", "--replica_config_device", "a100", "--replica_config_model_name", "meta-llama/Meta-Llama-3-8B", "--cluster_config_num_replicas", "1", "--replica_config_tensor_parallel_size", "1", "--request_generator_config_type", "synthetic", "--synthetic_request_generator_config_num_requests", str(args.requests), "--interval_generator_config_type", "poisson", "--poisson_request_interval_generator_config_qps", str(args.qps),"--random_forrest_execution_time_predictor_config_k_fold_cv_splits",
"2",
"--random_forrest_execution_time_predictor_config_num_estimators",
"250",
"--random_forrest_execution_time_predictor_config_max_depth",
"16",
"--random_forrest_execution_time_predictor_config_min_samples_split",
"2",
"--no-metrics_config_store_plots",
"--no-metrics_config_enable_chrome_trace",]
    Path("simulator_output").mkdir(exist_ok=True)
    Path("simulator_output/hyperion-vidur-command.json").write_text(json.dumps({"command": command, "purpose": "Hyperion no-GPU capacity simulation"}, indent=2))
    print(" ".join(command))
    if not args.dry_run:
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
