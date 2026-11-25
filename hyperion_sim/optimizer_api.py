from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from hyperion_sim.vidur_results import build_profile, load_request_metrics

app = FastAPI(title="Hyperion Vidur Optimizer", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
    allow_methods=["GET", "POST"],
    allow_headers=["content-type"],
)
RUN_ROOT = Path(os.getenv("HYPERION_OUTPUT_DIR", "/workspace/simulator_output"))
RUN_LOCK = asyncio.Lock()


class SweepRequest(BaseModel):
    qps: float = Field(default=4.0, gt=0, le=64)
    requests: int = Field(default=32, ge=8, le=64)
    replicas: list[int] = Field(default_factory=lambda: [1, 2, 4])
    prompt_tokens: int = Field(default=2048, ge=32, le=4096)
    output_tokens: int = Field(default=512, ge=8, le=2048)
    batch_size: int = Field(default=128, ge=8, le=256)
    scheduler: Literal["sarathi", "vllm"] = "sarathi"
    chunk_size: int = Field(default=512, ge=64, le=2048)
    alternate_chunk_size: int = Field(default=256, ge=64, le=2048)


def command_for(args: SweepRequest, replicas: int, chunk_size: int) -> list[str]:
    command = [
        sys.executable, "-m", "vidur.main",
        "--replica_config_device", "a100",
        "--replica_config_model_name", "meta-llama/Meta-Llama-3-8B",
        "--cluster_config_num_replicas", str(replicas),
        "--replica_config_tensor_parallel_size", "1",
        "--request_generator_config_type", "synthetic",
        "--synthetic_request_generator_config_num_requests", str(args.requests),
        "--fixed_request_length_generator_config_prefill_tokens", str(args.prompt_tokens),
        "--fixed_request_length_generator_config_decode_tokens", str(args.output_tokens),
        "--interval_generator_config_type", "poisson",
        "--poisson_request_interval_generator_config_qps", str(args.qps),
        "--random_forrest_execution_time_predictor_config_k_fold_cv_splits", "2",
        "--random_forrest_execution_time_predictor_config_num_estimators", "250",
        "--random_forrest_execution_time_predictor_config_max_depth", "16",
        "--random_forrest_execution_time_predictor_config_min_samples_split", "2",
        "--no-metrics_config_store_plots",
        "--no-metrics_config_enable_chrome_trace",
    ]
    scheduler = args.scheduler
    command.extend([f"--{scheduler}_scheduler_config_batch_size_cap", str(args.batch_size)])
    if scheduler == "sarathi":
        command.extend([f"--{scheduler}_scheduler_config_chunk_size", str(chunk_size)])
    return command


def run_one(args: SweepRequest, replicas: int, chunk_size: int) -> dict:
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    before = {
        path.name for path in RUN_ROOT.iterdir()
        if path.is_dir()
    }
    command = command_for(args, replicas, chunk_size)
    env = os.environ.copy()
    env.update({
        "OMP_NUM_THREADS": "2",
        "OPENBLAS_NUM_THREADS": "2",
        "MKL_NUM_THREADS": "2",
        "NUMEXPR_NUM_THREADS": "2",
        "PYTHONPATH": "/workspace:/opt/vidur",
    })
    try:
        completed = subprocess.run(
            command,
            cwd="/workspace",
            env=env,
            capture_output=True,
            text=True,
            timeout=1800,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(f"Vidur timed out for {replicas} replicas.") from error
    if completed.returncode:
        tail = (completed.stdout + "\n" + completed.stderr)[-5000:]
        raise RuntimeError(
            f"Vidur failed for {replicas} replicas (exit {completed.returncode}).\n{tail}"
        )

    new_runs = [
        path for path in RUN_ROOT.iterdir()
        if path.is_dir()
        and path.name not in before
        and (path / "request_metrics.csv").is_file()
        and (path / "config.json").is_file()
    ]
    if not new_runs:
        raise RuntimeError("Vidur finished but did not write a request metrics run.")
    run_dir = max(new_runs, key=lambda path: path.name)
    scenario = {
        "optimizer": "hyperion-vidur-sweep",
        "replicas": replicas,
        "scheduler": args.scheduler,
        "batch_size": args.batch_size,
        "chunk_size": chunk_size if args.scheduler == "sarathi" else None,
        "offered_qps": args.qps,
        "request_count": args.requests,
    }
    (run_dir / "hyperion_scenario.json").write_text(
        json.dumps(scenario, indent=2), encoding="utf-8"
    )
    rows = load_request_metrics(run_dir / "request_metrics.csv")
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    profile = build_profile(rows, config)
    profile["configuration"].update({
        "scheduler": args.scheduler,
        "batch_size": args.batch_size,
        "chunk_size": scenario["chunk_size"],
    })
    (run_dir / "hyperion_profile.json").write_text(
        json.dumps(profile, indent=2), encoding="utf-8"
    )
    return {
        "run": run_dir.name,
        "replicas": replicas,
        "scheduler": args.scheduler,
        "batch_size": args.batch_size,
        "chunk_size": scenario["chunk_size"],
        "command": command,
    }


@app.get("/healthz")
async def health() -> dict:
    return {"status": "ok", "engine": "Vidur offline simulator", "device_profile": "a100"}


@app.post("/sweep")
async def sweep(request: SweepRequest) -> dict:
    replicas = sorted(set(request.replicas))
    if not replicas or any(value not in (1, 2, 4) for value in replicas):
        raise HTTPException(422, "Choose one or more tested replica counts: 1, 2, or 4.")
    candidates: list[tuple[int, int]] = [
        (replica_count, request.chunk_size) for replica_count in replicas
    ]
    if request.scheduler == "sarathi" and len(replicas) > 1:
        middle = replicas[len(replicas) // 2]
        alternative = (middle, request.alternate_chunk_size)
        if alternative not in candidates:
            candidates.append(alternative)
    if len(candidates) > 4:
        candidates = candidates[:4]

    if RUN_LOCK.locked():
        raise HTTPException(409, "A Vidur sweep is already running. Wait for it to finish.")
    async with RUN_LOCK:
        results = []
        for replica_count, chunk_size in candidates:
            try:
                result = await asyncio.to_thread(
                    run_one, request, replica_count, chunk_size
                )
                results.append(result)
            except Exception as error:
                raise HTTPException(500, str(error)) from error
    return {
        "status": "completed",
        "simulated": True,
        "note": "These are offline estimates for an A100 profile, not physical GPU measurements.",
        "completed_runs": results,
    }
