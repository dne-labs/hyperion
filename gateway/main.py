from __future__ import annotations

import asyncio
import csv
import os
import time
import uuid

import httpx
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, Field

from hyperion_sim.metrics import metrics_response, record_request
from hyperion_sim.runtime import RuntimeState, runtime
import json
from pathlib import Path


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    model: str = "simulated-llama-3-8b"
    messages: list[Message]
    max_tokens: int = Field(default=128, ge=1, le=4096)
    temperature: float = Field(default=0.7, ge=0, le=2)
    stream: bool = False


class LoadRequest(BaseModel):
    requests: int = Field(default=12, ge=1, le=100)
    max_tokens: int = Field(default=250, ge=1, le=1024)
    shared_prefix: str = "You are Hyperion, an inference serving operating system."


@asynccontextmanager
async def lifespan(_: FastAPI):
    yield


app = FastAPI(title="Hyperion Gateway", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3001", "http://127.0.0.1:3001",
        "http://localhost:8501", "http://127.0.0.1:8501",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)


def prompt_from(messages: list[Message]) -> str:
    return "\n".join(f"{message.role}: {message.content}" for message in messages)


@app.get("/healthz")
async def health() -> dict:
    backend = os.getenv("HYPERION_BACKEND", "simulator")
    return {
        "status": "ok",
        "mode": "vllm_cpu" if backend == "vllm_cpu" else "simulation",
        "model": os.getenv(
            "HYPERION_VLLM_MODEL",
            "Qwen/Qwen2.5-0.5B-Instruct",
        )
        if backend == "vllm_cpu"
        else "simulated-llama-3-8b",
    }


@app.get("/topology")
async def topology() -> dict:
    return runtime.snapshot()


@app.post("/topology")
async def update_topology(update: RuntimeState) -> dict:
    runtime.apply(update)
    return runtime.snapshot()


@app.post("/controller/decision")
async def controller_decision(decision: str) -> dict:
    runtime.record_controller_decision(decision)
    return runtime.snapshot()


@app.get("/metrics")
async def metrics() -> Response:
    local_metrics = metrics_response()
    if os.getenv("HYPERION_BACKEND", "simulator") != "vllm_cpu":
        return local_metrics

    backend_url = os.getenv("HYPERION_VLLM_URL", "http://vllm-cpu:8000")
    try:
        async with httpx.AsyncClient(timeout=2) as client:
            upstream = await client.get(f"{backend_url}/metrics")
        if upstream.is_success:
            return Response(
                content=local_metrics.body + b"\n" + upstream.content,
                media_type=local_metrics.media_type,
            )
    except httpx.HTTPError:
        # Keep the Gateway's own metrics available while the optional model
        # service starts or restarts.
        pass
    return local_metrics

@app.get("/optimization/runs")
async def optimization_runs() -> dict:
    project_root = Path(__file__).resolve().parents[1]
    output_dir = project_root / "simulator_output"
    runs = []
    for profile_path in sorted(output_dir.glob("*/hyperion_profile.json")):
        run_dir = profile_path.parent
        profile = json.loads(profile_path.read_text(encoding="utf-8"))
        scenario_path = run_dir / "hyperion_scenario.json"
        if scenario_path.is_file():
            profile["configuration"] = json.loads(
                scenario_path.read_text(encoding="utf-8")
            )
        else:
            profile.setdefault("configuration", {})
        samples = []
        metrics_path = run_dir / "request_metrics.csv"
        if metrics_path.is_file():
            with metrics_path.open(newline="", encoding="utf-8") as file:
                for row in csv.DictReader(file):
                    raw = row.get(
                        "decode_time_execution_plus_preemption_normalized", ""
                    )
                    if raw:
                        try:
                            samples.append(float(raw))
                        except ValueError:
                            continue
        profile["run_id"] = run_dir.name
        profile["decode_samples_seconds"] = samples
        runs.append(profile)
    return {
        "runs": runs,
        "count": len(runs),
        "source": "Vidur offline simulation",
        "hardware_note": "A100 profile estimates, not measurements from physical GPU hardware.",
    }


@app.post("/optimization/run")
async def start_optimization(request: dict) -> dict:
    optimizer_url = os.getenv("HYPERION_VIDUR_URL", "http://vidur:8100")
    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(1800.0, connect=5.0)
        ) as client:
            response = await client.post(f"{optimizer_url}/sweep", json=request)
    except httpx.TimeoutException as error:
        raise HTTPException(
            status_code=504,
            detail="Vidur took too long. Check its container logs and the saved run folders.",
        ) from error
    except httpx.HTTPError as error:
        raise HTTPException(
            status_code=503,
            detail="Vidur optimizer is not running. Start it with: docker compose --profile vidur up -d --build vidur",
        ) from error
    if not response.is_success:
        try:
            detail = response.json().get("detail", "Vidur sweep failed.")
        except ValueError:
            detail = "Vidur sweep failed."
        raise HTTPException(status_code=response.status_code, detail=detail)
    return response.json()


@app.get("/vidur-profile")
async def get_vidur_profile() -> dict:
    project_root = Path(__file__).resolve().parents[1]
    output_dir = project_root / "simulator_output"
    profiles = sorted(output_dir.glob("*/hyperion_profile.json"))

    if not profiles:
        raise HTTPException(
            status_code=404,
            detail="No Vidur profile found. Run Vidur and create a profile first.",
        )

    latest_profile = profiles[-1]
    return json.loads(latest_profile.read_text(encoding="utf-8"))


@app.post("/v1/chat/completions")
async def chat(request: ChatRequest) -> dict:
    if request.stream:
        raise HTTPException(400, "Streaming is not enabled in this Hyperion gateway yet.")

    if os.getenv("HYPERION_BACKEND", "simulator") == "vllm_cpu":
        backend_url = os.getenv("HYPERION_VLLM_URL", "http://vllm-cpu:8000")
        model_name = os.getenv(
            "HYPERION_VLLM_MODEL",
            "Qwen/Qwen2.5-0.5B-Instruct",
        )
        payload = request.model_dump()
        if payload["model"] == "simulated-llama-3-8b":
            payload["model"] = model_name
        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(300, connect=5)
            ) as client:
                upstream = await client.post(
                    f"{backend_url}/v1/chat/completions",
                    json=payload,
                )
        except httpx.TimeoutException as error:
            raise HTTPException(
                status_code=504,
                detail="The CPU model took too long to respond.",
            ) from error
        except httpx.HTTPError as error:
            raise HTTPException(
                status_code=503,
                detail="The optional vLLM CPU service is unavailable.",
            ) from error

        if not upstream.is_success:
            raise HTTPException(
                status_code=502,
                detail=f"vLLM CPU service returned HTTP {upstream.status_code}.",
            )
        result = upstream.json()
        result["hyperion"] = {
            "mode": "vllm_cpu",
            "backend": "vLLM CPU",
            "model": model_name,
        }
        return result

    prompt = prompt_from(request.messages)
    started = time.perf_counter()
    outcome = await runtime.execute(prompt, request.max_tokens)
    total = time.perf_counter() - started
    record_request(outcome, total)
    text = (
        "[Hyperion simulation] Request passed through a disaggregated "
        f"prefill/decode path. Generated {request.max_tokens} simulated tokens."
    )
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": request.model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": outcome.prompt_tokens, "completion_tokens": request.max_tokens, "total_tokens": outcome.prompt_tokens + request.max_tokens},
        "hyperion": outcome.as_dict()
        | {"e2e_seconds": round(total, 4), "mode": "simulator"},
    }

@app.get("/vidur-comparison")
async def get_vidur_comparison() -> dict:
    project_root = Path(__file__).resolve().parents[1]
    output_dir = project_root / "simulator_output"

    profile_paths = sorted(
        output_dir.glob("*/hyperion_profile.json")
    )

    if len(profile_paths) < 2:
        raise HTTPException(
            status_code=404,
            detail="Need at least two saved Vidur profiles to compare.",
        )

    baseline_path = profile_paths[-2]
    candidate_path = profile_paths[-1]

    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))

    # A comparison is fair only if the workload settings match.
    workload_fields = (
        "model",
        "gpu_profile",
        "qps",
        "request_count",
        "prompt_tokens",
        "output_tokens",
    )
    same_workload = all(
        baseline[field] == candidate[field]
        for field in workload_fields
    )

    return {
        "baseline_run": baseline_path.parent.name,
        "candidate_run": candidate_path.parent.name,
        "same_workload": same_workload,
        "baseline": baseline,
        "candidate": candidate,
    }

@app.get("/vidur-recommendation")
async def get_vidur_recommendation(
    max_ttft_p95_ms: float = 600,
    max_e2e_p95_seconds: float = 10,
    max_scheduling_delay_p95_ms: float = 400,
    max_ttft_p95_ms_hyphen: float | None = Query(default=None, alias="max-ttft-p95-ms"),
    max_e2e_p95_seconds_hyphen: float | None = Query(default=None, alias="max-e2e-p95-seconds"),
    max_scheduling_delay_p95_ms_hyphen: float | None = Query(default=None, alias="max-scheduling-delay-p95-ms"),
) -> dict:
    """Recommend the fewest tested replicas meeting the requested latency targets.

    Both underscore and kebab-case query names are accepted.
    """
    limits = {
        "ttft_p95_ms": (
            max_ttft_p95_ms if max_ttft_p95_ms_hyphen is None
            else max_ttft_p95_ms_hyphen
        ),
        "e2e_p95_seconds": (
            max_e2e_p95_seconds if max_e2e_p95_seconds_hyphen is None
            else max_e2e_p95_seconds_hyphen
        ),
        "scheduling_delay_p95_ms": (
            max_scheduling_delay_p95_ms
            if max_scheduling_delay_p95_ms_hyphen is None
            else max_scheduling_delay_p95_ms_hyphen
        ),
    }
    if any(limit <= 0 for limit in limits.values()):
        raise HTTPException(status_code=422, detail="Latency targets must be greater than zero.")

    project_root = Path(__file__).resolve().parents[1]
    profile_paths = sorted(
        (project_root / "simulator_output").glob("*/hyperion_profile.json")
    )
    if not profile_paths:
        raise HTTPException(
            status_code=404,
            detail="No Vidur profiles found. Run Vidur and generate profiles first.",
        )

    workload_fields = (
        "model",
        "gpu_profile",
        "qps",
        "request_count",
        "prompt_tokens",
        "output_tokens",
    )
    profiles = [
        (path.parent.name, json.loads(path.read_text(encoding="utf-8")))
        for path in profile_paths
    ]
    reference_run, reference = profiles[-1]
    matching = [
        (run, profile)
        for run, profile in profiles
        if all(profile.get(field) == reference.get(field) for field in workload_fields)
    ]

    evaluated = []
    for run, profile in matching:
        latency = profile.get("latency", {})
        missing_metrics = [key for key in limits if key not in latency]
        if missing_metrics:
            raise HTTPException(
                status_code=422,
                detail=f"Profile {run} is missing latency metrics: {', '.join(missing_metrics)}",
            )
        failures = [
            key for key, limit in limits.items() if latency[key] > limit
        ]
        evaluated.append({
            "run": run,
            "replicas": profile["replicas"],
            "meets_targets": not failures,
            "latency": {key: latency[key] for key in limits},
            "failed_targets": failures,
        })

    passing = [item for item in evaluated if item["meets_targets"]]
    passing.sort(key=lambda item: item["replicas"])
    recommendation = passing[0] if passing else None

    return {
        "workload": {field: reference[field] for field in workload_fields},
        "latest_run": reference_run,
        "targets": limits,
        "tested_profiles": sorted(evaluated, key=lambda item: item["replicas"]),
        "recommendation": recommendation,
        "message": (
            "Fewest tested replica count meeting all latency targets."
            if recommendation
            else "No saved profile for this workload meets all latency targets."
        ),
        "hardware_note": (
            "Vidur estimates this workload using a GPU profile; this is not a "
            "measurement from a physical GPU."
        ),
    }


@app.post("/demo/load")
async def demo_load(request: LoadRequest) -> dict:
    """Generate concurrent, shared-prefix traffic to demonstrate control decisions."""
    started = time.perf_counter()
    # Warm the shared system prefix. The following burst should reuse it.
    await runtime.execute(f"system: {request.shared_prefix}\nuser: warm cache", 1)
    prompts = [f"system: {request.shared_prefix}\nuser: Explain request number {index}." for index in range(request.requests)]
    outcomes = await asyncio.gather(*(runtime.execute(prompt, request.max_tokens) for prompt in prompts))
    for outcome in outcomes:
        record_request(outcome, outcome.prefill_seconds + outcome.decode_seconds)
    return {
        "message": "Burst completed. Refresh the dashboard and inspect last_controller_decision.",
        "requests": request.requests,
        "cache_hits": sum(outcome.cache_hit for outcome in outcomes),
        "duration_seconds": round(time.perf_counter() - started, 3),
        "topology": runtime.snapshot(),
    }
