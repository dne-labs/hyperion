from __future__ import annotations

import asyncio
import time
import uuid
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

from hyperion_sim.metrics import metrics_response, record_request
from hyperion_sim.runtime import RuntimeState, runtime


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    model: str = "simulated-llama-3-8b"
    messages: list[Message]
    max_tokens: int = Field(default=128, ge=1, le=4096)
    temperature: float = Field(default=0.7, ge=0, le=2)
    stream: bool = False


@asynccontextmanager
async def lifespan(_: FastAPI):
    yield


app = FastAPI(title="Hyperion Gateway", version="0.1.0", lifespan=lifespan)


def prompt_from(messages: list[Message]) -> str:
    return "\n".join(f"{message.role}: {message.content}" for message in messages)


@app.get("/healthz")
async def health() -> dict:
    return {"status": "ok", "mode": "simulation"}


@app.get("/topology")
async def topology() -> dict:
    return runtime.snapshot()


@app.post("/topology")
async def update_topology(update: RuntimeState) -> dict:
    runtime.apply(update)
    return runtime.snapshot()


@app.get("/metrics")
async def metrics() -> Response:
    return metrics_response()


@app.post("/v1/chat/completions")
async def chat(request: ChatRequest) -> dict:
    if request.stream:
        raise HTTPException(400, "Streaming is planned for Phase 2; use stream=false in the simulator.")
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
        "hyperion": outcome.as_dict() | {"e2e_seconds": round(total, 4)},
    }
