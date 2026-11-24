from __future__ import annotations

import asyncio
import hashlib
import math
import time
from dataclasses import asdict, dataclass


@dataclass
class RuntimeState:
    prefill_replicas: int = 1
    decode_replicas: int = 1
    prefill_batch_size: int = 16
    decode_batch_size: int = 32
    aggressive_prefix_cache: bool = False
    speculative_decoding: bool = False


@dataclass
class ExecutionOutcome:
    prompt_tokens: int
    prefill_seconds: float
    decode_seconds: float
    cache_hit: bool
    prefill_queue_depth: int
    decode_queue_depth: int

    def as_dict(self) -> dict:
        return asdict(self)


class HyperionRuntime:
    """CPU-only performance model; replace it with vLLM/SGLang adapters in Phase 2."""
    def __init__(self) -> None:
        self.state = RuntimeState()
        self.prefixes: dict[str, float] = {}
        self.prefill_active = 0
        self.decode_active = 0

    def snapshot(self) -> dict:
        return {"mode": "simulated", "state": asdict(self.state), "cache_entries": len(self.prefixes), "queues": {"prefill": self.prefill_active, "decode": self.decode_active}}

    def apply(self, state: RuntimeState) -> None:
        self.state = state

    @staticmethod
    def _tokens(text: str) -> int:
        return max(1, math.ceil(len(text.split()) * 1.35))

    async def execute(self, prompt: str, output_tokens: int) -> ExecutionOutcome:
        prompt_tokens = self._tokens(prompt)
        # Cache common first 512 prompt characters, approximating shared system/context prefixes.
        key = hashlib.sha256(prompt[:512].encode()).hexdigest()
        cache_hit = key in self.prefixes
        self.prefill_active += 1
        prefill_queue = max(0, self.prefill_active - self.state.prefill_replicas * self.state.prefill_batch_size)
        prefill_rate = 450 * self.state.prefill_replicas
        prefill_seconds = (prompt_tokens / prefill_rate) * (0.18 if cache_hit else 1.0)
        await asyncio.sleep(min(prefill_seconds, 0.15))
        self.prefill_active -= 1
        self.prefixes[key] = time.time()

        self.decode_active += 1
        decode_queue = max(0, self.decode_active - self.state.decode_replicas * self.state.decode_batch_size)
        decode_rate = 85 * self.state.decode_replicas * (1.6 if self.state.speculative_decoding else 1.0)
        decode_seconds = output_tokens / decode_rate
        await asyncio.sleep(min(decode_seconds, 0.2))
        self.decode_active -= 1
        return ExecutionOutcome(prompt_tokens, prefill_seconds, decode_seconds, cache_hit, prefill_queue, decode_queue)


runtime = HyperionRuntime()
