from __future__ import annotations

import asyncio
import logging
import os

import httpx
from hyperion_sim.runtime import RuntimeState

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
INTERVAL = int(os.getenv("HYPERION_CONTROLLER_INTERVAL_SECONDS", "2"))
GATEWAY_URL = os.getenv("HYPERION_GATEWAY_URL", "http://gateway:8000")


def decide(snapshot: dict) -> tuple[RuntimeState, str]:
    state = RuntimeState(**snapshot["state"])
    queues = snapshot["queues"]
    if queues["prefill"] > state.prefill_replicas * state.prefill_batch_size:
        state.prefill_replicas = min(8, state.prefill_replicas + 1)
        return state, "scale_prefill"
    if queues["decode"] > state.decode_replicas * state.decode_batch_size:
        state.decode_replicas = min(8, state.decode_replicas + 1)
        return state, "scale_decode"
    if snapshot["cache_entries"] > 20:
        state.aggressive_prefix_cache = True
        state.speculative_decoding = True
        return state, "enable_cache_and_speculation"
    return state, "hold"


async def run() -> None:
    async with httpx.AsyncClient(timeout=3) as client:
        while True:
            try:
                before = (await client.get(f"{GATEWAY_URL}/topology")).json()
                next_state, decision = decide(before)
                after = (await client.post(f"{GATEWAY_URL}/topology", json=next_state.__dict__)).json()
                await client.post(f"{GATEWAY_URL}/controller/decision", params={"decision": decision})
                logging.info("controller=%s topology=%s", decision, after["state"])
            except httpx.HTTPError as error:
                logging.warning("controller could not reach gateway: %s", error)
            await asyncio.sleep(INTERVAL)


if __name__ == "__main__":
    asyncio.run(run())
