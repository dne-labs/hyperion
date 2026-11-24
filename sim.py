"""
sim.py — ties it together: synthetic traffic -> scheduler -> backend -> controller.

Now with:
- real arrival timing (exponential inter-arrival) instead of one-request-per-loop
- a decode stage: continuous batching across all in-flight requests, one
  generated token per request per scheduling step
- explicit idle-time advancement when there's nothing to do, so
  gpu_busy_frac stops being trivially 100%

Run: python3 sim.py
"""

import random
from backend import GPU_PROFILES, BatchSpec
from simulated_backend import SimulatedBackend
from controller import AgenticController, ServingConfig


def make_traffic(num_requests: int, rng: random.Random, mean_interarrival: float = 0.15):
    """Synthetic trace with real arrival timestamps and a deliberate burst
    of long-context requests partway through."""
    reqs = []
    t = 0.0
    for i in range(num_requests):
        t += rng.expovariate(1.0 / mean_interarrival)
        if 30 <= i < 60:
            context_len = rng.randint(6000, 12000)   # the long-context burst
            decode_len = rng.randint(50, 150)
        else:
            context_len = rng.randint(200, 1500)
            decode_len = rng.randint(50, 300)
        reqs.append({"id": f"req_{i}", "arrival_time": t,
                     "context_len": context_len, "decode_len": decode_len})
    return reqs


def main():
    rng = random.Random(42)
    backend = SimulatedBackend(GPU_PROFILES["A100_80G"], num_devices=2, seed=42)
    config = ServingConfig()
    controller = AgenticController(backend, config)

    traffic = make_traffic(num_requests=90, rng=rng)
    model_params = 7e9  # a 7B-class model, for the FLOPs estimate

    pending = list(traffic)          # not yet arrived
    active = {}                      # request_id -> remaining decode tokens
    window = []                      # rolling context lengths for avg_context_len signal

    while pending or active:
        wall_time = backend.metrics().sim_time

        # admit anything that has arrived
        arrived = [r for r in pending if r["arrival_time"] <= wall_time]
        pending = [r for r in pending if r["arrival_time"] > wall_time]

        if not arrived and not active:
            # nothing to do right now -> jump the clock to the next arrival
            next_t = min(r["arrival_time"] for r in pending)
            backend.advance_idle(next_t - wall_time)
            continue

        # prefill each newly arrived request (one batch each, for clarity)
        for req in arrived:
            batch = BatchSpec(
                request_ids=[req["id"]], stage="prefill",
                num_tokens=req["context_len"], model_params=model_params,
                attn_variant=config.attn_variant,
            )
            backend.submit_batch(batch)
            active[req["id"]] = req["decode_len"]
            window.append(req["context_len"])
        window[:] = window[-20:]

        # one decode step across every active request (continuous batching)
        if active:
            batch = BatchSpec(
                request_ids=list(active.keys()), stage="decode",
                num_tokens=len(active), model_params=model_params,
                attn_variant=config.attn_variant,
            )
            backend.submit_batch(batch)
            finished = []
            for rid in active:
                active[rid] -= 1
                if active[rid] <= 0:
                    finished.append(rid)
            for rid in finished:
                del active[rid]
                backend.release(rid)

        avg_context_len = sum(window) / len(window) if window else 0.0
        # queue_depth = concurrent in-flight requests (real backlog/load signal).
        # len(pending) would be wrong here — it counts requests that haven't
        # arrived yet in the trace, which the system can't know about yet.
        controller.tick(queue_depth=len(active), avg_context_len=avg_context_len)

    print("=== Hyperion sim: reconfiguration timeline ===")
    if config.log:
        for line in config.log:
            print(line)
    else:
        print("(no reconfigurations triggered — thresholds may need tuning)")

    m = backend.metrics()
    print("\n=== final backend metrics ===")
    print(f"sim_time (wall):  {m.sim_time:.2f}s (simulated)")
    print(f"busy_time:        {m.busy_time:.2f}s")
    print(f"gpu_busy_frac:    {m.gpu_busy_frac:.2%}")
    print(f"hbm_used (final): {m.hbm_used_bytes / (1024**3):.4f} GB / "
          f"{m.hbm_capacity_bytes / (1024**3):.2f} GB  (0 at the end is correct — queue drained)")
    print(f"hbm_used (peak):  {backend.peak_hbm_used_bytes / (1024**3):.4f} GB")
    print(f"final config:     TP={config.tensor_parallel_degree}, "
          f"spec_decode={config.speculative_decoding}, "
          f"chunked_prefill={config.chunked_prefill}")


if __name__ == "__main__":
    main()