"""
simulated_backend.py — the "fake GPU".

Models TIME and MEMORY analytically (roofline-style), the same approach
Vidur/Frontier use: no tensors are ever multiplied, but batch latency,
KV cache occupancy, and eviction behave the way they would on real
hardware, closely enough to test scheduling/control logic.
"""

import random
from collections import deque
from backend import GPUBackend, GPUProfile, BatchSpec, BackendMetrics


class SimulatedBackend(GPUBackend):
    def __init__(self, profile: GPUProfile, num_devices: int = 1, seed: int = 0):
        self.profile = profile
        self.num_devices = num_devices
        self.rng = random.Random(seed)

        self.sim_time = 0.0
        self.busy_time = 0.0
        self.hbm_used_bytes = 0
        self.peak_hbm_used_bytes = 0
        self.hbm_capacity_bytes = profile.hbm_capacity_bytes * num_devices
        self.queue: deque = deque()
        # per-request resident KV bytes, so eviction has something to evict
        self.resident_kv: dict[str, int] = {}

    # ---- core roofline time model ------------------------------------

    def estimate_latency(self, batch: BatchSpec) -> float:
        # Compute-bound estimate: 2 * params * tokens FLOPs (fwd pass, rough)
        flops = 2 * batch.model_params * batch.num_tokens
        achievable_flops = self.profile.peak_flops_fp16 * self.profile.achieved_mfu
        compute_time = flops / achievable_flops

        # Memory-bound estimate: reading KV cache (dominant cost in decode)
        kv_bytes = batch.num_tokens * batch.kv_bytes_per_token
        mem_time = kv_bytes / self.profile.hbm_bandwidth_bytes_per_sec

        # Roofline: you're bound by whichever is larger
        per_device_time = max(compute_time, mem_time)

        # crude parallel-efficiency penalty as device count grows
        parallel_overhead = 1.0 + 0.05 * max(0, self.num_devices - 1)
        return per_device_time * parallel_overhead

    def submit_batch(self, batch: BatchSpec) -> float:
        latency = self.estimate_latency(batch)
        self.sim_time += latency
        self.busy_time += latency

        if batch.stage == "prefill":
            # one request, num_tokens = its prompt length -> that much KV created
            added = batch.num_tokens * batch.kv_bytes_per_token
            self.hbm_used_bytes += added
            for rid in batch.request_ids:
                self.resident_kv[rid] = self.resident_kv.get(rid, 0) + added // max(1, len(batch.request_ids))
        elif batch.stage == "decode":
            # one token generated per active request in the batch this step
            added_per_req = batch.kv_bytes_per_token
            total_added = added_per_req * len(batch.request_ids)
            self.hbm_used_bytes += total_added
            for rid in batch.request_ids:
                self.resident_kv[rid] = self.resident_kv.get(rid, 0) + added_per_req

        self.peak_hbm_used_bytes = max(self.peak_hbm_used_bytes, self.hbm_used_bytes)
        self._maybe_evict()
        return latency

    def advance_idle(self, dt: float) -> None:
        self.sim_time += dt  # not counted as busy_time -> lowers gpu_busy_frac correctly

    def release(self, request_id: str) -> None:
        size = self.resident_kv.pop(request_id, 0)
        self.hbm_used_bytes = max(0, self.hbm_used_bytes - size)

    def _maybe_evict(self, target_util: float = 0.92):
        """Simple LRU-ish eviction: if we're over target utilization, drop
        the oldest resident KV until we're back under. This is the hook
        where Hyperion's own eviction/quantization policy plugs in instead."""
        if self.hbm_capacity_bytes == 0:
            return
        while (self.hbm_used_bytes / self.hbm_capacity_bytes) > target_util and self.resident_kv:
            rid, size = next(iter(self.resident_kv.items()))
            del self.resident_kv[rid]
            self.hbm_used_bytes = max(0, self.hbm_used_bytes - size)

    def memory_pressure(self) -> float:
        if self.hbm_capacity_bytes == 0:
            return 0.0
        return min(1.0, self.hbm_used_bytes / self.hbm_capacity_bytes)

    def transfer_kv_block(self, num_bytes: int, cross_region: bool) -> float:
        bw = (self.profile.inter_node_bw_bytes_per_sec if cross_region
              else self.profile.intra_node_bw_bytes_per_sec)
        latency = num_bytes / bw
        self.sim_time += latency
        return latency

    def metrics(self) -> BackendMetrics:
        busy_frac = 0.0 if self.sim_time == 0 else min(1.0, self.busy_time / self.sim_time)
        return BackendMetrics(
            sim_time=self.sim_time,
            busy_time=self.busy_time,
            gpu_busy_frac=busy_frac,
            hbm_used_bytes=self.hbm_used_bytes,
            hbm_capacity_bytes=self.hbm_capacity_bytes,
            pending_requests=len(self.resident_kv),
            queue_depth=len(self.queue),
        )