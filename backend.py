"""
backend.py — the GPUBackend abstraction.

Hyperion's control-plane code (scheduler, agentic controller, KV fabric,
economic brain) should NEVER import anything GPU-specific directly.
It only ever talks to a GPUBackend. Today that's SimulatedBackend
(backend.py's sibling simulated_backend.py). Later it can be a
RealBackend wrapping vLLM/SGLang on actual hardware, with zero changes
to the control logic above it.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class GPUProfile:
    """Public datasheet numbers — no measurement required to fill these in."""
    name: str
    peak_flops_fp16: float      # FLOPs/sec, dense tensor-core peak
    achieved_mfu: float         # fraction of peak you actually get (calibrate later)
    hbm_capacity_bytes: int
    hbm_bandwidth_bytes_per_sec: float
    intra_node_bw_bytes_per_sec: float   # NVLink/xGMI-class
    inter_node_bw_bytes_per_sec: float   # network fabric between nodes/regions


# Rough public specs — swap in calibrated numbers once you have real traces.
GPU_PROFILES = {
    "A100_80G": GPUProfile(
        name="A100_80G",
        peak_flops_fp16=312e12,
        achieved_mfu=0.35,
        hbm_capacity_bytes=80 * (1024**3),
        hbm_bandwidth_bytes_per_sec=2.0e12,
        intra_node_bw_bytes_per_sec=600e9,
        inter_node_bw_bytes_per_sec=100e9,
    ),
    "H100_80G": GPUProfile(
        name="H100_80G",
        peak_flops_fp16=989e12,
        achieved_mfu=0.40,
        hbm_capacity_bytes=80 * (1024**3),
        hbm_bandwidth_bytes_per_sec=3.35e12,
        intra_node_bw_bytes_per_sec=900e9,
        inter_node_bw_bytes_per_sec=100e9,
    ),
    "MI300X_192G": GPUProfile(
        name="MI300X_192G",
        peak_flops_fp16=1300e12,
        achieved_mfu=0.30,
        hbm_capacity_bytes=192 * (1024**3),
        hbm_bandwidth_bytes_per_sec=5.3e12,
        intra_node_bw_bytes_per_sec=896e9,
        inter_node_bw_bytes_per_sec=100e9,
    ),
}


@dataclass
class BatchSpec:
    """Describes a batch of work submitted to the backend — not real tensors,
    just the shape info needed to estimate time/memory."""
    request_ids: list
    stage: str              # "prefill" or "decode"
    num_tokens: int          # total tokens being processed this step
    model_params: float      # model parameter count (for FLOPs estimate)
    attn_variant: str = "GQA"  # "MHA" | "GQA" | "MLA" — affects KV bytes/token
    kv_bytes_per_token: float = 2 * 128 * 2  # bytes; override per attn_variant/model


@dataclass
class BackendMetrics:
    sim_time: float        # total elapsed clock time (busy + idle)
    busy_time: float       # time actually spent executing batches
    gpu_busy_frac: float   # busy_time / sim_time — meaningful now that idle is tracked
    hbm_used_bytes: int
    hbm_capacity_bytes: int
    pending_requests: int
    queue_depth: int


class GPUBackend(ABC):
    @abstractmethod
    def estimate_latency(self, batch: BatchSpec) -> float:
        """Seconds this batch would take to execute."""

    @abstractmethod
    def submit_batch(self, batch: BatchSpec) -> float:
        """Advance internal clock/state as if this batch ran. Returns latency."""

    @abstractmethod
    def memory_pressure(self) -> float:
        """0.0 (empty) .. 1.0 (full) fraction of HBM used for KV cache."""

    @abstractmethod
    def transfer_kv_block(self, num_bytes: int, cross_region: bool) -> float:
        """Seconds to move a KV block (prefix teleport). Returns transfer latency."""

    @abstractmethod
    def metrics(self) -> BackendMetrics:
        """Snapshot of current backend state for the controller to read."""

    @abstractmethod
    def advance_idle(self, dt: float) -> None:
        """Advance the clock by dt with no work happening (e.g. waiting for
        the next arrival). Needed so gpu_busy_frac means something instead
        of trivially being 1.0 whenever batches run back-to-back."""

    @abstractmethod
    def release(self, request_id: str) -> None:
        """Free a finished request's resident KV cache."""