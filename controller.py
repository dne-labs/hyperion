"""
controller.py — first slice of Hyperion's "agentic" controller.

Starts as plain threshold rules, not an LLM. The point right now is the
*loop shape*: read metrics -> decide -> reconfigure -> log why. Once this
shape is solid, swap the decide() body for an actual small-model call
that reasons over the same metrics and emits the same action objects.
"""

from dataclasses import dataclass, field
from backend import GPUBackend


@dataclass
class ServingConfig:
    tensor_parallel_degree: int = 1
    speculative_decoding: bool = False
    chunked_prefill: bool = False
    attn_variant: str = "GQA"
    log: list = field(default_factory=list)

    def note(self, msg: str):
        self.log.append(msg)


class AgenticController:
    def __init__(self, backend: GPUBackend, config: ServingConfig):
        self.backend = backend
        self.config = config

    def tick(self, queue_depth: int, avg_context_len: float):
        """Call once per scheduling step. Mutates self.config in place and
        records a human-readable reason for every change, which is exactly
        what you want for debugging an 'autonomous' system later."""
        m = self.backend.metrics()

        # Rule 1: memory pressure -> eviction is handled by the backend,
        # but we also react by getting more conservative up top.
        if m.hbm_used_bytes / max(1, m.hbm_capacity_bytes) > 0.85:
            if not self.config.speculative_decoding:
                pass  # spec decoding costs extra KV memory; don't enable under pressure
            if self.config.chunked_prefill is False:
                self.config.chunked_prefill = True
                self.config.note(
                    f"[t={m.sim_time:.2f}] HBM at "
                    f"{m.hbm_used_bytes / max(1, m.hbm_capacity_bytes):.0%} -> enabling chunked prefill"
                )

        # Rule 2: long-context-heavy traffic -> favor prefill capacity / chunking
        if avg_context_len > 4000 and not self.config.chunked_prefill:
            self.config.chunked_prefill = True
            self.config.note(
                f"[t={m.sim_time:.2f}] avg context {avg_context_len:.0f} tokens -> "
                f"enabling chunked prefill for long-context traffic"
            )

        # Rule 3: queue backing up but memory has headroom -> enable speculative decoding
        if queue_depth > 8 and m.hbm_used_bytes / max(1, m.hbm_capacity_bytes) < 0.6:
            if not self.config.speculative_decoding:
                self.config.speculative_decoding = True
                self.config.note(
                    f"[t={m.sim_time:.2f}] queue_depth={queue_depth}, memory headroom available -> "
                    f"enabling speculative decoding to cut decode latency"
                )
        elif queue_depth < 2 and self.config.speculative_decoding:
            self.config.speculative_decoding = False
            self.config.note(
                f"[t={m.sim_time:.2f}] queue_depth={queue_depth} -> "
                f"disabling speculative decoding (draft overhead not worth it)"
            )

        # Rule 4: sustained high GPU busy fraction -> request more TP degree
        if m.gpu_busy_frac > 0.9 and self.config.tensor_parallel_degree < 4:
            self.config.tensor_parallel_degree *= 2
            self.config.note(
                f"[t={m.sim_time:.2f}] gpu_busy_frac={m.gpu_busy_frac:.2f} -> "
                f"doubling tensor parallel degree to {self.config.tensor_parallel_degree}"
            )