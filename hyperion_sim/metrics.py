from __future__ import annotations

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from fastapi.responses import Response

REQUESTS = Counter("hyperion_requests_total", "Requests processed")
CACHE_HITS = Counter("hyperion_prefix_cache_hits_total", "Prefix cache hits")
CACHE_MISSES = Counter("hyperion_prefix_cache_misses_total", "Prefix cache misses")
TTFT = Histogram("hyperion_ttft_seconds", "Time to first token", buckets=(.001, .01, .05, .1, .25, .5, 1, 2))
TPOT = Histogram("hyperion_tpot_seconds", "Time per output token", buckets=(.001, .005, .01, .02, .05, .1, .2))
E2E = Histogram("hyperion_e2e_seconds", "End-to-end latency")
PREFILL_QUEUE = Gauge("hyperion_prefill_queue_depth", "Prefill queue depth")
DECODE_QUEUE = Gauge("hyperion_decode_queue_depth", "Decode queue depth")


def record_request(outcome, total: float) -> None:
    REQUESTS.inc()
    (CACHE_HITS if outcome.cache_hit else CACHE_MISSES).inc()
    TTFT.observe(outcome.prefill_seconds)
    TPOT.observe(outcome.decode_seconds / max(1, outcome.output_tokens))
    E2E.observe(total)
    PREFILL_QUEUE.set(outcome.prefill_queue_depth)
    DECODE_QUEUE.set(outcome.decode_queue_depth)


def metrics_response() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
