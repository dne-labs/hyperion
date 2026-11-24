import pytest
from hyperion_sim.runtime import HyperionRuntime


@pytest.mark.asyncio
async def test_shared_prompt_hits_prefix_cache():
    runtime = HyperionRuntime()
    first = await runtime.execute("system: shared instructions\nuser: hello", 2)
    second = await runtime.execute("system: shared instructions\nuser: hello", 2)
    assert not first.cache_hit
    assert second.cache_hit
    assert second.prefill_seconds < first.prefill_seconds


@pytest.mark.asyncio
async def test_more_replicas_reduce_decode_time():
    runtime = HyperionRuntime()
    one_replica = await runtime.execute("user: hello", 100)
    runtime.state.decode_replicas = 2
    two_replicas = await runtime.execute("user: another request", 100)
    assert two_replicas.decode_seconds < one_replica.decode_seconds
