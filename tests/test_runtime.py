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
