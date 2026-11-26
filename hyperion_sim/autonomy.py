"""Autonomous episodic Vidur control with persistent evidence and rollback."""
import asyncio
import json
import math
import time
from pathlib import Path
from pydantic import BaseModel, Field
from fastapi import HTTPException

class AutonomyRequest(BaseModel):
    traffic: list[float] = Field(default_factory=lambda: [4, 8, 4], min_length=1, max_length=12)
    cycles: int = Field(default=1, ge=1, le=5)
    ttft_ms: float = Field(default=600, gt=0)
    e2e_seconds: float = Field(default=10, gt=0)
    requests: int = Field(default=32, ge=8, le=64)
    interval_seconds: float = Field(default=5, ge=1, le=300)

class Autopilot:
    def __init__(self, root, lock, evaluator, request_type):
        self.root, self.lock = root, lock
        self.evaluate, self.request_type = evaluator, request_type
        self.task = None
        self.stop_requested = False
        self.state = {"status": "idle", "active_replicas": 1, "events": []}
        self.memory = {}
        path = root / "autonomy_memory.json"
        if path.exists():
            self.memory = json.loads(path.read_text())
        self.memory = self.memory if isinstance(self.memory, dict) else {}

    def event(self, kind, **details):
        self.state["events"].append({"time": time.time(), "kind": kind, **details})
        self.state["events"] = self.state["events"][-200:]
        self.save()

    def save(self):
        self.root.mkdir(parents=True, exist_ok=True)
        for name, data in [("autonomy_state.json", self.state), ("autonomy_memory.json", self.memory)]:
            path = self.root / name
            temp = path.with_suffix(".tmp")
            temp.write_text(json.dumps(data, indent=2, allow_nan=False))
            temp.replace(path)

    def score(self, p, targets):
        latency = p["latency"]
        a, b = float(latency["ttft_p95_ms"]), float(latency["e2e_p95_seconds"])
        if not all(math.isfinite(v) and v >= 0 for v in (a,b)):
            raise ValueError("Invalid latency evidence")
        return (max(0, a/targets.ttft_ms-1) + max(0,b/targets.e2e_seconds-1), p["replicas"])

    async def measure(self, qps, replicas, request):
        args = self.request_type(qps=qps, requests=request.requests)
        result = await asyncio.to_thread(self.evaluate, args, replicas, args.chunk_size)
        profile = json.loads((self.root / result["run"] / "hyperion_profile.json").read_text())
        self.score(profile, request)
        key = json.dumps([qps, request.requests, replicas])
        self.memory[key] = {"profile": profile, "run": result["run"], "time": time.time()}
        self.event("observed", qps=qps, replicas=replicas, run=result["run"], latency=profile["latency"])
        return profile

    async def run(self, request):
        try:
            async with self.lock:
                for epoch, qps in enumerate(request.traffic * request.cycles):
                    if self.stop_requested: break
                    self.state.update(epoch=epoch+1, qps=qps, phase="observe")
                    old = self.state["active_replicas"]
                    baseline = await self.measure(qps, old, request)
                    if self.stop_requested: break
                    choices = [n for n in (1,2,4) if n != old]
                    def rank(n):
                        evidence = self.memory.get(json.dumps([qps, request.requests, n]))
                        return self.score(evidence["profile"], request) if evidence else (-1, n)
                    choices.sort(key=rank)
                    choices = [n for n in choices if rank(n) < self.score(baseline, request)]
                    best = baseline
                    self.event("baseline", replicas=old, latency=baseline["latency"])
                    for candidate in choices:
                        if self.stop_requested: break
                        self.state["phase"] = "evaluate"
                        self.event("proposed", replicas=candidate, reason="Explore unobserved capacity or reuse workload-specific evidence")
                        measured = await self.measure(qps, candidate, request)
                        if self.score(measured, request) < self.score(best, request):
                            best = measured
                    if self.stop_requested: break
                    chosen = best["replicas"]
                    if chosen != old:
                        self.state.update(active_replicas=chosen, phase="validate", rollback_replicas=old)
                        self.event("applied", previous=old, replicas=chosen, scope="next independent Vidur episode")
                        validation = await self.measure(qps, chosen, request)
                        if self.score(validation, request) >= self.score(baseline, request):
                            self.state["active_replicas"] = old
                            self.event("rollback", replicas=old, reason="Validation did not improve SLO violation/resource objective")
                        else:
                            self.event("accepted", replicas=chosen, before=baseline["latency"], after=validation["latency"])
                    else:
                        self.event("held", replicas=old, reason="No tested candidate improves the objective")
                    self.state.pop("rollback_replicas", None)
                    self.state["phase"] = "cooldown"
                    self.save()
                    for _ in range(math.ceil(request.interval_seconds)):
                        if self.stop_requested: break
                        await asyncio.sleep(1)
                self.state["status"] = "stopped" if self.stop_requested else "completed"
        except Exception as error:
            if "rollback_replicas" in self.state:
                self.state["active_replicas"] = self.state.pop("rollback_replicas")
                self.event("rollback", replicas=self.state["active_replicas"], reason="Validation failed")
            self.state.update(status="failed", error=str(error))
            self.event("error", message=str(error))
        finally:
            self.save()

    def start(self, request):
        if self.task and not self.task.done() or self.lock.locked():
            raise HTTPException(409, "A simulation job is already running.")
        if any(not math.isfinite(q) or q <= 0 or q > 64 for q in request.traffic):
            raise HTTPException(422, "Traffic must contain finite QPS values in (0, 64].")
        self.stop_requested = False
        self.state = {"status":"running", "phase":"starting", "active_replicas":1,
                      "events":[], "targets":request.model_dump(),
                      "mode":"episodic Vidur simulation; no live GPU actuation"}
        self.save()
        self.task = asyncio.create_task(self.run(request))
        return self.state

def install_autonomy(app, root, lock, evaluator, request_type):
    pilot = Autopilot(root, lock, evaluator, request_type)
    @app.get("/autonomy")
    async def status():
        return pilot.state
    @app.post("/autonomy/start", status_code=202)
    async def start(request: AutonomyRequest):
        return pilot.start(request)
    @app.post("/autonomy/stop")
    async def stop():
        pilot.stop_requested = True
        pilot.event("stop_requested", note="Stops after the current Vidur subprocess finishes")
        return pilot.state
    return pilot
