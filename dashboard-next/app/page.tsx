"use client";

import { useCallback, useEffect, useState } from "react";
import {
  Activity,
  ArrowRight,
  BrainCircuit,
  Database,
  Gauge,
  Play,
  Server,
  Zap,
} from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader } from "@/components/ui/card";

type Topology = {
  mode: string;
  state: {
    prefill_replicas: number;
    decode_replicas: number;
    prefill_batch_size: number;
    decode_batch_size: number;
    aggressive_prefix_cache: boolean;
    speculative_decoding: boolean;
  };
  cache_entries: number;
  queues: {
    prefill: number;
    decode: number;
  };
  last_controller_decision: string;
};

type VidurProfile = {
  model: string;
  gpu_profile: string;
  replicas: number;
  qps: number;
  request_count: number;
  prompt_tokens: number;
  output_tokens: number;
  latency: {
    ttft_p95_ms: number;
    e2e_p95_seconds: number;
    tpot_p95_ms: number;
    scheduling_delay_p95_ms: number;
  };
};

type ProfileComparison = {
  baseline_run: string;
  candidate_run: string;
  same_workload: boolean;
  baseline: VidurProfile;
  candidate: VidurProfile;
};

const gateway =
  process.env.NEXT_PUBLIC_GATEWAY_URL || "http://localhost:8000";

function Metric({
  icon: Icon,
  label,
  value,
  help,
}: {
  icon: typeof Server;
  label: string;
  value: string | number;
  help: string;
}) {
  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between">
        <p className="text-sm text-slate-400">{label}</p>
        <Icon className="h-4 w-4 text-indigo-300" />
      </CardHeader>

      <CardContent>
        <p className="text-3xl font-bold">{value}</p>
        <p className="mt-2 text-xs leading-5 text-slate-500">{help}</p>
      </CardContent>
    </Card>
  );
}

export default function Home() {
  const [topology, setTopology] = useState<Topology | null>(null);
  const [profile, setProfile] = useState<VidurProfile | null>(null);
  const [comparison, setComparison] = useState<ProfileComparison | null>(null);
  const [status, setStatus] = useState("Waiting for the gateway…");
  const [loading, setLoading] = useState(false);

  const refresh = useCallback(async () => {
    try {
      // Fetch live topology and the saved Vidur profile at the same time.
      const [topologyResponse, profileResponse, comparisonResponse] =
        await Promise.all([
          fetch(`${gateway}/topology`),
          fetch(`${gateway}/vidur-profile`),
          fetch(`${gateway}/vidur-comparison`),
        ]);

      if (
        !topologyResponse.ok ||
        !profileResponse.ok ||
        !comparisonResponse.ok
      ) {
        throw new Error("A Gateway request failed");
      }

      const topologyData: Topology = await topologyResponse.json();
      const profileData: VidurProfile = await profileResponse.json();
      const comparisonData: ProfileComparison =
        await comparisonResponse.json();

      setTopology(topologyData);
      setProfile(profileData);
      setComparison(comparisonData);
      setStatus("Connected — updates every 2 seconds");
    } catch {
      setStatus("Gateway or Vidur profile is unavailable");
    }
  }, []);

  useEffect(() => {
    void refresh();

    const timer = window.setInterval(() => {
      void refresh();
    }, 2000);

    return () => window.clearInterval(timer);
  }, [refresh]);

  async function runDemo() {
    setLoading(true);
    setStatus(
      "Sending 12 shared-prefix requests. The controller is watching pressure…",
    );

    try {
      const response = await fetch(`${gateway}/demo/load`, {
        method: "POST",
        headers: {
          "content-type": "application/json",
        },
        body: JSON.stringify({
          requests: 12,
          max_tokens: 250,
          shared_prefix:
            "You are Hyperion, an inference serving operating system.",
        }),
      });

      if (!response.ok) {
        throw new Error("The load demo request failed");
      }

      await refresh();
      setStatus("Burst complete. Hyperion recorded the result below.");
    } catch {
      setStatus("Could not reach the Gateway. Is Docker running?");
    } finally {
      setLoading(false);
    }
  }

  const state = topology?.state;
  const isConnected = status.startsWith("Connected");

  const comparisonRows = comparison
    ? [
        {
          label: "TTFT p95",
          baseline: comparison.baseline.latency.ttft_p95_ms,
          candidate: comparison.candidate.latency.ttft_p95_ms,
          unit: "ms",
        },
        {
          label: "E2E p95",
          baseline: comparison.baseline.latency.e2e_p95_seconds,
          candidate: comparison.candidate.latency.e2e_p95_seconds,
          unit: "sec",
        },
        {
          label: "TPOT p95",
          baseline: comparison.baseline.latency.tpot_p95_ms,
          candidate: comparison.candidate.latency.tpot_p95_ms,
          unit: "ms/token",
        },
        {
          label: "Scheduling delay p95",
          baseline: comparison.baseline.latency.scheduling_delay_p95_ms,
          candidate: comparison.candidate.latency.scheduling_delay_p95_ms,
          unit: "ms",
        },
      ]
    : [];

  return (
    <main className="min-h-screen bg-[radial-gradient(circle_at_top,#172554_0%,#020617_42%)] px-5 py-8 font-sans md:px-10">
      <div className="mx-auto max-w-7xl">
        <header className="mb-8 flex flex-col justify-between gap-4 border-b border-slate-800 pb-6 md:flex-row md:items-center">
          <div>
            <div className="mb-2 flex items-center gap-2">
              <Zap className="h-5 w-5 text-indigo-300" />
              <Badge>No physical GPU required</Badge>
            </div>

            <h1 className="text-3xl font-bold tracking-tight text-white">
              Hyperion Control Room
            </h1>

            <p className="mt-2 text-slate-400">
              A simple view of what your inference system is doing.
            </p>
          </div>

          <div className="text-sm text-slate-400">
            <span
              className={`mr-2 inline-block h-2 w-2 rounded-full ${
                isConnected ? "bg-emerald-400" : "bg-amber-400"
              }`}
            />
            {status}
          </div>
        </header>

        {/* Live simulated Hyperion topology */}
        <section className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
          <Metric
            icon={Server}
            label="Prefill workers"
            value={state?.prefill_replicas ?? "—"}
            help="Workers that read and process incoming prompts."
          />

          <Metric
            icon={Zap}
            label="Decode workers"
            value={state?.decode_replicas ?? "—"}
            help="Workers that generate answers token by token."
          />

          <Metric
            icon={Database}
            label="Saved prefixes"
            value={topology?.cache_entries ?? "—"}
            help="Reusable prompt context stored in the simulated cache."
          />

          <Metric
            icon={Gauge}
            label="Waiting requests"
            value={
              topology
                ? topology.queues.prefill + topology.queues.decode
                : "—"
            }
            help="Requests currently waiting in the simulated worker pools."
          />
        </section>

        {/* Results from the saved Vidur experiment */}
        <section className="mt-8">
          <h2 className="mb-2 text-lg font-semibold text-white">
            Vidur performance profile
          </h2>

          <p className="mb-4 text-sm text-slate-400">
            {profile
              ? `${profile.model} · ${profile.gpu_profile.toUpperCase()} profile · ${profile.qps} requests/sec · ${profile.request_count} requests`
              : "Loading Vidur results…"}
          </p>

          <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
            <Metric
              icon={Gauge}
              label="TTFT p95"
              value={
                profile
                  ? `${profile.latency.ttft_p95_ms.toFixed(0)} ms`
                  : "—"
              }
              help="95% of simulated requests got their first token within this time."
            />

            <Metric
              icon={Zap}
              label="TPOT p95"
              value={
                profile
                  ? `${profile.latency.tpot_p95_ms.toFixed(1)} ms`
                  : "—"
              }
              help="High-end time per generated token in this simulation."
            />

            <Metric
              icon={Activity}
              label="E2E p95"
              value={
                profile
                  ? `${profile.latency.e2e_p95_seconds.toFixed(1)} sec`
                  : "—"
              }
              help="95% of complete simulated requests finished within this time."
            />

            <Metric
              icon={Gauge}
              label="Queue delay p95"
              value={
                profile
                  ? `${profile.latency.scheduling_delay_p95_ms.toFixed(0)} ms`
                  : "—"
              }
              help="High-end time requests waited for the scheduler."
            />
          </div>

          {profile && (
            <p className="mt-3 text-xs text-slate-500">
              Workload: {profile.prompt_tokens} prompt tokens and{" "}
              {profile.output_tokens} output tokens per request. These are
              simulated hardware-profile results.
            </p>
          )}
        </section>

        <section className="mt-8">
          <h2 className="mb-2 text-lg font-semibold text-white">
            Replica comparison
          </h2>

          {comparison?.same_workload ? (
            <>
              <p className="mb-4 text-sm text-slate-400">
                {comparison.baseline.replicas} replica
                {comparison.baseline.replicas === 1 ? "" : "s"} vs{" "}
                {comparison.candidate.replicas} replicas. Same workload
                settings.
              </p>

              <Card className="overflow-x-auto">
                <table className="w-full text-left text-sm">
                  <thead className="border-b border-slate-800 text-slate-400">
                    <tr>
                      <th className="p-4">Metric</th>
                      <th className="p-4">Baseline</th>
                      <th className="p-4">Candidate</th>
                      <th className="p-4">Change</th>
                    </tr>
                  </thead>
                  <tbody>
                    {comparisonRows.map((row) => {
                      const change =
                        row.baseline === 0
                          ? null
                          : ((row.baseline - row.candidate) / row.baseline) *
                            100;
                      const changeLabel =
                        change === null
                          ? "—"
                          : change >= 0
                            ? `${change.toFixed(1)}% lower`
                            : `${Math.abs(change).toFixed(1)}% higher`;

                      return (
                        <tr
                          key={row.label}
                          className="border-b border-slate-800"
                        >
                          <td className="p-4 font-medium text-white">
                            {row.label}
                          </td>
                          <td className="p-4 text-slate-300">
                            {row.baseline.toFixed(2)} {row.unit}
                          </td>
                          <td className="p-4 text-slate-300">
                            {row.candidate.toFixed(2)} {row.unit}
                          </td>
                          <td
                            className={`p-4 ${
                              change === null
                                ? "text-slate-400"
                                : change >= 0
                                  ? "text-emerald-300"
                                  : "text-rose-300"
                            }`}
                          >
                            {changeLabel}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </Card>
              <p className="mt-3 text-xs text-slate-500">
                Baseline run: {comparison.baseline_run} · Candidate run:{" "}
                {comparison.candidate_run}
              </p>
            </>
          ) : (
            <p className="text-sm text-amber-300">
              {comparison
                ? "The workloads differ, so these results are not a fair replica comparison."
                : "Loading the comparison…"}
            </p>
          )}
        </section>

        <section className="mt-6 grid gap-6 lg:grid-cols-[1.4fr_1fr]">
          <Card>
            <CardHeader>
              <div className="flex items-center gap-2">
                <BrainCircuit className="h-5 w-5 text-violet-300" />
                <h2 className="font-semibold text-white">
                  What Hyperion decided
                </h2>
              </div>
            </CardHeader>

            <CardContent>
              <p className="text-2xl font-bold capitalize text-indigo-200">
                {topology?.last_controller_decision?.replaceAll("_", " ") ??
                  "Waiting for data"}
              </p>

              <p className="mt-3 leading-7 text-slate-400">
                {topology?.last_controller_decision === "scale_decode"
                  ? "Many requests were generating answers together, so Hyperion added a Decode worker."
                  : "Hyperion is watching request pressure, cache reuse, and worker capacity. With light traffic, one worker per stage is enough."}
              </p>

              <div className="mt-5 flex flex-wrap gap-2">
                <Badge
                  className={
                    state?.aggressive_prefix_cache
                      ? ""
                      : "border-slate-700 bg-slate-800 text-slate-300"
                  }
                >
                  Prefix cache:{" "}
                  {state?.aggressive_prefix_cache ? "aggressive" : "normal"}
                </Badge>

                <Badge
                  className={
                    state?.speculative_decoding
                      ? ""
                      : "border-slate-700 bg-slate-800 text-slate-300"
                  }
                >
                  Speculative decoding:{" "}
                  {state?.speculative_decoding ? "on" : "off"}
                </Badge>
              </div>
            </CardContent>
          </Card>

          <Card className="border-indigo-500/30 bg-indigo-500/5">
            <CardHeader>
              <h2 className="font-semibold text-white">
                Try the scaling demo
              </h2>
              <p className="mt-1 text-sm text-slate-400">
                Send fake traffic through the simulator. No model or GPU is
                required.
              </p>
            </CardHeader>

            <CardContent>
              <Button
                disabled={loading}
                onClick={runDemo}
                className="w-full gap-2"
              >
                <Play className="h-4 w-4" />
                {loading ? "Running traffic burst…" : "Send 12 fake requests"}
              </Button>

              <p className="mt-4 text-xs leading-5 text-slate-500">
                The controller watches the burst and may increase the simulated
                Decode worker count.
              </p>
            </CardContent>
          </Card>
        </section>

        <section className="mt-6">
          <h2 className="mb-3 text-lg font-semibold text-white">
            How one request travels through Hyperion
          </h2>

          <div className="grid gap-3 md:grid-cols-4">
            {[
              ["1", "Gateway", "Receives a chat request."],
              ["2", "Prefill", "Reads and processes the prompt."],
              ["3", "KV cache", "Can reuse shared prompt context."],
              ["4", "Decode", "Generates the answer token by token."],
            ].map(([number, title, description], index) => (
              <Card key={title}>
                <CardContent className="pt-5">
                  <p className="text-sm text-indigo-300">Step {number}</p>
                  <h3 className="mt-1 font-semibold text-white">{title}</h3>
                  <p className="mt-2 text-sm text-slate-400">{description}</p>

                  {index < 3 && (
                    <ArrowRight className="mt-3 hidden h-4 w-4 text-slate-600 md:block" />
                  )}
                </CardContent>
              </Card>
            ))}
          </div>
        </section>

        <section className="mt-6 grid gap-4 md:grid-cols-3">
          <Card>
            <CardContent className="pt-5">
              <Activity className="h-5 w-5 text-emerald-300" />
              <h3 className="mt-3 font-semibold">What you can try</h3>
              <p className="mt-2 text-sm leading-6 text-slate-400">
                Click “Send 12 fake requests” and watch the topology and
                controller decision.
              </p>
            </CardContent>
          </Card>

          <Card>
            <CardContent className="pt-5">
              <Database className="h-5 w-5 text-amber-300" />
              <h3 className="mt-3 font-semibold">Why the cache matters</h3>
              <p className="mt-2 text-sm leading-6 text-slate-400">
                Repeated system prompts can reuse cached context instead of
                processing that shared part from scratch.
              </p>
            </CardContent>
          </Card>

          <Card>
            <CardContent className="pt-5">
              <BrainCircuit className="h-5 w-5 text-violet-300" />
              <h3 className="mt-3 font-semibold">What Vidur adds</h3>
              <p className="mt-2 text-sm leading-6 text-slate-400">
                Vidur estimates performance using GPU profiles. The results
                shown above come from your saved experiment.
              </p>
            </CardContent>
          </Card>
        </section>
      </div>
    </main>
  );
}