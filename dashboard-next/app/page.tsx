"use client";

import { useCallback, useEffect, useState, type FormEvent } from "react";
import {
  Activity,
  ArrowRight,
  BrainCircuit,
  Cpu,
  Database,
  Gauge,
  Layers3,
  Network,
  Play,
  RefreshCw,
  Server,
  ShieldCheck,
  SlidersHorizontal,
  Sparkles,
  Timer,
  Zap,
} from "lucide-react";

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
  queues: { prefill: number; decode: number };
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

type VidurRecommendation = {
  workload: {
    model: string;
    gpu_profile: string;
    qps: number;
    request_count: number;
    prompt_tokens: number;
    output_tokens: number;
  };
  targets: {
    ttft_p95_ms: number;
    e2e_p95_seconds: number;
    scheduling_delay_p95_ms: number;
  };
  tested_profiles: Array<{
    run: string;
    replicas: number;
    meets_targets: boolean;
    latency: {
      ttft_p95_ms: number;
      e2e_p95_seconds: number;
      scheduling_delay_p95_ms: number;
    };
    failed_targets: string[];
  }>;
  recommendation: {
    run: string;
    replicas: number;
    latency: {
      ttft_p95_ms: number;
      e2e_p95_seconds: number;
      scheduling_delay_p95_ms: number;
    };
  } | null;
  message: string;
  hardware_note: string;
};

const gateway =
  process.env.NEXT_PUBLIC_GATEWAY_URL || "http://localhost:8000";

const navItems = [
  { label: "Overview", href: "#overview", icon: Activity },
  { label: "Try a request", href: "#try-request", icon: Play },
  { label: "Capacity planner", href: "#capacity", icon: Gauge },
  { label: "Worker pools", href: "#workers", icon: Server },
  { label: "Controller", href: "#controller", icon: BrainCircuit },
  { label: "Request path", href: "#request-path", icon: Network },
];

function StatCard({
  label,
  value,
  detail,
  icon: Icon,
  accent,
}: {
  label: string;
  value: string | number;
  detail: string;
  icon: typeof Server;
  accent: string;
}) {
  return (
    <div className="group rounded-2xl border border-white/[0.07] bg-[#0e131c] p-5 transition duration-200 hover:border-white/[0.13] hover:bg-[#111823]">
      <div className="flex items-start justify-between">
        <div>
          <p className="text-[11px] font-semibold uppercase tracking-[0.16em] text-slate-500">
            {label}
          </p>
          <p className="mt-4 text-3xl font-semibold tracking-tight text-slate-100">
            {value}
          </p>
        </div>
        <div className={"rounded-xl border border-white/[0.06] p-2.5 " + accent}>
          <Icon className="h-4 w-4" />
        </div>
      </div>
      <p className="mt-3 text-xs text-slate-500">{detail}</p>
    </div>
  );
}

function SectionHeading({
  eyebrow,
  title,
  detail,
}: {
  eyebrow: string;
  title: string;
  detail: string;
}) {
  return (
    <div className="mb-5">
      <p className="text-[10px] font-bold uppercase tracking-[0.2em] text-cyan-400/80">
        {eyebrow}
      </p>
      <h2 className="mt-2 text-xl font-semibold tracking-tight text-white">
        {title}
      </h2>
      <p className="mt-1.5 text-sm leading-6 text-slate-500">{detail}</p>
    </div>
  );
}

export default function Home() {
  const [topology, setTopology] = useState<Topology | null>(null);
  const [profile, setProfile] = useState<VidurProfile | null>(null);
  const [comparison, setComparison] = useState<ProfileComparison | null>(null);
  const [recommendation, setRecommendation] =
    useState<VidurRecommendation | null>(null);
  const [targets, setTargets] = useState({
    ttft_p95_ms: 600,
    e2e_p95_seconds: 10,
    scheduling_delay_p95_ms: 400,
  });
  const [targetDraft, setTargetDraft] = useState({
    ttft_p95_ms: "600",
    e2e_p95_seconds: "10",
    scheduling_delay_p95_ms: "400",
  });
  const [status, setStatus] = useState("Connecting to Gateway");
  const [loading, setLoading] = useState(false);
  const [chatPrompt, setChatPrompt] = useState("Explain why a model uses a KV cache in two short sentences.");
  const [chatAnswer, setChatAnswer] = useState("");
  const [chatMode, setChatMode] = useState("");
  const [chatBusy, setChatBusy] = useState(false);

  const refresh = useCallback(async () => {
    try {
      const topologyResponse = await fetch(gateway + "/topology");
      if (!topologyResponse.ok) {
        throw new Error("Gateway topology request failed");
      }

      const targetQuery = new URLSearchParams({
        "max-ttft-p95-ms": String(targets.ttft_p95_ms),
        "max-e2e-p95-seconds": String(targets.e2e_p95_seconds),
        "max-scheduling-delay-p95-ms": String(
          targets.scheduling_delay_p95_ms,
        ),
      });

      const [profileData, comparisonData, recommendationData] =
        await Promise.all(
          [
            gateway + "/vidur-profile",
            gateway + "/vidur-comparison",
            gateway + "/vidur-recommendation?" + targetQuery.toString(),
          ].map(async (url) => {
            try {
              const response = await fetch(url);
              return response.ok ? await response.json() : null;
            } catch {
              return null;
            }
          }),
        );

      setTopology(await topologyResponse.json());
      setProfile(profileData);
      setComparison(comparisonData);
      setRecommendation(recommendationData);
      setStatus("All systems operational");
    } catch {
      setStatus("Gateway unavailable");
    }
  }, [targets]);

  useEffect(() => {
    void refresh();
    const timer = window.setInterval(() => void refresh(), 3000);
    return () => window.clearInterval(timer);
  }, [refresh]);

  async function runDemo() {
    setLoading(true);
    try {
      const response = await fetch(gateway + "/demo/load", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          requests: 12,
          max_tokens: 250,
          shared_prefix:
            "You are Hyperion, an inference serving operating system.",
        }),
      });
      if (!response.ok) throw new Error("Traffic demo failed");
      await refresh();
      setStatus("Traffic burst complete");
    } catch {
      setStatus("Could not send traffic — check the Gateway");
    } finally {
      setLoading(false);
    }
  }

  async function sendChat(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setChatBusy(true);
    setChatAnswer("");
    try {
      const response = await fetch(gateway + "/v1/chat/completions", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          model: "simulated-llama-3-8b",
          messages: [{ role: "user", content: chatPrompt }],
          max_tokens: 96,
          temperature: 0.7,
        }),
      });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(typeof data.detail === "string" ? data.detail : "Request failed");
      }
      setChatAnswer(data.choices?.[0]?.message?.content ?? "No answer returned.");
      setChatMode(data.hyperion?.mode === "vllm_cpu" ? "Real vLLM CPU inference" : "Simulator placeholder");
    } catch (error) {
      setChatAnswer(error instanceof Error ? error.message : "Could not reach Gateway.");
      setChatMode("Request failed");
    } finally {
      setChatBusy(false);
    }
  }

  function applyTargets(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setTargets({
      ttft_p95_ms: Number(targetDraft.ttft_p95_ms),
      e2e_p95_seconds: Number(targetDraft.e2e_p95_seconds),
      scheduling_delay_p95_ms: Number(
        targetDraft.scheduling_delay_p95_ms,
      ),
    });
  }

  const state = topology?.state;
  const profiles = recommendation?.tested_profiles ?? [];
  const maxTtft = Math.max(
    1,
    ...profiles.map((item) => item.latency.ttft_p95_ms),
    recommendation?.targets.ttft_p95_ms ?? 0,
  );
  const comparisonRows = comparison
    ? [
        {
          label: "TTFT p95",
          baseline: comparison.baseline.latency.ttft_p95_ms,
          candidate: comparison.candidate.latency.ttft_p95_ms,
          unit: "ms",
        },
        {
          label: "End-to-end p95",
          baseline: comparison.baseline.latency.e2e_p95_seconds,
          candidate: comparison.candidate.latency.e2e_p95_seconds,
          unit: "sec",
        },
        {
          label: "Time per output token",
          baseline: comparison.baseline.latency.tpot_p95_ms,
          candidate: comparison.candidate.latency.tpot_p95_ms,
          unit: "ms",
        },
        {
          label: "Queue delay p95",
          baseline: comparison.baseline.latency.scheduling_delay_p95_ms,
          candidate:
            comparison.candidate.latency.scheduling_delay_p95_ms,
          unit: "ms",
        },
      ]
    : [];

  return (
    <main className="min-h-screen bg-[#080b10] text-slate-100">
      <div className="pointer-events-none fixed inset-0 overflow-hidden">
        <div className="absolute -left-40 -top-48 h-[520px] w-[520px] rounded-full bg-cyan-500/[0.06] blur-[130px]" />
        <div className="absolute right-0 top-0 h-[460px] w-[460px] rounded-full bg-indigo-500/[0.05] blur-[140px]" />
      </div>

      <div className="relative flex min-h-screen">
        <aside className="hidden w-[248px] shrink-0 flex-col border-r border-white/[0.07] bg-[#0a0d13]/90 px-5 py-6 lg:flex">
          <a href="#overview" className="flex items-center gap-3 px-2">
            <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-gradient-to-br from-cyan-300 to-blue-500 text-[#071018] shadow-lg shadow-cyan-900/30">
              <Zap className="h-5 w-5 fill-current" />
            </div>
            <div>
              <p className="text-base font-semibold tracking-wide text-white">
                HYPERION
              </p>
              <p className="mt-0.5 text-[9px] font-bold uppercase tracking-[0.2em] text-slate-500">
                Inference control
              </p>
            </div>
          </a>

          <div className="mt-10 px-3 text-[10px] font-bold uppercase tracking-[0.19em] text-slate-600">
            Workspace
          </div>
          <nav className="mt-3 space-y-1">
            {navItems.map((item, index) => {
              const Icon = item.icon;
              return (
                <a
                  key={item.href}
                  href={item.href}
                  className={
                    "group flex items-center gap-3 rounded-xl px-3 py-2.5 text-sm transition " +
                    (index === 0
                      ? "border border-cyan-300/10 bg-cyan-300/[0.08] text-cyan-100"
                      : "text-slate-500 hover:bg-white/[0.04] hover:text-slate-200")
                  }
                >
                  <Icon
                    className={
                      "h-4 w-4 " +
                      (index === 0
                        ? "text-cyan-300"
                        : "text-slate-600 group-hover:text-slate-300")
                    }
                  />
                  {item.label}
                  {index === 0 && (
                    <span className="ml-auto h-1.5 w-1.5 rounded-full bg-cyan-300" />
                  )}
                </a>
              );
            })}
          </nav>

          <div className="mt-auto rounded-2xl border border-white/[0.07] bg-[#10151e] p-4">
            <div className="flex items-center gap-2">
              <span
                className={
                  "h-2 w-2 rounded-full " +
                  (topology ? "bg-emerald-400" : "bg-amber-400")
                }
              />
              <span className="text-xs font-medium text-slate-300">
                {topology ? "Gateway connected" : "Waiting for Gateway"}
              </span>
            </div>
            <p className="mt-3 text-[11px] leading-5 text-slate-500">
              Simulator is the default
              <br />
              Optional real vLLM on CPU
            </p>
            <div className="mt-3 flex items-center gap-1.5 text-[10px] font-semibold uppercase tracking-[0.14em] text-cyan-400/80">
              <ShieldCheck className="h-3.5 w-3.5" />
              Safe demo environment
            </div>
          </div>
        </aside>

        <div className="min-w-0 flex-1">
          <header className="sticky top-0 z-20 border-b border-white/[0.07] bg-[#080b10]/85 backdrop-blur-xl">
            <div className="mx-auto flex h-[68px] max-w-[1500px] items-center justify-between px-5 md:px-8">
              <div className="flex items-center gap-3">
                <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-cyan-300/10 text-cyan-300 lg:hidden">
                  <Zap className="h-4 w-4" />
                </div>
                <div>
                  <p className="text-[10px] font-bold uppercase tracking-[0.18em] text-slate-600">
                    Hyperion / Control room
                  </p>
                  <p className="mt-1 text-xs text-slate-400">
                    Local inference cluster
                  </p>
                </div>
              </div>

              <div className="flex items-center gap-3">
                <div className="hidden items-center gap-2 rounded-full border border-white/[0.07] bg-white/[0.025] px-3 py-1.5 sm:flex">
                  <span
                    className={
                      "h-1.5 w-1.5 rounded-full " +
                      (topology ? "bg-emerald-400" : "bg-amber-400")
                    }
                  />
                  <span className="text-[11px] text-slate-400">{status}</span>
                </div>
                <button
                  type="button"
                  onClick={() => void refresh()}
                  className="flex h-9 w-9 items-center justify-center rounded-xl border border-white/[0.08] bg-white/[0.025] text-slate-400 transition hover:border-white/[0.14] hover:text-white"
                  aria-label="Refresh dashboard"
                  title="Refresh"
                >
                  <RefreshCw className="h-4 w-4" />
                </button>
              </div>
            </div>
          </header>

          <div
            id="overview"
            className="mx-auto max-w-[1500px] px-5 pb-12 pt-8 md:px-8 md:pt-10"
          >
            <section className="relative overflow-hidden rounded-[26px] border border-cyan-300/[0.12] bg-gradient-to-br from-[#111d29] via-[#101723] to-[#10131c] p-6 shadow-2xl shadow-black/20 md:p-8">
              <div className="pointer-events-none absolute -right-16 -top-28 h-80 w-80 rounded-full border border-cyan-300/[0.06]" />
              <div className="pointer-events-none absolute -right-4 -top-16 h-56 w-56 rounded-full border border-cyan-300/[0.08]" />
              <div className="pointer-events-none absolute right-16 top-8 h-28 w-28 rounded-full bg-cyan-400/[0.07] blur-3xl" />
              <div className="relative grid gap-8 xl:grid-cols-[minmax(0,1fr)_330px] xl:items-center">
                <div>
                  <div className="inline-flex items-center gap-2 rounded-full border border-cyan-300/15 bg-cyan-300/[0.06] px-3 py-1.5">
                    <Cpu className="h-3.5 w-3.5 text-cyan-300" />
                    <span className="text-[10px] font-bold uppercase tracking-[0.16em] text-cyan-100/80">
                      Simulator default · real CPU inference optional
                    </span>
                  </div>
                  <h1 className="mt-5 max-w-3xl text-3xl font-semibold leading-tight tracking-[-0.04em] text-white sm:text-4xl lg:text-[42px]">
                    Inference control,
                    <span className="text-cyan-300"> made observable.</span>
                  </h1>
                  <p className="mt-4 max-w-2xl text-sm leading-7 text-slate-400 sm:text-[15px]">
                    Explore how traffic, worker capacity, prefix reuse, and
                    latency targets interact. The simulator is the default;
                    you can opt into a real small CPU model below.
                  </p>
                  <div className="mt-6 flex flex-wrap items-center gap-3">
                    <button
                      type="button"
                      onClick={() => void runDemo()}
                      disabled={loading}
                      className="inline-flex items-center gap-2 rounded-xl bg-cyan-300 px-4 py-2.5 text-sm font-semibold text-[#071018] shadow-lg shadow-cyan-950/30 transition hover:bg-cyan-200 disabled:cursor-wait disabled:opacity-60"
                    >
                      <Play className="h-4 w-4 fill-current" />
                      {loading ? "Sending traffic…" : "Run traffic burst"}
                    </button>
                    <a
                      href="#capacity"
                      className="inline-flex items-center gap-2 rounded-xl border border-white/[0.09] bg-white/[0.025] px-4 py-2.5 text-sm font-medium text-slate-300 transition hover:bg-white/[0.06] hover:text-white"
                    >
                      View capacity study
                      <ArrowRight className="h-4 w-4" />
                    </a>
                  </div>
                </div>

                <div className="relative rounded-2xl border border-white/[0.08] bg-[#080d14]/65 p-5 backdrop-blur">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-[10px] font-bold uppercase tracking-[0.18em] text-slate-500">
                        Current topology
                      </p>
                      <p className="mt-1 text-xs text-slate-400">
                        Gateway snapshot
                      </p>
                    </div>
                    <div className="rounded-lg border border-emerald-400/10 bg-emerald-300/[0.06] p-2 text-emerald-300">
                      <Network className="h-4 w-4" />
                    </div>
                  </div>
                  <div className="mt-5 grid grid-cols-2 gap-3">
                    <div className="rounded-xl border border-white/[0.06] bg-white/[0.025] p-3">
                      <p className="text-[10px] uppercase tracking-[0.14em] text-slate-600">
                        Prefill
                      </p>
                      <p className="mt-2 text-2xl font-semibold text-white">
                        {state?.prefill_replicas ?? "—"}
                      </p>
                      <p className="mt-1 text-[10px] text-slate-500">
                        {topology?.queues.prefill ?? 0} waiting
                      </p>
                    </div>
                    <div className="rounded-xl border border-white/[0.06] bg-white/[0.025] p-3">
                      <p className="text-[10px] uppercase tracking-[0.14em] text-slate-600">
                        Decode
                      </p>
                      <p className="mt-2 text-2xl font-semibold text-white">
                        {state?.decode_replicas ?? "—"}
                      </p>
                      <p className="mt-1 text-[10px] text-slate-500">
                        {topology?.queues.decode ?? 0} waiting
                      </p>
                    </div>
                  </div>
                  <div className="mt-4 flex items-center gap-2 border-t border-white/[0.07] pt-4">
                    <span className="h-1.5 w-1.5 rounded-full bg-cyan-300" />
                    <p className="text-xs text-slate-400">
                      Last decision:{" "}
                      <span className="font-medium text-cyan-100">
                        {topology?.last_controller_decision?.replaceAll(
                          "_",
                          " ",
                        ) ?? "waiting for metrics"}
                      </span>
                    </p>
                  </div>
                </div>
              </div>
            </section>

            <section className="mt-6 grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
              <StatCard
                label="Prefill replicas"
                value={state?.prefill_replicas ?? "—"}
                detail={"Batch capacity " + (state?.prefill_batch_size ?? "—") + " per worker"}
                icon={Layers3}
                accent="bg-sky-400/[0.08] text-sky-300"
              />
              <StatCard
                label="Decode replicas"
                value={state?.decode_replicas ?? "—"}
                detail={"Batch capacity " + (state?.decode_batch_size ?? "—") + " per worker"}
                icon={Zap}
                accent="bg-violet-400/[0.08] text-violet-300"
              />
              <StatCard
                label="Queued requests"
                value={
                  topology
                    ? topology.queues.prefill + topology.queues.decode
                    : "—"
                }
                detail={
                  topology
                    ? topology.queues.prefill + " prefill · " + topology.queues.decode + " decode"
                    : "Waiting for Gateway data"
                }
                icon={Activity}
                accent="bg-amber-300/[0.08] text-amber-200"
              />
              <StatCard
                label="Cached prefixes"
                value={topology?.cache_entries ?? "—"}
                detail={
                  state?.aggressive_prefix_cache
                    ? "Aggressive reuse mode enabled"
                    : "Standard reuse mode"
                }
                icon={Database}
                accent="bg-emerald-300/[0.08] text-emerald-300"
              />
            </section>

            <section id="try-request" className="mt-10 scroll-mt-24">
              <SectionHeading
                eyebrow="Live request"
                title="Try Hyperion"
                detail="Send one OpenAI-compatible request through the Gateway. In simulator mode it returns a labeled placeholder; with the optional vLLM CPU profile it runs the downloaded model."
              />
              <div className="grid gap-4 xl:grid-cols-[0.9fr_1.1fr]">
                <form onSubmit={sendChat} className="rounded-2xl border border-white/[0.07] bg-[#0e131c] p-5 md:p-6">
                  <label htmlFor="chat-prompt" className="text-xs font-semibold text-slate-300">Your prompt</label>
                  <textarea
                    id="chat-prompt"
                    value={chatPrompt}
                    onChange={(event) => setChatPrompt(event.target.value)}
                    rows={5}
                    required
                    className="mt-3 w-full resize-y rounded-xl border border-white/[0.08] bg-[#080b10] p-3 text-sm leading-6 text-slate-200 outline-none transition placeholder:text-slate-600 focus:border-cyan-300/40"
                  />
                  <div className="mt-3 flex items-center justify-between gap-3">
                    <span className="text-[10px] text-slate-600">Model mode is reported with the reply.</span>
                    <button
                      type="submit"
                      disabled={chatBusy || !chatPrompt.trim()}
                      className="inline-flex shrink-0 items-center gap-2 rounded-xl bg-cyan-300 px-4 py-2.5 text-xs font-semibold text-[#071018] transition hover:bg-cyan-200 disabled:cursor-wait disabled:opacity-50"
                    >
                      <Play className="h-3.5 w-3.5 fill-current" />
                      {chatBusy ? "Generating…" : "Send request"}
                    </button>
                  </div>
                </form>
                <div className="min-h-48 rounded-2xl border border-white/[0.07] bg-[#0e131c] p-5 md:p-6">
                  <div className="flex items-center justify-between gap-3">
                    <p className="text-xs font-semibold text-slate-300">Gateway response</p>
                    {chatMode && <span className="rounded-full border border-cyan-300/10 bg-cyan-300/[0.05] px-2.5 py-1 text-[10px] text-cyan-100/80">{chatMode}</span>}
                  </div>
                  <p className="mt-4 whitespace-pre-wrap text-sm leading-7 text-slate-300">
                    {chatAnswer || "Your response will appear here. A simulator reply is intentionally labeled and is not generated by an LLM."}
                  </p>
                </div>
              </div>
            </section>

            <section id="capacity" className="mt-12 scroll-mt-24">
              <SectionHeading
                eyebrow="Offline performance study"
                title="Capacity planner"
                detail="Compare tested replica counts under the same request mix. The smallest profile meeting every target is recommended."
              />

              {recommendation ? (
                <div className="grid gap-5 xl:grid-cols-[minmax(0,1.35fr)_minmax(310px,0.65fr)]">
                  <div className="rounded-2xl border border-white/[0.07] bg-[#0e131c] p-5 md:p-6">
                    <div className="flex flex-col justify-between gap-4 border-b border-white/[0.07] pb-5 sm:flex-row sm:items-center">
                      <div>
                        <p className="text-sm font-semibold text-white">
                          TTFT by replica count
                        </p>
                        <p className="mt-1 text-xs text-slate-500">
                          {recommendation.workload.model} · {recommendation.workload.gpu_profile.toUpperCase()} profile
                        </p>
                      </div>
                      <div className="flex flex-wrap gap-2 text-[10px]">
                        <span className="rounded-lg border border-white/[0.07] bg-white/[0.025] px-2.5 py-1.5 text-slate-400">
                          {recommendation.workload.qps} QPS
                        </span>
                        <span className="rounded-lg border border-white/[0.07] bg-white/[0.025] px-2.5 py-1.5 text-slate-400">
                          {recommendation.workload.request_count} requests
                        </span>
                        <span className="rounded-lg border border-white/[0.07] bg-white/[0.025] px-2.5 py-1.5 text-slate-400">
                          {recommendation.workload.prompt_tokens} in / {recommendation.workload.output_tokens} out
                        </span>
                      </div>
                    </div>

                    {profiles.length > 0 ? (
                      <div className="mt-5 space-y-5">
                        {profiles.map((item) => {
                          const width = Math.max(
                            3,
                            Math.min(
                              100,
                              (item.latency.ttft_p95_ms / maxTtft) * 100,
                            ),
                          );
                          return (
                            <div key={item.run}>
                              <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
                                <div className="flex items-center gap-2.5">
                                  <span className="flex h-7 w-7 items-center justify-center rounded-lg border border-white/[0.07] bg-white/[0.03] text-xs font-semibold text-slate-200">
                                    {item.replicas}
                                  </span>
                                  <span className="text-sm font-medium text-slate-200">
                                    {item.replicas} replica{item.replicas === 1 ? "" : "s"}
                                  </span>
                                  {item.meets_targets ? (
                                    <span className="rounded-full border border-emerald-300/15 bg-emerald-300/[0.07] px-2 py-0.5 text-[9px] font-bold uppercase tracking-[0.12em] text-emerald-300">
                                      Meets targets
                                    </span>
                                  ) : (
                                    <span className="rounded-full border border-rose-300/10 bg-rose-300/[0.05] px-2 py-0.5 text-[9px] font-bold uppercase tracking-[0.12em] text-rose-300/80">
                                      Misses targets
                                    </span>
                                  )}
                                </div>
                                <span className="font-mono text-sm text-slate-300">
                                  {item.latency.ttft_p95_ms.toFixed(0)}
                                  <span className="ml-1 text-xs text-slate-600">ms</span>
                                </span>
                              </div>
                              <div className="h-2 overflow-hidden rounded-full bg-[#1b2430]">
                                <div
                                  className={
                                    "h-full rounded-full transition-all " +
                                    (item.meets_targets
                                      ? "bg-gradient-to-r from-cyan-400 to-emerald-300"
                                      : "bg-gradient-to-r from-indigo-500 to-sky-400")
                                  }
                                  style={{ width: width + "%" }}
                                />
                              </div>
                              <div className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-[10px] text-slate-500">
                                <span>E2E p95 {item.latency.e2e_p95_seconds.toFixed(2)} sec</span>
                                <span>Queue p95 {item.latency.scheduling_delay_p95_ms.toFixed(0)} ms</span>
                                <span>Run {item.run}</span>
                              </div>
                            </div>
                          );
                        })}
                      </div>
                    ) : (
                      <div className="mt-6 rounded-xl border border-dashed border-white/[0.1] px-5 py-8 text-center">
                        <Database className="mx-auto h-5 w-5 text-slate-600" />
                        <p className="mt-3 text-sm font-medium text-slate-300">
                          No capacity profiles yet
                        </p>
                        <p className="mt-1 text-xs text-slate-500">
                          Run Vidur with two or more replica counts to compare capacity.
                        </p>
                      </div>
                    )}
                  </div>

                  <div className="rounded-2xl border border-cyan-300/[0.12] bg-gradient-to-b from-[#10202b] to-[#0d131b] p-5 md:p-6">
                    <div className="flex items-start justify-between">
                      <div>
                        <div className="flex items-center gap-2 text-cyan-300">
                          <SlidersHorizontal className="h-4 w-4" />
                          <p className="text-[10px] font-bold uppercase tracking-[0.17em]">
                            Latency objectives
                          </p>
                        </div>
                        <p className="mt-2 text-xs leading-5 text-slate-500">
                          Set the limits your service should meet.
                        </p>
                      </div>
                      <Sparkles className="h-4 w-4 text-cyan-300/50" />
                    </div>

                    <form className="mt-5 space-y-3" onSubmit={applyTargets}>
                      <label className="block">
                        <span className="mb-1.5 flex items-center justify-between text-xs text-slate-400">
                          TTFT p95
                          <span className="font-mono text-[10px] text-slate-600">milliseconds</span>
                        </span>
                        <input
                          className="w-full rounded-xl border border-white/[0.08] bg-[#080c12] px-3.5 py-2.5 font-mono text-sm text-white outline-none transition placeholder:text-slate-700 focus:border-cyan-300/40 focus:ring-2 focus:ring-cyan-300/10"
                          type="number"
                          min="1"
                          step="10"
                          required
                          value={targetDraft.ttft_p95_ms}
                          onChange={(event) =>
                            setTargetDraft({
                              ...targetDraft,
                              ttft_p95_ms: event.target.value,
                            })
                          }
                        />
                      </label>
                      <label className="block">
                        <span className="mb-1.5 flex items-center justify-between text-xs text-slate-400">
                          End-to-end p95
                          <span className="font-mono text-[10px] text-slate-600">seconds</span>
                        </span>
                        <input
                          className="w-full rounded-xl border border-white/[0.08] bg-[#080c12] px-3.5 py-2.5 font-mono text-sm text-white outline-none transition focus:border-cyan-300/40 focus:ring-2 focus:ring-cyan-300/10"
                          type="number"
                          min="0.1"
                          step="0.5"
                          required
                          value={targetDraft.e2e_p95_seconds}
                          onChange={(event) =>
                            setTargetDraft({
                              ...targetDraft,
                              e2e_p95_seconds: event.target.value,
                            })
                          }
                        />
                      </label>
                      <label className="block">
                        <span className="mb-1.5 flex items-center justify-between text-xs text-slate-400">
                          Queue delay p95
                          <span className="font-mono text-[10px] text-slate-600">milliseconds</span>
                        </span>
                        <input
                          className="w-full rounded-xl border border-white/[0.08] bg-[#080c12] px-3.5 py-2.5 font-mono text-sm text-white outline-none transition focus:border-cyan-300/40 focus:ring-2 focus:ring-cyan-300/10"
                          type="number"
                          min="1"
                          step="10"
                          required
                          value={targetDraft.scheduling_delay_p95_ms}
                          onChange={(event) =>
                            setTargetDraft({
                              ...targetDraft,
                              scheduling_delay_p95_ms: event.target.value,
                            })
                          }
                        />
                      </label>
                      <button
                        type="submit"
                        className="mt-2 inline-flex w-full items-center justify-center gap-2 rounded-xl bg-cyan-300 px-4 py-2.5 text-sm font-semibold text-[#061018] transition hover:bg-cyan-200"
                      >
                        Recalculate recommendation
                        <ArrowRight className="h-4 w-4" />
                      </button>
                    </form>

                    <div className="mt-5 border-t border-white/[0.08] pt-5">
                      {recommendation.recommendation ? (
                        <div className="flex items-center gap-3">
                          <div className="flex h-11 w-11 items-center justify-center rounded-xl bg-emerald-300/10 text-emerald-300">
                            <Server className="h-5 w-5" />
                          </div>
                          <div>
                            <p className="text-[10px] font-bold uppercase tracking-[0.15em] text-slate-500">
                              Lowest passing profile
                            </p>
                            <p className="mt-1 text-lg font-semibold text-white">
                              {recommendation.recommendation.replicas} replicas
                            </p>
                          </div>
                        </div>
                      ) : (
                        <p className="text-sm font-medium text-amber-200">
                          No tested profile meets all three limits.
                        </p>
                      )}
                      <p className="mt-3 text-[10px] leading-5 text-slate-500">
                        This is an offline Vidur estimate, not a physical GPU measurement.
                      </p>
                    </div>
                  </div>
                </div>
              ) : (
                <div className="rounded-2xl border border-dashed border-white/[0.1] bg-[#0e131c]/70 px-6 py-10 text-center">
                  <Gauge className="mx-auto h-6 w-6 text-slate-600" />
                  <p className="mt-3 text-sm font-medium text-slate-300">
                    Run a Vidur study to unlock capacity planning
                  </p>
                  <p className="mt-1 text-xs text-slate-500">
                    The dashboard continues to work without saved Vidur profiles.
                  </p>
                </div>
              )}
            </section>

            <section id="workers" className="mt-12 scroll-mt-24">
              <SectionHeading
                eyebrow="Live simulated runtime"
                title="Worker pools"
                detail="These are the topology values currently held by the CPU simulator. They are separate logical stages, not separate model servers."
              />
              <div className="grid gap-4 xl:grid-cols-2">
                <div className="rounded-2xl border border-white/[0.07] bg-[#0e131c] p-5 md:p-6">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-3">
                      <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-sky-300/[0.08] text-sky-300">
                        <Layers3 className="h-5 w-5" />
                      </div>
                      <div>
                        <h3 className="font-semibold text-white">Prefill pool</h3>
                        <p className="mt-1 text-xs text-slate-500">Prompt processing stage</p>
                      </div>
                    </div>
                    <span className="rounded-lg border border-sky-300/10 bg-sky-300/[0.05] px-2.5 py-1 text-[10px] font-semibold text-sky-200">
                      COMPUTE MODEL
                    </span>
                  </div>
                  <div className="mt-6 grid grid-cols-3 gap-3">
                    <div className="rounded-xl bg-white/[0.025] p-3">
                      <p className="text-[10px] text-slate-600">Replicas</p>
                      <p className="mt-2 text-xl font-semibold text-white">{state?.prefill_replicas ?? "—"}</p>
                    </div>
                    <div className="rounded-xl bg-white/[0.025] p-3">
                      <p className="text-[10px] text-slate-600">Batch size</p>
                      <p className="mt-2 text-xl font-semibold text-white">{state?.prefill_batch_size ?? "—"}</p>
                    </div>
                    <div className="rounded-xl bg-white/[0.025] p-3">
                      <p className="text-[10px] text-slate-600">Queue</p>
                      <p className="mt-2 text-xl font-semibold text-white">{topology?.queues.prefill ?? "—"}</p>
                    </div>
                  </div>
                  <div className="mt-4 h-1.5 overflow-hidden rounded-full bg-slate-800">
                    <div
                      className="h-full rounded-full bg-sky-300"
                      style={{
                        width:
                          state && state.prefill_replicas > 0
                            ? Math.min(
                                100,
                                (topology!.queues.prefill /
                                  (state.prefill_replicas *
                                    state.prefill_batch_size)) *
                                  100,
                              ) + "%"
                            : "0%",
                      }}
                    />
                  </div>
                  <p className="mt-2 text-[10px] text-slate-600">
                    Queue pressure relative to configured batch capacity
                  </p>
                </div>

                <div className="rounded-2xl border border-white/[0.07] bg-[#0e131c] p-5 md:p-6">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-3">
                      <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-violet-300/[0.08] text-violet-300">
                        <Zap className="h-5 w-5" />
                      </div>
                      <div>
                        <h3 className="font-semibold text-white">Decode pool</h3>
                        <p className="mt-1 text-xs text-slate-500">Token generation stage</p>
                      </div>
                    </div>
                    <span className="rounded-lg border border-violet-300/10 bg-violet-300/[0.05] px-2.5 py-1 text-[10px] font-semibold text-violet-200">
                      MEMORY MODEL
                    </span>
                  </div>
                  <div className="mt-6 grid grid-cols-3 gap-3">
                    <div className="rounded-xl bg-white/[0.025] p-3">
                      <p className="text-[10px] text-slate-600">Replicas</p>
                      <p className="mt-2 text-xl font-semibold text-white">{state?.decode_replicas ?? "—"}</p>
                    </div>
                    <div className="rounded-xl bg-white/[0.025] p-3">
                      <p className="text-[10px] text-slate-600">Batch size</p>
                      <p className="mt-2 text-xl font-semibold text-white">{state?.decode_batch_size ?? "—"}</p>
                    </div>
                    <div className="rounded-xl bg-white/[0.025] p-3">
                      <p className="text-[10px] text-slate-600">Queue</p>
                      <p className="mt-2 text-xl font-semibold text-white">{topology?.queues.decode ?? "—"}</p>
                    </div>
                  </div>
                  <div className="mt-4 h-1.5 overflow-hidden rounded-full bg-slate-800">
                    <div
                      className="h-full rounded-full bg-violet-300"
                      style={{
                        width:
                          state && state.decode_replicas > 0
                            ? Math.min(
                                100,
                                (topology!.queues.decode /
                                  (state.decode_replicas *
                                    state.decode_batch_size)) *
                                  100,
                              ) + "%"
                            : "0%",
                      }}
                    />
                  </div>
                  <p className="mt-2 text-[10px] text-slate-600">
                    Queue pressure relative to configured batch capacity
                  </p>
                </div>
              </div>
            </section>

            <section
              id="controller"
              className="mt-12 grid scroll-mt-24 gap-5 xl:grid-cols-[0.8fr_1.2fr]"
            >
              <div className="rounded-2xl border border-white/[0.07] bg-[#0e131c] p-5 md:p-6">
                <SectionHeading
                  eyebrow="Autonomous policy loop"
                  title="Controller state"
                  detail="Rules inspect simulated queue and cache counts."
                />
                <div className="rounded-xl border border-indigo-300/10 bg-indigo-300/[0.04] p-4">
                  <div className="flex items-center gap-3">
                    <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-indigo-300/10 text-indigo-200">
                      <BrainCircuit className="h-5 w-5" />
                    </div>
                    <div>
                      <p className="text-[10px] font-bold uppercase tracking-[0.14em] text-slate-500">
                        Last decision
                      </p>
                      <p className="mt-1 font-semibold capitalize text-indigo-100">
                        {topology?.last_controller_decision?.replaceAll(
                          "_",
                          " ",
                        ) ?? "Waiting for data"}
                      </p>
                    </div>
                  </div>
                </div>
                <div className="mt-4 space-y-3">
                  <div className="flex items-center justify-between rounded-xl bg-white/[0.025] px-3.5 py-3">
                    <span className="flex items-center gap-2 text-xs text-slate-400">
                      <Database className="h-3.5 w-3.5 text-emerald-300" />
                      Aggressive prefix cache
                    </span>
                    <span className={state?.aggressive_prefix_cache ? "text-xs font-semibold text-emerald-300" : "text-xs text-slate-600"}>
                      {state?.aggressive_prefix_cache ? "ON" : "OFF"}
                    </span>
                  </div>
                  <div className="flex items-center justify-between rounded-xl bg-white/[0.025] px-3.5 py-3">
                    <span className="flex items-center gap-2 text-xs text-slate-400">
                      <Sparkles className="h-3.5 w-3.5 text-violet-300" />
                      Speculative decode flag
                    </span>
                    <span className={state?.speculative_decoding ? "text-xs font-semibold text-violet-300" : "text-xs text-slate-600"}>
                      {state?.speculative_decoding ? "ON" : "OFF"}
                    </span>
                  </div>
                </div>
                <p className="mt-4 text-[10px] leading-5 text-slate-600">
                  The switches show simulated controller state. They do not enable model kernels in this CPU demo.
                </p>
              </div>

              <div className="rounded-2xl border border-white/[0.07] bg-[#0e131c] p-5 md:p-6">
                <SectionHeading
                  eyebrow="Experiment comparison"
                  title="Before and after"
                  detail="Compare the most recent two saved profiles when their workload settings match."
                />
                {comparison?.same_workload ? (
                  <>
                    <div className="mb-4 flex flex-wrap gap-2 text-[10px] text-slate-500">
                      <span className="rounded-lg bg-white/[0.03] px-2.5 py-1.5">
                        {comparison.baseline.replicas} → {comparison.candidate.replicas} replicas
                      </span>
                      <span className="rounded-lg bg-white/[0.03] px-2.5 py-1.5">
                        {comparison.baseline.qps} QPS
                      </span>
                      <span className="rounded-lg bg-white/[0.03] px-2.5 py-1.5">
                        {comparison.baseline.request_count} requests
                      </span>
                    </div>
                    <div className="overflow-hidden rounded-xl border border-white/[0.06]">
                      {comparisonRows.map((row) => {
                        const reduction =
                          row.baseline === 0
                            ? null
                            : ((row.baseline - row.candidate) /
                                row.baseline) *
                              100;
                        return (
                          <div
                            key={row.label}
                            className="grid grid-cols-[1fr_auto_auto] items-center gap-3 border-b border-white/[0.05] px-3.5 py-3 last:border-b-0 sm:grid-cols-[1fr_110px_110px_100px]"
                          >
                            <span className="text-xs text-slate-400">{row.label}</span>
                            <span className="text-right font-mono text-xs text-slate-500">
                              {row.baseline.toFixed(2)} {row.unit}
                            </span>
                            <span className="text-right font-mono text-xs text-slate-200">
                              {row.candidate.toFixed(2)} {row.unit}
                            </span>
                            <span
                              className={
                                "col-span-3 text-right text-[10px] font-semibold sm:col-span-1 " +
                                (reduction !== null && reduction >= 0
                                  ? "text-emerald-300"
                                  : "text-rose-300")
                              }
                            >
                              {reduction === null
                                ? "—"
                                : reduction >= 0
                                  ? reduction.toFixed(1) + "% lower"
                                  : Math.abs(reduction).toFixed(1) + "% higher"}
                            </span>
                          </div>
                        );
                      })}
                    </div>
                    <p className="mt-3 text-[10px] text-slate-600">
                      {comparison.baseline_run} → {comparison.candidate_run}
                    </p>
                  </>
                ) : (
                  <div className="rounded-xl border border-dashed border-white/[0.1] px-4 py-8 text-center text-xs text-slate-500">
                    {comparison
                      ? "The latest profiles use different workloads. Run a matching pair to compare fairly."
                      : "Save at least two Vidur profiles to see a comparison."}
                  </div>
                )}
              </div>
            </section>

            <section id="request-path" className="mt-12 scroll-mt-24">
              <SectionHeading
                eyebrow="Request lifecycle"
                title="One request, four stages"
                detail="This diagram describes the simulator's logical request path."
              />
              <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
                {[
                  {
                    step: "01",
                    title: "Gateway",
                    text: "Validates an OpenAI-shaped request and records timing.",
                    icon: Network,
                    color: "text-cyan-300",
                  },
                  {
                    step: "02",
                    title: "Prefill",
                    text: "Models processing the prompt and checks prefix reuse.",
                    icon: Layers3,
                    color: "text-sky-300",
                  },
                  {
                    step: "03",
                    title: "KV cache",
                    text: "Keeps a small in-memory record for shared prefixes.",
                    icon: Database,
                    color: "text-emerald-300",
                  },
                  {
                    step: "04",
                    title: "Decode",
                    text: "Models token generation and reports queue pressure.",
                    icon: Zap,
                    color: "text-violet-300",
                  },
                ].map((item, index) => {
                  const Icon = item.icon;
                  return (
                    <div
                      key={item.step}
                      className="relative rounded-2xl border border-white/[0.07] bg-[#0e131c] p-5"
                    >
                      <div className="flex items-center justify-between">
                        <span className="font-mono text-[10px] text-slate-600">
                          {item.step}
                        </span>
                        <Icon className={"h-4 w-4 " + item.color} />
                      </div>
                      <h3 className="mt-5 text-sm font-semibold text-white">
                        {item.title}
                      </h3>
                      <p className="mt-2 text-xs leading-5 text-slate-500">
                        {item.text}
                      </p>
                      {index < 3 && (
                        <ArrowRight className="absolute -right-3 top-1/2 z-10 hidden h-5 w-5 -translate-y-1/2 rounded-full border border-white/[0.08] bg-[#0e131c] p-0.5 text-slate-500 xl:block" />
                      )}
                    </div>
                  );
                })}
              </div>
            </section>

            <footer className="mt-12 flex flex-col justify-between gap-3 border-t border-white/[0.07] pt-5 text-[10px] text-slate-600 sm:flex-row sm:items-center">
              <span>Hyperion · Local inference systems lab</span>
              <span className="flex items-center gap-2">
                <Timer className="h-3.5 w-3.5" />
                Topology refreshes every 3 seconds
                <span className="mx-1 h-1 w-1 rounded-full bg-slate-700" />
                <span className="flex items-center gap-1">
                  <Gauge className="h-3.5 w-3.5" />
                  Vidur results are estimates
                </span>
              </span>
            </footer>
          </div>
        </div>
      </div>
    </main>
  );
}
