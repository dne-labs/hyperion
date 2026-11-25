"use client";

import { useCallback, useEffect, useMemo, useState, type ReactNode } from "react";
import { Activity, ArrowDownRight, Check, ChevronDown, CircleHelp, Cpu, Gauge, Layers3, Play, RefreshCw, Server, SlidersHorizontal, Sparkles, Zap } from "lucide-react";

const gateway = process.env.NEXT_PUBLIC_GATEWAY_URL || "http://localhost:8000";
type Run = {
  run_id: string; model: string; gpu_profile: string; replicas: number; qps: number;
  request_count: number; prompt_tokens: number; output_tokens: number;
  latency: { ttft_p50_ms?: number; ttft_p90_ms?: number; ttft_p95_ms?: number; e2e_p95_seconds?: number; tpot_p50_ms?: number; tpot_p95_ms?: number; tpot_p99_ms?: number; scheduling_delay_p95_ms?: number };
  configuration?: { scheduler?: string; batch_size?: number; chunk_size?: number | null; pipeline_parallel?: number; tensor_parallel?: number };
  decode_samples_seconds: number[];
};
type FormState = {
  qps: string; requests: string; prompt: string; output: string; batch: string;
  scheduler: "sarathi" | "vllm"; chunk: string; alternate: string; cost: string;
  ttftSlo: string; tpotSlo: string;
};
const defaults: FormState = { qps: "4", requests: "32", prompt: "2048", output: "512", batch: "128", scheduler: "sarathi", chunk: "512", alternate: "256", cost: "20", ttftSlo: "2.0", tpotSlo: "0.20" };

function MetricCard({ icon: Icon, label, value, hint }: { icon: typeof Cpu; label: string; value: string; hint: string }) {
  return <article className="metric-card"><div className="metric-top"><span className="metric-icon"><Icon size={17} /></span><span className="metric-tag">VIDUR ESTIMATE</span></div><p className="metric-label">{label}</p><strong className="metric-value">{value}</strong><p className="metric-hint">{hint}</p></article>;
}
function InputField({ label, value, suffix, onChange, min, step }: { label: string; value: string; suffix: string; onChange: (v: string) => void; min?: number; step?: string }) {
  return <label className="control-field"><span>{label}</span><div><input type="number" min={min} step={step || "any"} value={value} onChange={(e) => onChange(e.target.value)} /><small>{suffix}</small></div></label>;
}
function Pill({ children }: { children: ReactNode }) { return <span className="run-pill">{children}</span>; }

function Scatter({
  runs, xValue, yValue, xLimit, yLimit, xLabel, yLabel, bestId, cost, colorMap, isCompliant,
}: {
  runs: Run[]; xValue: (run: Run) => number; yValue: (run: Run) => number;
  xLimit?: number; yLimit?: number; xLabel: string; yLabel: string;
  bestId?: string; cost: number; colorMap?: boolean; isCompliant?: (run: Run) => boolean;
}) {
  const W = 560, H = 300, L = 63, R = 18, T = 19, B = 49, PW = W - L - R, PH = H - T - B;
  const xs = runs.map(xValue), ys = runs.map(yValue);
  const xMax = Math.max(xLimit || 0, ...xs, 0.01) * 1.16;
  const yMax = Math.max(yLimit || 0, ...ys, 0.01) * 1.16;
  const x = (n: number) => L + (n / xMax) * PW;
  const y = (n: number) => T + PH - (n / yMax) * PH;
  const ratios = [0, .25, .5, .75, 1];
  return <svg viewBox={"0 0 " + W + " " + H} className="plot" role="img" aria-label={yLabel + " by " + xLabel}>
    {xLimit !== undefined && <rect x={L} y={T} width={Math.max(0, x(xLimit) - L)} height={PH} fill="#dff1e5" />}
    {yLimit !== undefined && <rect x={L} y={y(yLimit)} width={PW} height={T + PH - y(yLimit)} fill="#dff1e5" opacity=".6" />}
    {ratios.map((r) => <g key={r}><line x1={L} x2={L + PW} y1={T + PH * (1 - r)} y2={T + PH * (1 - r)} stroke="#e4eaed" /><line x1={L + PW * r} x2={L + PW * r} y1={T} y2={T + PH} stroke="#e9eef0" /><text x={L - 8} y={T + PH * (1 - r) + 4} textAnchor="end" className="plot-tick">{(yMax * r).toFixed(2)}</text><text x={L + PW * r} y={T + PH + 17} textAnchor="middle" className="plot-tick">{(xMax * r).toFixed(2)}</text></g>)}
    {xLimit !== undefined && <line x1={x(xLimit)} x2={x(xLimit)} y1={T} y2={T + PH} stroke="#ef5350" strokeWidth="1.5" />}
    {yLimit !== undefined && <line x1={L} x2={L + PW} y1={y(yLimit)} y2={y(yLimit)} stroke="#ef5350" strokeWidth="1.5" />}
    <line x1={L} x2={L} y1={T} y2={T + PH} stroke="#93a1a8" /><line x1={L} x2={L + PW} y1={T + PH} y2={T + PH} stroke="#93a1a8" />
    {runs.map((run) => {
      const compliant = isCompliant
        ? isCompliant(run)
        : (xLimit === undefined || xValue(run) <= xLimit) && (yLimit === undefined || yValue(run) <= yLimit);
      const efficiency = run.qps / Math.max(1, run.replicas) / Math.max(0.01, cost);
      const maxEfficiency = Math.max(...runs.map((item) => item.qps / Math.max(1, item.replicas) / Math.max(0.01, cost)), 0.01);
      const color = colorMap ? "hsl(" + (220 - Math.min(170, efficiency / maxEfficiency * 170)) + " 72% 46%)" : compliant ? "#16804a" : "#4256e8";
      return <g key={run.run_id}><circle cx={x(xValue(run))} cy={y(yValue(run))} r={run.run_id === bestId ? 7 : 5} fill={color} stroke="white" strokeWidth="1.5" /><title>{run.replicas + " replicas · " + (run.configuration?.scheduler || "scheduler") + " · " + run.run_id}</title>{run.run_id === bestId && <text x={x(xValue(run))} y={y(yValue(run)) + 5} textAnchor="middle" fill="#f59e0b" fontSize="18">★</text>}</g>;
    })}
    <text x={L + PW / 2} y={H - 7} textAnchor="middle" className="plot-label">{xLabel}</text>
    <text x="15" y={T + PH / 2} transform={"rotate(-90 15 " + (T + PH / 2) + ")"} textAnchor="middle" className="plot-label">{yLabel}</text>
  </svg>;
}

function Cdf({ run }: { run?: Run }) {
  const values = [...(run?.decode_samples_seconds || [])].sort((a, b) => a - b);
  const W = 820, H = 270, L = 60, R = 20, T = 18, B = 45, PW = W - L - R, PH = H - T - B;
  const max = Math.max(...values, 0.01) * 1.05;
  const points = values.map((v, i) => (L + (v / max) * PW) + "," + (T + PH - ((i + 1) / values.length) * PH)).join(" ");
  return <svg viewBox={"0 0 " + W + " " + H} className="plot cdf-plot" role="img" aria-label="Cumulative distribution of normalized decode time">
    {[0,.25,.5,.75,1].map((r) => <g key={r}><line x1={L} x2={L + PW} y1={T + PH * (1-r)} y2={T + PH * (1-r)} stroke="#e3eaed" /><text x={L - 9} y={T + PH * (1-r) + 4} textAnchor="end" className="plot-tick">{r.toFixed(2)}</text></g>)}
    <line x1={L} x2={L} y1={T} y2={T+PH} stroke="#93a1a8" /><line x1={L} x2={L+PW} y1={T+PH} y2={T+PH} stroke="#93a1a8" />
    {values.length > 1 && <polyline points={points} fill="none" stroke="#ed4a4a" strokeWidth="2.2" />}
    {values.length < 2 && <text x={L + PW/2} y={T + PH/2} textAnchor="middle" className="plot-label">Run Vidur to see per-request latency distribution.</text>}
    <text x={L+PW/2} y={H-7} textAnchor="middle" className="plot-label">Normalized decode time per token (seconds)</text>
    <text x="16" y={T+PH/2} transform={"rotate(-90 16 " + (T+PH/2) + ")"} textAnchor="middle" className="plot-label">CDF</text>
  </svg>;
}

export default function Home() {
  const [runs, setRuns] = useState<Run[]>([]);
  const [form, setForm] = useState(defaults);
  const [gatewayOnline, setGatewayOnline] = useState(false);
  const [optimizerOnline, setOptimizerOnline] = useState(false);
  const [busy, setBusy] = useState(false);
  const [notice, setNotice] = useState("");
  const [selectedId, setSelectedId] = useState("");

  const refresh = useCallback(async () => {
    try {
      const [healthResponse, runsResponse] = await Promise.all([
        fetch(gateway + "/healthz", { cache: "no-store" }),
        fetch(gateway + "/optimization/runs", { cache: "no-store" }),
      ]);
      setGatewayOnline(healthResponse.ok);
      if (runsResponse.ok) {
        const data = await runsResponse.json();
        const nextRuns = (data.runs || []) as Run[];
        setRuns(nextRuns);
        setSelectedId((old) => old || nextRuns[nextRuns.length - 1]?.run_id || "");
      }
    } catch { setGatewayOnline(false); }
    try {
      const response = await fetch("http://localhost:8101/healthz", { cache: "no-store" });
      setOptimizerOnline(response.ok);
    } catch { setOptimizerOnline(false); }
  }, []);

  useEffect(() => {
    void refresh();
    const interval = window.setInterval(() => void refresh(), 8000);
    return () => window.clearInterval(interval);
  }, [refresh]);

  const cost = Math.max(.01, Number(form.cost) || .01);
  const ttftLimit = Math.max(.001, Number(form.ttftSlo) || .001);
  const tpotLimit = Math.max(.001, Number(form.tpotSlo) || .001);
  const candidates = useMemo(() => runs.map((run) => {
    const ttft = (run.latency.ttft_p90_ms ?? run.latency.ttft_p95_ms ?? 0) / 1000;
    const tpot = (run.latency.tpot_p99_ms ?? run.latency.tpot_p95_ms ?? 0) / 1000;
    const efficiency = run.qps / Math.max(1, run.replicas) / cost;
    return { run, ttft, tpot, efficiency, compliant: ttft <= ttftLimit && tpot <= tpotLimit };
  }), [runs, cost, ttftLimit, tpotLimit]);
  const bestItem = [...candidates].filter((item) => item.compliant).sort((a,b) => b.efficiency-a.efficiency)[0];
  const best = bestItem?.run;
  const selected = runs.find((run) => run.run_id === selectedId) || best || runs[runs.length - 1];
  const passing = candidates.filter((item) => item.compliant).length;

  function set<K extends keyof FormState>(key: K, value: FormState[K]) {
    setForm((old) => ({ ...old, [key]: value }));
  }

  async function optimize() {
    setBusy(true);
    setNotice("Vidur is running scenarios sequentially. This can take several minutes and use CPU.");
    try {
      const response = await fetch(gateway + "/optimization/run", {
        method: "POST", headers: { "content-type": "application/json" },
        body: JSON.stringify({
          qps: Number(form.qps), requests: Number(form.requests),
          prompt_tokens: Number(form.prompt), output_tokens: Number(form.output),
          batch_size: Number(form.batch), scheduler: form.scheduler,
          chunk_size: Number(form.chunk), alternate_chunk_size: Number(form.alternate),
          replicas: [1,2,4],
        }),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "Vidur run failed.");
      setNotice("Vidur finished " + (data.completed_runs?.length || 0) + " scenarios. Updating plots…");
      await refresh();
      setNotice("Optimization complete. The star marks the best tested configuration under your SLOs.");
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "Could not start the experiment.");
    } finally { setBusy(false); }
  }

  const efficiency = (run: Run) => run.qps / Math.max(1, run.replicas) / cost;
  const ttft = (run: Run) => (run.latency.ttft_p90_ms ?? run.latency.ttft_p95_ms ?? 0) / 1000;
  const tpot = (run: Run) => (run.latency.tpot_p99_ms ?? run.latency.tpot_p95_ms ?? 0) / 1000;

  return <main className="hyperion-page">
    <header className="topbar"><div className="topbar-inner">
      <a href="#top" className="wordmark"><span className="logo-mark"><Zap size={17} fill="currentColor" /></span><b>HYPERION</b><i>INFERENCE LAB</i></a>
      <nav><a href="#optimizer">Optimizer</a><a href="#frontier">Trade-offs</a><a href="#latency">Latency</a><a href="http://localhost:8000/docs" target="_blank" rel="noreferrer">API ↗</a></nav>
      <div className="topbar-actions"><span className={gatewayOnline ? "connection on" : "connection"}><i />{gatewayOnline ? "Gateway online" : "Gateway offline"}</span><button className="refresh-button" onClick={() => void refresh()} title="Refresh"><RefreshCw size={16} /></button></div>
    </div></header>

    <div id="top" className="page-container">
      <section className="hero">
        <div className="hero-copy">
          <span className="overline"><span /> INFERENCE OPTIMIZATION, WITH EVIDENCE</span>
          <h1>Find the config that<br /><em>earns its place.</em></h1>
          <p>Run controlled Vidur scenarios, compare their latency and cost trade-offs, and select the strongest configuration that meets your service goals.</p>
          <div className="hero-ctas"><a href="#optimizer" className="button-primary"><Sparkles size={16} /> Optimize a workload <ArrowDownRight size={16} /></a><a href="#frontier" className="button-quiet">Explore the frontier <ArrowDownRight size={15} /></a></div>
          <div className="honesty-note"><CircleHelp size={14} /> Vidur estimates GPU-profile behavior. It does not emulate a GPU or measure live hardware.</div>
        </div>
        <aside className="best-card">
          <div className="best-card-top"><span>BEST TESTED CONFIG</span><span className="estimate-badge"><i /> SIMULATION</span></div>
          {best ? <><div className="best-score">{efficiency(best).toFixed(3)} <small>offered QPS / $·h</small></div><p className="subtle">using your editable GPU-hour cost assumption</p><div className="best-tags"><Pill>{best.replicas} replicas</Pill><Pill>{best.configuration?.scheduler || "Sarathi"}</Pill><Pill>chunk {best.configuration?.chunk_size ?? form.chunk}</Pill></div><div className="best-stats"><div><small>TTFT p90</small><b>{ttft(best).toFixed(2)} s</b></div><div><small>Token time p99</small><b>{tpot(best).toFixed(3)} s</b></div></div></> : <div className="best-empty"><h2>No passing config yet</h2><p>Set your latency limits and run a Vidur sweep. Hyperion will mark the best tested point that meets both limits.</p></div>}
        </aside>
      </section>

      <section className="metrics-row">
        <MetricCard icon={Layers3} label="Saved candidates" value={String(runs.length)} hint="Vidur profiles available for comparison" />
        <MetricCard icon={Check} label="SLO compliant" value={passing + " / " + runs.length} hint="Candidates inside both latency limits" />
        <MetricCard icon={Gauge} label="Best QPS / $·h" value={best ? efficiency(best).toFixed(3) : "—"} hint="Offered QPS per replica and assumed hourly cost" />
        <MetricCard icon={Activity} label="Optimizer" value={optimizerOnline ? "Ready" : "Stopped"} hint={optimizerOnline ? "Vidur service can accept a sweep" : "Start the optional Vidur service below"} />
      </section>

      <section id="optimizer" className="content-section">
        <div className="section-intro"><div><span className="section-kicker">01 / SET THE EXPERIMENT</span><h2>What does “better” mean for this workload?</h2><p>Every candidate receives the same request mix. Hyperion compares replica count and a Sarathi chunk-size alternative.</p></div><span className={optimizerOnline ? "service-state online" : "service-state"}><i />{optimizerOnline ? "Vidur optimizer ready" : "Vidur optimizer offline"}</span></div>
        <div className="experiment-grid">
          <div className="panel">
            <div className="panel-title"><span className="panel-icon"><SlidersHorizontal size={17} /></span><div><h3>Workload and scheduler</h3><p>Small, repeatable synthetic workload</p></div></div>
            <div className="form-grid">
              <InputField label="Arrival rate" suffix="QPS" min={.1} value={form.qps} onChange={(v) => set("qps",v)} />
              <InputField label="Request count" suffix="requests" min={8} step="1" value={form.requests} onChange={(v) => set("requests",v)} />
              <InputField label="Prompt" suffix="tokens" min={32} step="1" value={form.prompt} onChange={(v) => set("prompt",v)} />
              <InputField label="Generated answer" suffix="tokens" min={8} step="1" value={form.output} onChange={(v) => set("output",v)} />
              <InputField label="Batch size cap" suffix="requests" min={8} step="1" value={form.batch} onChange={(v) => set("batch",v)} />
              <label className="control-field"><span>Scheduler</span><div className="select-wrap"><select value={form.scheduler} onChange={(e) => set("scheduler",e.target.value as FormState["scheduler"])}><option value="sarathi">Sarathi</option><option value="vllm">vLLM scheduler</option></select><ChevronDown size={14} /></div></label>
              {form.scheduler === "sarathi" && <><InputField label="Chunk size" suffix="tokens" min={64} step="1" value={form.chunk} onChange={(v) => set("chunk",v)} /><InputField label="Alternative chunk" suffix="tokens" min={64} step="1" value={form.alternate} onChange={(v) => set("alternate",v)} /></>}
            </div>
            <div className="profile-note"><Cpu size={15} /><span>This sweep uses Vidur’s A100 profile. GPU prices and performance remain assumptions/estimates, not a cloud quote.</span></div>
          </div>
          <div className="panel">
            <div className="panel-title"><span className="panel-icon green"><Gauge size={17} /></span><div><h3>Latency and efficiency goals</h3><p>Change the red target lines on the plots</p></div></div>
            <div className="form-grid">
              <InputField label="Maximum TTFT p90" suffix="seconds" min={.001} value={form.ttftSlo} onChange={(v) => set("ttftSlo",v)} />
              <InputField label="Maximum token time p99" suffix="seconds" min={.001} value={form.tpotSlo} onChange={(v) => set("tpotSlo",v)} />
              <div className="span-two"><InputField label="Assumed price per GPU" suffix="USD / hour" min={.01} value={form.cost} onChange={(v) => set("cost",v)} /></div>
            </div>
            <div className="target-summary"><span className="target-check"><Check size={13} /></span><div><b>SLO-compliant region</b><small>TTFT ≤ {ttftLimit.toFixed(2)} s <i>·</i> token time ≤ {tpotLimit.toFixed(2)} s</small></div></div>
            <button disabled={busy || !gatewayOnline} onClick={() => void optimize()} className="optimize-button">{busy ? <><RefreshCw size={16} className="spin" /> Simulating configs…</> : <><Play size={15} fill="currentColor" /> Run Vidur optimization <ArrowDownRight size={16} /> </>}</button>
            <p className="button-caption">Runs scenarios one at a time. It can take minutes and use CPU; the Vidur container is capped at two cores.</p>
            {notice && <p className="operation-notice">{notice}</p>}
            {!optimizerOnline && <div className="start-hint"><b>Start the simulation service</b><code>docker compose --profile vidur up -d --build vidur</code><small>Then wait for “Vidur optimizer ready” and press the run button.</small></div>}
          </div>
        </div>
      </section>

      <section id="frontier" className="content-section">
        <div className="section-intro"><div><span className="section-kicker">02 / READ THE TRADE-OFFS</span><h2>Every dot is a configuration.</h2><p>Green means both latency goals pass. Blue means at least one misses. The gold star is the passing point with the highest estimated QPS per assumed GPU-hour.</p></div><div className="legend"><span><i className="blue-dot" /> Tested</span><span><i className="green-dot" /> Meets SLO</span><span><i className="gold-star">★</i> Best</span></div></div>
        {runs.length > 0 ? <div className="chart-grid">
          <article className="chart-panel"><div className="chart-heading"><div><h3>Throughput value vs first token</h3><p>The green zone meets the TTFT p90 limit.</p></div><span>TTFT p90</span></div><Scatter runs={runs} xValue={(r) => ((r.latency.ttft_p90_ms ?? r.latency.ttft_p95_ms ?? 0)/1000)} yValue={efficiency} xLimit={ttftLimit} xLabel="Time to first token p90 (s)" yLabel="Offered QPS / assumed $·h" bestId={best?.run_id} cost={cost} /></article>
          <article className="chart-panel"><div className="chart-heading"><div><h3>Throughput value vs token time</h3><p>Lower tail token latency is better.</p></div><span>TPOT p99</span></div><Scatter runs={runs} xValue={(r) => ((r.latency.tpot_p99_ms ?? r.latency.tpot_p95_ms ?? 0)/1000)} yValue={efficiency} xLimit={tpotLimit} xLabel="Time between tokens p99 (s)" yLabel="Offered QPS / assumed $·h" bestId={best?.run_id} cost={cost} /></article>
          <article className="chart-panel chart-wide"><div className="chart-heading"><div><h3>Latency SLO map</h3><p>The shaded lower-left corner meets both limits. Color hints at the efficiency score.</p></div><span>TTFT vs TPOT</span></div><Scatter runs={runs} xValue={ttft} yValue={tpot} xLimit={ttftLimit} yLimit={tpotLimit} xLabel="Time to first token p90 (s)" yLabel="Time between tokens p99 (s)" bestId={best?.run_id} cost={cost} colorMap /></article>
        </div> : <div className="empty-state"><span><Gauge size={23} /></span><h3>Your Pareto plots will appear here</h3><p>Run an optimization above. The charts will use the per-request metrics Vidur saves for each candidate.</p><a href="#optimizer">Configure a sweep <ArrowDownRight size={14} /></a></div>}
      </section>

      <section id="latency" className="content-section">
        <div className="section-intro"><div><span className="section-kicker">03 / INSPECT THE TAIL</span><h2>Latency distribution, request by request.</h2><p>A cumulative distribution shows whether a config is consistently quick or hides slow outliers in its tail.</p></div>{runs.length > 0 && <label className="run-picker"><span>Run</span><select value={selected?.run_id || ""} onChange={(e) => setSelectedId(e.target.value)}>{runs.map((run) => <option key={run.run_id} value={run.run_id}>{run.run_id} · {run.replicas} replica(s)</option>)}</select><ChevronDown size={14} /></label>}</div>
        <div className="chart-panel cdf-panel"><div className="chart-heading"><div><h3>Normalized decode-time CDF</h3><p>{selected ? selected.model + " · " + selected.request_count + " requests · " + selected.gpu_profile + " profile" : "No request samples yet."}</p></div>{selected && <Pill>{selected.replicas} replicas · {selected.configuration?.scheduler || "Sarathi"}</Pill>}</div><Cdf run={selected} /><div className="cdf-legend"><span><i /> Per-request samples from Vidur output</span><span>100% = all simulated requests completed</span></div></div>
      </section>

      <section className="content-section">
        <div className="section-intro"><div><span className="section-kicker">04 / KEEP THE EVIDENCE</span><h2>Candidate runbook.</h2><p>Compare settings and percentile latency for every saved run.</p></div><button className="button-quiet refresh-runs" onClick={() => void refresh()}><RefreshCw size={14} /> Refresh</button></div>
        <div className="table-scroll"><table className="run-table"><thead><tr><th>Configuration</th><th>Workload</th><th>TTFT p90</th><th>TPOT p99</th><th>QPS / $·h</th><th>Result</th></tr></thead><tbody>{candidates.map((item) => <tr key={item.run.run_id}><td><b>{item.run.configuration?.scheduler || "Sarathi"} · {item.run.replicas} replica(s)</b><small>chunk {item.run.configuration?.chunk_size ?? "—"} · batch {item.run.configuration?.batch_size ?? "—"} · {item.run.gpu_profile}</small><small>{item.run.run_id}</small></td><td>{item.run.qps} QPS<small>{item.run.request_count} requests</small></td><td>{item.ttft.toFixed(3)} s</td><td>{item.tpot.toFixed(3)} s</td><td>{item.efficiency.toFixed(3)}</td><td>{item.compliant ? <span className="pass-label"><Check size={13} /> Pass</span> : <span className="fail-label">Outside SLO</span>}</td></tr>)}{!candidates.length && <tr><td className="no-rows" colSpan={6}>No saved profiles yet. Run your first sweep above.</td></tr>}</tbody></table></div>
        <p className="table-note">QPS / $·h is calculated from offered QPS ÷ replica count ÷ your cost assumption. It is not measured throughput or an actual cloud bill.</p>
      </section>

      <footer className="footer"><div className="footer-brand"><span className="logo-mark"><Zap size={14} fill="currentColor" /></span><b>HYPERION</b><span>Inference optimization lab</span></div><p>Vidur is an offline GPU-profile simulator, not a physical GPU.</p></footer>
    </div>
  </main>;
}
