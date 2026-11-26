"use client";
import {useEffect, useState} from "react";
type State = {status:string; phase?:string; qps?:number; active_replicas?:number; error?:string; events:{time:number;kind:string;replicas?:number;reason?:string;latency?:{ttft_p95_ms:number};message?:string}[]};
export default function Autonomy(){
 const [state,setState]=useState<State>({status:"offline",events:[]});
 const [error,setError]=useState("");
 const [traffic,setTraffic]=useState("4,8,4");
 const url="http://localhost:8101";
 async function refresh(){try{const r=await fetch(url+"/autonomy");if(!r.ok)throw Error("Service unavailable");setState(await r.json());}catch{setState(s=>({...s,status:"offline"}));}}
 useEffect(()=>{void refresh();const t=setInterval(()=>void refresh(),2000);return()=>clearInterval(t);},[]);
 async function action(start:boolean){setError("");try{const r=await fetch(url+"/autonomy/"+(start?"start":"stop"),{method:"POST",headers:{"Content-Type":"application/json"},body:start?JSON.stringify({traffic:traffic.split(",").map(Number)}):undefined});const data=await r.json();if(!r.ok)throw Error(JSON.stringify(data.detail));await refresh();}catch(e){setError(String(e));}}
 const observations=state.events.filter(e=>e.latency);
 const max=Math.max(600,...observations.map(e=>e.latency!.ttft_p95_ms));
 return <section className="content-section" id="autonomy"><div className="section-intro"><div><span className="section-kicker">AUTONOMOUS CONTROL / VIDUR</span><h2>Observe. Experiment. Apply. Validate.</h2><p>Independent simulation episodes, with persistent workload-specific evidence. The controller minimizes target violations, then replica count. Targets: TTFT p95 600 ms; E2E p95 10 seconds.</p></div></div>
 <div className="panel"><label>Traffic schedule (QPS, comma separated) <input value={traffic} onChange={e=>setTraffic(e.target.value)} disabled={state.status==="running"}/></label>
 <button className="button-primary" disabled={state.status==="running"||state.status==="offline"} onClick={()=>void action(true)}>Start autonomous session</button>
 <button className="button-quiet" disabled={state.status!=="running"} onClick={()=>void action(false)}>Stop after current simulation</button>
 <p>Status: {state.status} · Phase: {state.phase||"—"} · Workload: {state.qps??"—"} QPS · Active simulation replicas: {state.active_replicas??"—"}</p>
 <p>Each traffic period triggers observation, candidate experiments, and validation. Each is a fresh Vidur run: queues and KV state do not carry between periods. Starting can use two CPU cores for several minutes per run.</p>
 {error&&<p role="alert">{error}</p>}{state.error&&<p role="alert">{state.error}</p>}
 <svg viewBox="0 0 800 180" className="plot" aria-label="Observed TTFT p95 by experiment"><line x1="20" x2="780" y1={160-600/max*140} y2={160-600/max*140} stroke="red"/><polyline fill="none" stroke="#16804a" strokeWidth="3" points={observations.map((e,i)=>(20+i*760/Math.max(1,observations.length-1))+","+(160-e.latency!.ttft_p95_ms/max*140)).join(" ")}/><text x="20" y="178">Experiment order → · red line = 600 ms TTFT target</text></svg>
 <div style={{maxHeight:360,overflow:"auto"}}>{[...state.events].reverse().map((e,i)=><p key={i}><b>{e.kind}</b> {e.replicas!==undefined?e.replicas+" replicas · ":""}{e.reason||e.message||""}{e.latency?" TTFT p95 "+e.latency.ttft_p95_ms.toFixed(1)+" ms":""}</p>)}</div></div></section>;
}
