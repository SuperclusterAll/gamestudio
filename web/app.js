// Every lookup goes through a null-safe stub: if a cached copy of index.html is missing a panel the
// newer script writes to, one throw would abort render() and freeze the whole dashboard.
const MISSING = { innerHTML: "", textContent: "", className: "", href: "", hidden: true, style: {} };
const $ = (id) => document.getElementById(id) || MISSING;
$("code-model-id").innerHTML = $("model-id").innerHTML;
fetch('/api/model-status').then(r=>r.json()).then(s=>{
  const bedrock = s.configured ? "Bedrock 인증 설정 확인됨 · 모델 접근 권한은 실행 시 확인합니다." : "Bedrock 인증 미설정 · AWS 프로필 또는 프로젝트 .env 연결이 필요합니다.";
  // Image generation used to be an unchecked box with no hint about whether it could work, so runs
  // quietly shipped Canvas-only games while the art agent was planning a sprite for every object.
  const images = $("images");
  if (images.tagName) { images.checked = Boolean(s.comfyui_available); images.disabled = !s.comfyui_available; }
  const comfy = s.comfyui_available
    ? `이미지 생성 사용 가능 (ComfyUI ${s.comfyui_server}) · 객체별 스프라이트 생성이 켜졌습니다.`
    : `ComfyUI 응답 없음 (${s.comfyui_server}) · 캔버스 전용으로 제작합니다.`;
  // A Godot run cannot be verified - or built at all - without the engine on this machine, so the
  // option says so up front instead of failing several minutes into a run.
  const engine = $("engine");
  if (engine.tagName) {
    const godot = [...engine.options].find(o => o.value === "godot");
    if (godot && !s.godot_available) { godot.disabled = true; godot.textContent += " · 엔진 없음"; }
  }
  const godotNote = s.godot_available
    ? `Godot ${s.godot_version || "설치됨"} 감지 · Godot 모드 사용 가능.`
    : "Godot 실행 파일 없음 · HTML5 모드만 사용할 수 있습니다.";
  $("model-status").textContent = `${bedrock} ${comfy} ${godotNote}`;
}).catch(()=>{$("model-status").textContent="모델 연결 상태를 확인할 수 없습니다.";});
// The supervisor is the hub every stage reports back to, so it heads the trace and its row is the
// one that shows a run's escalations as repeat visits (×N).
const steps = ["supervisor","idea","design_document","approval","art","code","qa","repair","package"];
// Steps whose model call streams, so the raw-output panel can show one while it is still writing.
const LIVE_STEPS = ["code","repair","qa","supervisor","design_document","idea"];
let runs = new Map(), selectedId = null;

function esc(value="") { return String(value).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])); }
function elapsedSeconds(iso) { return iso ? Math.max(0, Math.floor((Date.now() - new Date(iso).getTime())/1000)) : null; }
// Built by hand rather than via toLocaleTimeString, whose output varies by engine and locale data
// (some builds append the timezone name, which is noise in a dense list).
function clockTime(iso) {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return "";
  const pad = (n) => String(n).padStart(2, "0");
  return `${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`;
}
// render() runs on every stream chunk and once a second. Rewriting innerHTML that has not changed
// destroys live DOM state (an open <details>, a text selection, scroll position) many times per
// second, so only assign when the markup is actually different.
function setHTML(el, html) { if (el && el.innerHTML !== html) el.innerHTML = html; }

// The design document and the graph live on their own tabs so streaming updates to the monitor
// never disturb them.
const VIEWS = ["monitor", "graph", "doc"];
document.querySelectorAll(".tab").forEach(tab => {
  tab.onclick = () => {
    document.querySelectorAll(".tab").forEach(t => t.classList.toggle("active", t === tab));
    VIEWS.forEach(v => { $(`view-${v}`).hidden = tab.dataset.view !== v; });
    render();
  };
});

// The topology comes from the compiled graph itself (/api/graph), so the picture cannot drift from
// what actually runs. Mermaid is vendored under /static/vendor rather than loaded from a CDN.
let graphSource = null;
let graphSignature = "";
if (window.mermaid) {
  mermaid.initialize({ startOnLoad: false, theme: "dark", flowchart: { curve: "basis" },
                       securityLevel: "strict" });
}
fetch("/api/graph").then(r => r.json()).then(d => { graphSource = d.mermaid; render(); })
  .catch(() => { $("graph-view").textContent = "그래프 정의를 불러올 수 없습니다."; });

async function renderGraph(run) {
  if (!graphSource || !window.mermaid || $("view-graph").hidden) return;
  const states = run ? stepStates(run) : {};
  // Re-render only when the colouring actually changes: mermaid rebuilds the whole SVG each time,
  // and the monitor updates many times a second while a model is streaming.
  const signature = JSON.stringify(states);
  if (signature === graphSignature) return;
  graphSignature = signature;
  const classes = Object.entries(states).map(([node, state]) => `  class ${node} ${state};`);
  const definition = [
    graphSource.replace(/^\s*class\s+\w+\s+(first|last);?\s*$/gm, ""),
    "  classDef done fill:#123f42,stroke:#73e6d2,stroke-width:2px,color:#dffef8;",
    "  classDef active fill:#4e3b11,stroke:#ffdf75,stroke-width:3px,color:#ffebad;",
    "  classDef error fill:#4a1f2c,stroke:#ff7a91,stroke-width:2px,color:#ffb4c2;",
    "  classDef pending fill:#0b1b2d,stroke:#294b67,color:#6f8ca3;",
    ...classes,
  ].join("\n");
  try {
    const { svg } = await mermaid.render(`pipeline-${Date.now()}`, definition);
    $("graph-view").innerHTML = svg;
  } catch (err) {
    $("graph-view").textContent = `그래프를 그릴 수 없습니다: ${err && err.message ? err.message : err}`;
  }
}
function label(step) { return ({supervisor:"총괄 감독",idea:"기획 Agent",design_document:"기획 문서",approval:"사람 승인",art:"아트 기획",code:"코드 Agent",qa:"QA 검증",repair:"자동 수정",package:"게임 패키징",abandoned:"QA 미통과 종료",orchestrator:"실행 준비","hitl-design-approval":"사람 승인",rejected:"승인 거부",complete:"완료",failed:"실패"}[step] || step); }
// Renders one entry of the structured "who is doing what" log the backend streams live: which
// model call started, which tool an agent decided to call (with what arguments), what that tool
// returned, or what the model finally answered — so it's never a mystery which agent/model/tool
// is currently running.
function agentLogLine(e) {
  const agent = esc(e.agent || "Agent");
  if (e.kind === "model_call") {
    const model = e.model ? ` <span class="log-model">${esc(e.model)}</span>` : "";
    const note = e.note ? ` — ${esc(e.note)}` : "";
    return `<li class="log-call"><b>${agent}</b> 모델 호출${model}${note}</li>`;
  }
  if (e.kind === "tool_call") {
    const args = Object.entries(e.args || {}).map(([k, v]) => `${esc(k)}=${esc(String(v))}`).join(", ");
    return `<li class="log-tool"><b>${agent}</b> 도구 호출 <code>${esc(e.name)}</code>(${args})</li>`;
  }
  if (e.kind === "tool_result") {
    return `<li class="log-result"><b>${agent}</b> 도구 결과 <code>${esc(e.name)}</code> — ${esc(e.text)}</li>`;
  }
  if (e.kind === "model_text") {
    return `<li class="log-text"><b>${agent}</b> — ${esc(e.text)}</li>`;
  }
  return `<li>${esc(JSON.stringify(e))}</li>`;
}
// A LangSmith-style trace: one row per step with its real state and a bar sized by how long it
// actually took. Derived from the event log, not from a position in the step list — the old index
// approach painted every earlier step green, so 자동 수정 looked like it had run on a clean build
// when it never executed at all.
// How long each step took, how many times it ran, and what it spent. Derived from the event log.
function stepSpans(run) {
  const events = run.events || [];
  const spans = new Map();
  events.forEach((e, i) => {
    if (!steps.includes(e.step)) return;
    const prev = i > 0 ? new Date(events[i - 1].at).getTime() : new Date(e.at).getTime();
    const took = Math.max(0, new Date(e.at).getTime() - prev) / 1000;
    const span = spans.get(e.step) || { seconds: 0, runs: 0, tokens: 0 };
    span.seconds += took;
    span.runs += 1;
    span.tokens = (run.usage_by_step || {})[e.step]?.total_tokens || 0;
    spans.set(e.step, span);
  });
  return spans;
}

// One source of truth for node state, shared by the trace list and the graph diagram so the two
// views can never disagree about what ran.
function stepStates(run) {
  const spans = stepSpans(run);
  const running = run.status === "running" || run.status === "waiting_approval";
  const current = run.current_step === "hitl-design-approval" ? "approval" : run.current_step;
  const failedAt = run.status === "failed" ? current : null;
  const states = {};
  steps.forEach(step => {
    if (step === failedAt) states[step] = "error";
    else if (running && step === current) states[step] = "active";
    else if (spans.has(step)) states[step] = "done";
    else states[step] = "pending";
  });
  return states;
}

function traceRows(run) {
  const spans = stepSpans(run);
  const states = stepStates(run);
  const slowest = Math.max(0.001, ...[...spans.values()].map(s => s.seconds));

  return steps.map(step => {
    const span = spans.get(step);
    const state = states[step];
    const width = span ? Math.max(2, (span.seconds / slowest) * 100) : 0;
    const spent = span
      ? `${span.seconds.toFixed(1)}s${span.runs > 1 ? ` ×${span.runs}` : ""}${span.tokens ? ` · ${span.tokens.toLocaleString()} tok` : ""}`
      : "";
    const meta = state === "active" ? "실행 중"
      : state === "error" ? (spent ? `실패 · ${spent}` : "실패")
      : span ? spent
      : "미실행";
    return `<div class="trace ${state}"><span class="trace-dot"></span><span class="trace-name">${label(step)}</span>` +
      `<span class="trace-bar"><i style="width:${width}%"></i></span><span class="trace-meta">${meta}</span></div>`;
  }).join("");
}

function formatUsage(usage) {
  if (!usage || !usage.total_tokens) return "";
  const cost = usage.cost_usd ? `$${usage.cost_usd.toFixed(4)}` : "$0";
  const caveat = usage.priced === false ? " (일부 모델 단가 미등록)" : "";
  return `${usage.total_tokens.toLocaleString()} tok (in ${usage.input_tokens.toLocaleString()} / out ${usage.output_tokens.toLocaleString()}) · 예상 ${cost}${caveat}`;
}

function save(run) { runs.set(run.id, run); if (!selectedId) selectedId = run.id; render(); }
function render() {
  setHTML($("runs"), [...runs.values()].reverse().map(r => `<button class="run-card ${r.id===selectedId?'active':''}" data-run="${r.id}">${esc(r.genre || '게임')} · ${esc(r.brief.slice(0,36))}<small>${esc(r.model_id || 'Bedrock')} · ${esc(r.status)} · ${esc(r.id)}</small></button>`).join("") || '<p class="empty">새 실행을 시작하세요.</p>');
  document.querySelectorAll("[data-run]").forEach(el => el.onclick = () => { selectedId = el.dataset.run; render(); });
  const run = runs.get(selectedId); if (!run) return;
  $("run-title").textContent = run.state?.design_document?.title || run.brief;
  $("status").className = `badge ${run.status}`; $("status").textContent = ({queued:"대기",running:"실행 중",waiting_approval:"승인 대기",completed:"완료",rejected:"중단",qa_failed:"QA 미통과",failed:"실패"}[run.status] || run.status);
  setHTML($("pipeline"), traceRows(run));
  const d = run.state?.design_document;
  if (d) {
    // Monitor tab keeps only the pitch, so the reviewer can just hit a button. The full contract
    // lives on the 기획서 tab, which streaming never rewrites.
    const p = d.implementation_plan;
    const criteria = p ? [...p.mechanics, p.win_condition, p.loss_condition, ...p.acceptance_tests] : [];
    setHTML($("design-document"),
      `<p class="doc-pitch">${esc(d.summary)}</p>` +
      `<p class="muted">전체 내용은 위 "기획서" 탭에서 볼 수 있습니다${criteria.length ? ` · 검증 기준 ${criteria.length}개` : ""}</p>`);
    setHTML($("design-document-full"),
      `<h3 class="doc-title">${esc(d.title)}</h3><p class="doc-pitch">${esc(d.summary)}</p>` +
      `<div class="doc-grid">${(d.reference_games||[]).length ? `<strong>참조 게임</strong><span>${(d.reference_games||[]).map(esc).join("<br>")}</span>` : ""}<strong>플레이 목표</strong><span>${esc(d.player_goal)}</span><strong>조작</strong><span>${(d.controls||[]).map(esc).join(" · ")}</span><strong>핵심 루프</strong><span>${(d.core_loop||[]).map(esc).join(" → ")}</span><strong>난이도</strong><span>${esc(d.difficulty_curve)}</span><strong>비주얼</strong><span>${esc(d.visual_direction)}</span><strong>제작 범위</strong><span>${esc(d.delivery_scope)}</span></div>` +
      (p ? `<h4>구현·검증 기준 (${esc(p.genre)})</h4><ul>${criteria.map(v=>`<li>${esc(v)}</li>`).join('')}</ul>` +
           `<h4>상태 전이</h4><ul>${(p.state_transitions||[]).map(v=>`<li>${esc(v)}</li>`).join('')}</ul>` : ""));
  } else {
    const streamStep = run.stream?.design_document ? "design_document" : "idea";
    const stepLabel = streamStep === "design_document" ? "기획서 구조화" : "컨셉 기획";
    const secs = elapsedSeconds(run.stream_started_at?.[streamStep]);
    const waitHint = run.status !== "running" ? "" : secs === null ? " · 응답 대기 중" : ` · ${secs}초 경과`;
    setHTML($("design-document"), `<span class="empty">기획 Agent(${esc(stepLabel)})가 문서를 만드는 중입니다${waitHint}</span>`);
  }
  $("approval").hidden = run.status !== "waiting_approval";
  const qa = run.state?.qa, result = $("results");
  const log = run.agent_log || [];
  const last = log[log.length - 1];
  const lastSummary = last ? `${esc(last.agent || "Agent")}: ${esc(last.name || last.text || last.note || last.model || "")}` : "아직 활동 없음";
  const retries = run.state?.rethink_cycles;
  setHTML(result, `<dl><dt>모델</dt><dd>${esc(run.model_id || "Bedrock")}</dd><dt>현재 단계</dt><dd>${esc(label(run.current_step))}</dd><dt>승인</dt><dd>${esc(run.state?.approval?.decision || "대기")}</dd><dt>QA</dt><dd>${esc(qa?.status || "대기")}${retries ? ` · 총괄 감독 재검토 ${retries}회` : ""}${qa?.findings?.length ? `<br>${qa.findings.map(esc).join("<br>")}` : ""}</dd><dt>엔진</dt><dd>${esc(run.engine === "godot" ? "Godot 4" : "HTML5 Canvas")}</dd><dt>산출물</dt><dd>${esc(run.state?.godot_project_path || run.state?.game_path || run.state?.qa_report_path || "아직 없음")}${run.state?.godot_project_path && !run.state?.game_path ? '<br><span class="muted">웹 빌드 없음 · Godot에서 프로젝트를 열어 실행하세요.</span>' : ""}</dd><dt>최근 활동</dt><dd>${lastSummary}</dd>${run.error?`<dt>오류</dt><dd>${esc(run.error)}</dd>`:""}</dl>`
    + `<p class="muted">코딩·검증 모델: ${esc(run.state?.code_model_id || run.model_id)}</p>`);
  // A QA-failed run publishes its draft for review, so it is playable - just labelled as such.
  const finished = ["completed", "qa_failed"].includes(run.status);
  const play = $("play-game");
  // A Godot run has an embeddable build only when export templates were available; without one
  // this link would open a 404, so it is the presence of the artifact that decides, not the status.
  play.hidden = !finished || !run.state?.game_path;
  play.href = `/games/${run.id}`;
  play.textContent = run.status === "qa_failed" ? "게임 실행 (QA 미통과 · 검토용)" : "완성된 게임 실행";
  play.className = run.status === "qa_failed" ? "play-game warn" : "play-game";
  // A browser cannot execute a .bat, so this button asks the local server to start the game.
  const launch = $("launch-godot"), launchNote = $("launch-note");
  const launchable = finished && run.state?.launch_script_path;
  launch.hidden = !launchable;
  if (launchable) launch.dataset.run = run.id;
  // A Godot run with no launcher is the one case where the button's absence needs explaining:
  // the folder predates run.bat, or the run never reached packaging. Silence there reads as a
  // broken button.
  else if (finished && run.engine === "godot") {
    launchNote.hidden = false;
    launchNote.textContent = run.state?.godot_project_path
      ? "이 런에는 run.bat이 없습니다 (해당 기능 이전에 만들어진 프로젝트). Godot에서 직접 열어 실행하세요."
      : "이 런은 패키징까지 도달하지 못해 실행할 프로젝트가 없습니다.";
  } else if (!launchable) launchNote.hidden = true;

  // The one judgement no check in this pipeline can make, and the service's headline metric. Only
  // offered once there is a finished game to judge - asking before then measures an opinion about
  // nothing. A decision already on disk comes back with the run, so a restart does not ask twice.
  const adoption = run.state?.adoption;
  $("adoption").hidden = !finished;
  if (finished) {
    $("adopt-yes").setAttribute("aria-pressed", String(adoption?.adopted === true));
    $("adopt-no").setAttribute("aria-pressed", String(adoption?.adopted === false));
    $("adopt-state").textContent = adoption
      ? `${adoption.adopted ? "채택함" : "보류함"} · ${clockTime(adoption.decided_at)}`
      : "아직 판단하지 않음";
    refreshAdoptionRate();
  }

  // Newest first for readability, but numbered in the order the steps actually ran, with the wall
  // clock and how long each step took so a slow stage is obvious at a glance.
  const events = run.events || [];
  setHTML($("events"), events.map((e, i) => {
    const took = i > 0 ? (new Date(e.at).getTime() - new Date(events[i-1].at).getTime()) / 1000 : null;
    // Cumulative spend at the moment this step finished, so the history doubles as a cost trail.
    const tok = e.total_tokens ? `${e.total_tokens.toLocaleString()} tok` : "";
    const cost = e.cost_usd ? `$${e.cost_usd.toFixed(4)}` : "";
    const spend = tok ? ` <span class="evt-spend">${tok}${cost ? ` · ${cost}` : ""}</span>` : "";
    return `<li><span class="evt-n">${i + 1}</span><span class="evt-body"><b>${esc(label(e.step))}</b> ${esc(e.message)}</span><span class="evt-time">${clockTime(e.at)}${took !== null ? ` <em>+${took.toFixed(1)}s</em>` : ""}${spend}</span></li>`;
  }).reverse().join("") || '<li class="empty">대기 중</li>');
  setHTML($("usage-total"), formatUsage(run.usage));
  setHTML($("agent-log"), log.length ? log.slice().reverse().map(agentLogLine).join("") : '<li class="empty">아직 활동이 없습니다.</li>');
  renderGraph(run);

  // Raw output of whichever model call is running right now. The code and repair agents stream, so
  // this fills in character by character while a long generation is still in flight.
  const liveStep = LIVE_STEPS.find(s => run.stream?.[s] && run.current_step === s)
    || LIVE_STEPS.find(s => run.stream?.[s]);
  const liveText = liveStep ? run.stream[liveStep] : "";
  const streaming = Boolean(liveText) && run.status === "running";
  $("live-stream").hidden = !liveText;
  $("live-stream-text").textContent = liveText || "";
  const secs = liveStep ? elapsedSeconds(run.stream_started_at?.[liveStep]) : null;
  $("log-now").textContent = run.status !== "running"
    ? ""
    : `${label(run.current_step)} 진행 중${secs !== null && streaming ? ` · ${secs}초 경과 · ${liveText.length}자 생성` : " · 응답 대기 중"}`;
}
async function decision(choice) { const run = runs.get(selectedId); if (!run) return; const res = await fetch(`/api/runs/${run.id}/decision`, {method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({decision:choice,comment:$("comment").value})}); if (!res.ok) alert(await res.text()); else save(await res.json()); }
$("approve").onclick = () => decision("approve"); $("reject").onclick = () => decision("reject");
$("launch-godot").onclick = async (e) => {
  const button = e.currentTarget, note = $("launch-note");
  note.hidden = false;
  // A click that does nothing tells the user nothing. Every path out of here says something,
  // including the ones that should be impossible.
  const runId = button.dataset.run;
  if (!runId) {
    note.textContent = "실행할 런이 선택되지 않았습니다. 왼쪽에서 완료된 Godot 런을 고르세요.";
    return;
  }
  button.disabled = true;
  note.textContent = "Godot을 실행하는 중…";
  try {
    const res = await fetch(`/api/runs/${runId}/launch`, { method: "POST" });
    note.textContent = res.ok
      ? "이 PC에서 Godot으로 게임을 실행했습니다. 별도 창을 확인하세요."
      : `실행하지 못했습니다: ${(await res.json().catch(() => ({}))).detail || res.status}`;
  } catch (error) {
    note.textContent = `실행하지 못했습니다: ${error}`;
  } finally {
    button.disabled = false;
  }
};

// Counted off the manifests rather than off the run list, because the rate has to survive a
// restart to mean anything - the same reason the decision is written into the manifest at all.
async function refreshAdoptionRate() {
  try {
    const stats = await (await fetch("/api/adoption")).json();
    $("adopt-rate").textContent = stats.decided
      ? `채택률 ${(stats.rate * 100).toFixed(0)}% — 판단한 ${stats.decided}건 중 ${stats.adopted}건 채택`
        + (stats.undecided ? ` · 미판단 ${stats.undecided}건` : "")
      : `아직 판단한 게임이 없습니다 (완료 ${stats.finished}건).`;
  } catch { $("adopt-rate").textContent = ""; }
}

async function adopt(adopted) {
  const run = runs.get(selectedId);
  if (!run) return;
  const buttons = [$("adopt-yes"), $("adopt-no")];
  buttons.forEach(b => b.disabled = true);
  try {
    const res = await fetch(`/api/runs/${run.id}/adopt`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ adopted }),
    });
    if (!res.ok) {
      $("adopt-state").textContent =
        `기록하지 못했습니다: ${(await res.json().catch(() => ({}))).detail || res.status}`;
      return;
    }
    // Kept locally too: the websocket only carries a run this process is actually tracking, and a
    // restored run is not one of those until something else updates it.
    run.state = { ...(run.state || {}), adoption: await res.json() };
    save(run);
  } catch (error) {
    $("adopt-state").textContent = `기록하지 못했습니다: ${error}`;
  } finally {
    buttons.forEach(b => b.disabled = false);
  }
}
$("adopt-yes").onclick = () => adopt(true);
$("adopt-no").onclick = () => adopt(false);

$("run-form").onsubmit = async (e) => { e.preventDefault(); const res = await fetch("/api/runs", {method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({genre:$("genre").value,brief:$("brief").value,engine:$("engine").value,model_id:$("model-id").value,code_model_id:$("code-model-id").value,generate_images:$("images").checked})}); if (!res.ok) return alert(await res.text()); const run = await res.json(); selectedId=run.id; save(run); };
async function initial() { const res = await fetch("/api/runs"); (await res.json()).runs.forEach(save); }
function socket() { const ws = new WebSocket(`${location.protocol==='https:'?'wss':'ws'}://${location.host}/ws`); ws.onopen=()=>$("connection").textContent="Live connected"; ws.onmessage=e=>{const x=JSON.parse(e.data); if(x.type==="run:update")save(x.run); if(x.type==="runs:initial")x.runs.forEach(save)}; ws.onclose=()=>{ $("connection").textContent="Reconnecting…"; setTimeout(socket,1000); }; }
initial();socket();
// Ticks the elapsed-time label even when no new stream chunk has arrived yet, so a genuinely
// stalled agent call is visibly distinguishable from one that is simply still generating.
setInterval(() => { if (runs.get(selectedId)?.status === "running") render(); }, 1000);
