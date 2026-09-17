"""Local real-time dashboard for reviewable game-generation runs."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import time
import uuid
import re
import boto3
import requests
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command
from pydantic import BaseModel, Field, field_validator

from game_studio.agents import DEFAULT_MODEL_ID
from game_studio.godot import LAUNCH_SCRIPT
from game_studio.graph import build_graph

BEDROCK_MODELS = {
    "global.anthropic.claude-sonnet-4-6": "Claude Sonnet 4.6 (권장 · Global)",
    "us.anthropic.claude-sonnet-4-6": "Claude Sonnet 4.6 (US)",
    "global.anthropic.claude-sonnet-4-5-20250929-v1:0": "Claude Sonnet 4.5 (Global)",
    "us.anthropic.claude-haiku-4-5-20251001-v1:0": "Claude Haiku 4.5 (US · 빠름)",
    "global.anthropic.claude-haiku-4-5-20251001-v1:0": "Claude Haiku 4.5 (Global · 빠름)",
    "us.amazon.nova-pro-v1:0": "Amazon Nova Pro (US)",
    "us.amazon.nova-2-lite-v1:0": "Amazon Nova 2 Lite (US · 경제적)",
    "global.amazon.nova-2-lite-v1:0": "Amazon Nova 2 Lite (Global · 경제적)",
    "us.amazon.nova-lite-v1:0": "Amazon Nova Lite (US · 경제적)",
}


def bedrock_credentials_configured() -> bool:
    """Include shared AWS profiles and role/SSO providers, without exposing credentials."""
    try:
        return boto3.Session().get_credentials() is not None
    except Exception:
        return False


def comfyui_available() -> bool:
    """Whether the local image backend is actually up.

    Image generation used to be an unchecked box with no indication of whether it could even work,
    so runs silently produced Canvas-only games while the art agent was planning sprites for every
    object. The dashboard now probes this and turns the option on by itself when ComfyUI answers.
    """
    server = os.getenv("COMFYUI_SERVER", "http://127.0.0.1:8188").rstrip("/")
    try:
        return requests.get(f"{server}/system_stats", timeout=2).ok
    except Exception:
        return False

if getattr(sys, "frozen", False):
    PROJECT_ROOT = Path(sys.executable).resolve().parent.parent
    WEB_ROOT = Path(sys._MEIPASS) / "web"  # type: ignore[attr-defined]
else:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    WEB_ROOT = PROJECT_ROOT / "web"

GAME_OUTPUT_ROOT = Path(os.getenv("GAME_OUTPUT_DIR", r"C:\dev\games")).resolve()


# What a run can be built with. The label is what the dashboard shows in its dropdown.
GAME_ENGINES = {
    "html5": "HTML5 Canvas (브라우저에서 바로 실행)",
    "godot": "Godot 4 프로젝트 (엔진 검증 포함)",
}


class CreateRun(BaseModel):
    genre: str = Field(default="auto", max_length=80)
    brief: str = Field(default="", max_length=1000)
    offline: bool = False
    engine: str = "html5"
    generate_images: bool = False
    model_id: str = "global.anthropic.claude-sonnet-4-6"
    code_model_id: str = "global.anthropic.claude-sonnet-4-6"

    @field_validator("model_id", "code_model_id")
    @classmethod
    def validate_model_id(cls, value: str) -> str:
        if value not in BEDROCK_MODELS:
            raise ValueError("Only the configured Amazon Bedrock models are allowed.")
        return value

    @field_validator("engine")
    @classmethod
    def validate_engine(cls, value: str) -> str:
        if value not in GAME_ENGINES:
            raise ValueError(f"Unknown engine: {value}")
        return value


class ReviewDecision(BaseModel):
    decision: str = Field(pattern="^(approve|reject)$")
    comment: str = Field(default="", max_length=1000)


def _record_usage(run: "Run") -> None:
    """Write what the run actually consumed into its manifest, once it is over.

    Only this process ever sees these totals: the usage callback reports each model call onto the
    run's stream and the server adds them up, so the graph node that writes the manifest has no
    access to them. They then lived in an in-memory dict and died with the process.

    Which made every question about consumption unanswerable after the fact. "Is 50 calls the right
    budget", "does Godot really cost more than the Canvas path", "did that model change help" - all
    of them need runs to compare, and there were none to compare because nothing was kept. The
    manifest is where the rest of the run's record already lives.

    Best effort. A run that produced a game is finished whether or not its accounting gets written,
    and this is the last thing that happens to it.
    """
    usage = dict(run.usage or {})
    if not usage.get("calls"):
        return
    manifest_path = (run_folder(run.id) / "production-manifest.json").resolve()
    if not manifest_path.is_relative_to(GAME_OUTPUT_ROOT) or not manifest_path.is_file():
        return
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["usage"] = {**usage, "by_step": dict(run.usage_by_step or {}),
                             "status": run.status, "engine": run.engine,
                             "recorded_at": datetime.now(UTC).isoformat()}
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False),
                                 encoding="utf-8")
    except (OSError, ValueError):
        return


def _stage_draft_for_revision(workspace: Path, engine: str) -> None:
    """Put the shipped game where the code agent's tools expect to find a work in progress.

    A Godot project needs nothing: read_godot_file and write_godot_file address the project
    directory itself, which is where the game already is. The Canvas path writes and reads
    draft.html and publishes index.html, so without this the agent opens a revision by finding no
    draft at all and writing a new game from the contract - which is exactly what a revision is
    not.
    """
    if engine == "godot":
        return
    published, draft = workspace / "index.html", workspace / "draft.html"
    if published.is_file():
        shutil.copy2(published, draft)


class RevisionRequest(BaseModel):
    """What the player wants changed, having actually played the finished game.

    The one kind of feedback this pipeline cannot generate for itself. Static QA proves the game
    runs and the design review proves it matches its contract; neither has played it, and "점프가
    너무 무겁다" is not a thing either of them can notice.
    """

    request: str = Field(min_length=2, max_length=1000)


class AdoptionMark(BaseModel):
    """Whether a finished game was worth carrying on with.

    The one judgement no check in this pipeline can make. Static QA proves a game runs and the
    design review proves it matches its contract; neither says whether anyone wants to build it,
    and that is the number this whole studio exists to move. It was defined as a target and then
    never recorded, which meant the service's headline metric was the only one running on memory.
    """

    adopted: bool
    note: str = Field(default="", max_length=500)


class Connections:
    def __init__(self) -> None:
        self.clients: set[WebSocket] = set()

    async def connect(self, client: WebSocket) -> None:
        await client.accept()
        self.clients.add(client)

    def disconnect(self, client: WebSocket) -> None:
        self.clients.discard(client)

    async def broadcast(self, message: dict[str, Any]) -> None:
        for client in list(self.clients):
            try:
                await client.send_json(message)
            except Exception:
                self.disconnect(client)


@dataclass
class Run:
    id: str
    genre: str
    brief: str
    model_id: str
    config: dict[str, Any]
    engine: str = "html5"
    # Rebuilt from a manifest on startup rather than driven in this process. Tracked so a later
    # restore can replace its own entries without touching a run that is actually executing.
    restored: bool = False
    status: str = "queued"
    current_step: str = "queued"
    state: dict[str, Any] = field(default_factory=dict)
    events: list[dict[str, str]] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    error: str | None = None
    # Live text the idea/design-document agents are generating right now, keyed by step, plus when
    # each one started streaming. Lets the dashboard prove the agent is actively working instead of
    # leaving the viewer guessing whether a long wait means "still planning" or "stuck".
    stream: dict[str, str] = field(default_factory=dict)
    stream_started_at: dict[str, str] = field(default_factory=dict)
    _last_stream_publish: dict[str, float] = field(default_factory=dict, repr=False, compare=False)
    # Structured, chronological "who is doing what" feed: one entry per model call, tool call, tool
    # result, or model text answer, across every agent in the pipeline (supervisor, idea, design
    # document, art, code agent with its image generation, QA, repair). This is what answers
    # "어떤 모델/에이전트가 지금 무슨 도구를 쓰고 있는지" instead of guessing from a static step label.
    agent_log: list[dict[str, Any]] = field(default_factory=list)
    # Token spend for the whole run and per pipeline step, reported by the usage callback attached
    # to the model itself. cost_usd is an estimate from a static price table, not billing data.
    usage: dict[str, Any] = field(default_factory=lambda: {
        "input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "cost_usd": 0.0, "calls": 0,
        "cache_read_tokens": 0, "priced": True,
    })
    usage_by_step: dict[str, dict[str, Any]] = field(default_factory=dict)

    def event(self, step: str, message: str) -> None:
        self.current_step = step
        self.events.append({
            "at": datetime.now(UTC).isoformat(), "step": step, "message": message,
            # Snapshot the running total so each history row can show the spend at that point.
            "total_tokens": self.usage["total_tokens"], "cost_usd": self.usage["cost_usd"],
            "step_tokens": self.usage_by_step.get(step, {}).get("total_tokens", 0),
        })

    def add_usage(self, step: str, payload: dict[str, Any]) -> None:
        prompt_tokens = int(payload.get("input_tokens") or 0)
        output_tokens = int(payload.get("output_tokens") or 0)
        # Already counted inside input_tokens; tracked separately so the dashboard can show whether
        # prompt caching is actually engaging rather than only its effect on the estimate.
        cached = int(payload.get("cache_read_tokens") or 0)
        cost = payload.get("cost_usd")
        for bucket in (self.usage, self.usage_by_step.setdefault(step, {
            "input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "cost_usd": 0.0, "calls": 0,
            "cache_read_tokens": 0,
        })):
            bucket["input_tokens"] += prompt_tokens
            bucket["output_tokens"] += output_tokens
            bucket["total_tokens"] += prompt_tokens + output_tokens
            bucket["cache_read_tokens"] = bucket.get("cache_read_tokens", 0) + cached
            bucket["calls"] += 1
            bucket["cost_usd"] = round(bucket["cost_usd"] + (cost or 0.0), 6)
        if cost is None:
            # An unpriced model would make the total silently understate the real spend.
            self.usage["priced"] = False

    # Graph state the dashboard never reads, and that is large enough to matter: this whole object
    # is serialised and broadcast on every live-stream tick (several times a second while a game is
    # being written), and game_html alone is the finished 20KB+ game. design_review is the raw audit
    # - the dashboard shows qa.findings instead. Both stay in the graph state and the manifest.
    _PRIVATE_STATE = ("messages", "game_html", "design_review")

    def public(self) -> dict[str, Any]:
        state = {key: value for key, value in self.state.items() if key not in self._PRIVATE_STATE}
        return {
            "id": self.id, "genre": self.genre, "brief": self.brief, "model_id": self.model_id,
            "engine": self.engine, "status": self.status,
            "current_step": self.current_step, "state": state,
            "events": self.events, "created_at": self.created_at, "error": self.error,
            "stream": self.stream, "stream_started_at": self.stream_started_at,
            "agent_log": self.agent_log,
            "usage": self.usage, "usage_by_step": self.usage_by_step,
        }


# Where run checkpoints live. The approval gate is a durable interrupt - the graph stops and waits
# for a human Command that may arrive minutes or days later - but durability was only ever as good
# as the checkpointer behind it, and an in-memory one lasts exactly as long as the process. A
# reviewer who left a design document open overnight and restarted the dashboard lost the run:
# the finished games on disk survived, the pending decision did not. Set CHECKPOINT_DB to ":memory:"
# to opt back out.
CHECKPOINT_DB = os.getenv("CHECKPOINT_DB", str(GAME_OUTPUT_ROOT / "studio-checkpoints.sqlite"))


def _checkpointer():
    """A checkpointer that outlives the process, falling back to memory if it cannot be opened."""
    if CHECKPOINT_DB.strip() in {":memory:", "", "memory"}:
        return InMemorySaver()
    try:
        path = Path(CHECKPOINT_DB)
        path.parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False: runs are driven on worker threads via asyncio.to_thread, while
        # the HTTP handlers read state on the event loop thread.
        connection = sqlite3.connect(path, check_same_thread=False)
        saver = SqliteSaver(connection)
        saver.setup()
        return saver
    except (OSError, sqlite3.Error):
        # A dashboard that cannot write its checkpoint file must still run; it just cannot promise
        # that a pending approval survives a restart.
        return InMemorySaver()


# A run id as the dashboard mints them: uuid4().hex[:12].
_RUN_ID = re.compile(r"^[0-9a-f]{8,32}$")
# A folder holding one run's game. Named "<제목>_<엔진>_<런 id>" since the id alone made a directory
# of games a directory of hex strings; the trailing id is what every lookup resolves by. Folders
# from before the rename are the bare id, and still resolve.
_RUN_FOLDER = re.compile(r"^(?:.*_)?([0-9a-f]{8,32})$")
# What may name a folder at all. No dots and no separators, so a token from a request can be
# compared against directory entries without ever being able to leave the output root.
_SAFE_TOKEN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def run_id_of(folder_name: str) -> str:
    """The run id a folder belongs to, or "" if the name is not one of ours."""
    match = _RUN_FOLDER.match(folder_name)
    return match.group(1) if match else ""


def run_folder(run_id: str) -> Path:
    """Where this run's game lives.

    Found by its trailing id rather than built from the id, because the folder name now carries a
    title and an engine that the caller does not know. Returning the bare-id path when nothing
    matches keeps every caller's existing `is_file()` / `is_relative_to()` check doing its job -
    this resolves a location, it does not assert that anything is there.
    """
    # The general safe-token form, not _RUN_ID: /games and /launch have always accepted any plain
    # token so a hand-named folder stays reachable, and narrowing that here would have silently
    # unpublished every game not made by this pipeline.
    if not _SAFE_TOKEN.match(run_id):
        return GAME_OUTPUT_ROOT / "__invalid__"
    exact = GAME_OUTPUT_ROOT / run_id
    if exact.is_dir():
        return exact.resolve()
    if GAME_OUTPUT_ROOT.is_dir():
        for folder in GAME_OUTPUT_ROOT.iterdir():
            if folder.is_dir() and folder.name.endswith(f"_{run_id}"):
                return folder.resolve()
    return exact.resolve()


def _restore_finished_runs() -> dict[str, "Run"]:
    """Rebuild the run list from what previous runs actually produced.

    Checkpoints became durable, and the run list did not follow: StudioService.runs is an in-memory
    dict that nothing repopulates, so every restart emptied the dashboard. A finished Godot project
    with its run.bat sitting on disk became unreachable - the run could not be selected, so its
    launch button could not be shown, and pressing where it used to be did nothing at all.

    Rebuilt from the production manifest rather than from the checkpointer. The manifest is the
    record of what was delivered, it is one small file per run, and it survives a wiped checkpoint
    database; enumerating threads instead would mean paging through tens of thousands of
    checkpoints to find the handful that finished.
    """
    if not GAME_OUTPUT_ROOT.is_dir():
        return {}
    restored: dict[str, Run] = {}
    for folder in sorted(GAME_OUTPUT_ROOT.iterdir(), key=lambda p: p.stat().st_mtime):
        manifest_path = folder / "production-manifest.json"
        run_id = run_id_of(folder.name)
        if not folder.is_dir() or not run_id or not manifest_path.is_file():
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        concept = manifest.get("concept") or {}
        qa = manifest.get("qa") or {}
        engine = manifest.get("engine", "html5")
        failed = manifest.get("generation_mode", "").endswith("qa_failed")
        state: dict[str, Any] = {
            "engine": engine,
            "design_document": {"title": concept.get("title", run_id),
                                "implementation_plan": manifest.get("implementation_plan", {})},
            "implementation_plan": manifest.get("implementation_plan", {}),
            "qa": qa,
            "approval": {"decision": "approved"},
            "code_model_id": manifest.get("code_model_id", ""),
            "trace_notes": manifest.get("trace_notes", []),
        }
        # Adoption outlives the process that recorded it - that is the entire point of keeping it
        # in the manifest - so it has to come back with the run, or the dashboard offers to decide
        # something that was already decided.
        if (adoption := manifest.get("adoption")) is not None:
            state["adoption"] = adoption
        # Only paths that still exist: a folder the user has since cleaned out must not be offered.
        for key, candidate in (("game_path", folder / "index.html"),
                               ("godot_project_path", folder / "project.godot"),
                               ("launch_script_path", folder / LAUNCH_SCRIPT),
                               ("qa_report_path", folder / "qa-report.json")):
            if candidate.is_file():
                state[key] = str(candidate)
        if engine == "godot" and (folder / "build" / "index.html").is_file():
            state["game_path"] = str(folder / "build" / "index.html")
        run = Run(
            id=run_id,
            genre=(manifest.get("implementation_plan") or {}).get("genre", "이전 실행"),
            brief=concept.get("elevator_pitch", "")[:120],
            model_id=manifest.get("code_model_id", ""),
            engine=engine,
            config={"configurable": {"thread_id": run_id}, "recursion_limit": 200},
            status="qa_failed" if failed else "completed",
            restored=True,
            current_step="complete",
            created_at=datetime.fromtimestamp(manifest_path.stat().st_mtime, UTC).isoformat(),
        )
        run.state = state
        # What the run consumed, if it got as far as recording it. Restored onto the same fields
        # the dashboard reads for a live run, so a finished run keeps showing its own totals
        # instead of zeroes - which is the whole reason for writing them to disk.
        if isinstance(usage := manifest.get("usage"), dict):
            run.usage = {key: usage.get(key, 0) for key in run.usage}
            run.usage_by_step = dict(usage.get("by_step") or {})
        run.event("complete", "이전 실행에서 복원했습니다")
        restored[run_id] = run
    return restored


class StudioService:
    def __init__(self) -> None:
        self.graph = build_graph(_checkpointer())
        self.runs: dict[str, Run] = {}
        self.connections = Connections()
        self.loop: asyncio.AbstractEventLoop | None = None

    def restore(self) -> int:
        """Bring back what previous runs delivered. Called from lifespan, never at import.

        Deliberately not in __init__: the module-level service is constructed at import time, which
        is before lifespan loads .env - so a GAME_OUTPUT_DIR configured there would be read after
        the scan had already looked somewhere else. Doing it at startup also keeps import free of
        disk work, and keeps whatever happens to be in a developer's real output folder out of the
        tests.
        """
        # Idempotent: drop what a previous restore added, keep anything this process is driving.
        for run_id in [rid for rid, run in self.runs.items() if run.restored]:
            del self.runs[run_id]
        for run_id, run in _restore_finished_runs().items():
            self.runs.setdefault(run_id, run)
        return sum(1 for run in self.runs.values() if run.restored)

    def publish(self, run: Run) -> None:
        if self.loop:
            asyncio.run_coroutine_threadsafe(
                self.connections.broadcast({"type": "run:update", "run": run.public()}), self.loop
            )

    def _refresh_state(self, run: Run) -> None:
        snapshot = self.graph.get_state(run.config)
        run.state = dict(snapshot.values)

    def _on_agent_chunk(self, run: Run, payload: dict[str, Any]) -> None:
        """Forward one live text update from an agent (e.g. the idea/design-document LLM call
        streaming its structured answer) to the dashboard, throttled so a fast stream of tokens
        doesn't flood the websocket with a broadcast per character."""
        step = str(payload.get("step", "agent"))
        text = str(payload.get("text", ""))
        # A preview only grows within one model turn, so a shorter payload means a new turn started
        # (the code agent streams once per tool-loop pass). Restart the clock so "N초 경과" measures
        # the generation actually in flight rather than the whole phase.
        if len(text) < len(run.stream.get(step, "")):
            run.stream_started_at[step] = datetime.now(UTC).isoformat()
        run.stream[step] = text
        run.stream_started_at.setdefault(step, datetime.now(UTC).isoformat())
        now = time.monotonic()
        last = run._last_stream_publish.get(step, 0.0)
        if now - last < 0.2:
            return
        run._last_stream_publish[step] = now
        self.publish(run)

    def _on_agent_log(self, run: Run, payload: dict[str, Any]) -> None:
        """Record one discrete "who is doing what" event (model call started, a tool it chose to
        call and with what arguments, or its final text answer). Unlike the throttled raw-token
        stream above, these are infrequent and each one is worth showing immediately."""
        run.agent_log.append({"at": datetime.now(UTC).isoformat(), **payload})
        if len(run.agent_log) > 200:
            run.agent_log = run.agent_log[-200:]
        self.publish(run)

    def _drive(self, run: Run, payload: dict[str, Any] | Command) -> None:
        run.status = "running"
        run.event("orchestrator", "Run started")
        self.publish(run)
        try:
            for mode, event in self.graph.stream(payload, run.config, stream_mode=["updates", "custom"]):
                if mode == "custom":
                    if event.get("kind") == "usage":
                        run.add_usage(event.get("step") or run.current_step, event)
                        self.publish(run)
                    elif "kind" in event:
                        self._on_agent_log(run, event)
                    else:
                        self._on_agent_chunk(run, event)
                    continue
                if "__interrupt__" in event:
                    self._refresh_state(run)
                    run.status = "waiting_approval"
                    run.event("hitl-design-approval", "Design document is ready for review")
                    self.publish(run)
                    return
                step = next(iter(event), "orchestrator")
                self._refresh_state(run)
                run.event(step, f"{step} finished")
                self.publish(run)
            self._refresh_state(run)
            if run.state.get("approval", {}).get("decision") == "rejected":
                run.status = "rejected"
                run.event("complete", "Production stopped by reviewer")
            elif run.state.get("qa_report_path"):
                # The supervisor spent its whole repair and rethink budget without QA passing. The
                # run is over and reported as such, with the draft published only for review.
                findings = run.state.get("qa", {}).get("findings", [])
                run.status = "qa_failed"
                run.event("complete", f"QA 미통과 ({len(findings)}건) — 검토용으로 게임을 배포했습니다")
            else:
                run.status = "completed"
                run.event("complete", "Game package and QA report are ready")
        except Exception as error:
            run.status = "failed"
            run.error = f"{type(error).__name__}: {error}"
            run.event("failed", "The run stopped with an error")
        _record_usage(run)
        self.publish(run)

    async def start(self, request: CreateRun) -> Run:
        if request.offline:
            raise HTTPException(400, "새 게임 제작에는 기획·코드 모델 연결이 필요합니다. 오프라인 모드를 해제하세요.")
        if not await asyncio.to_thread(bedrock_credentials_configured):
            raise HTTPException(503, "Bedrock 인증이 없습니다. 로컬 AWS 프로필 또는 프로젝트 .env를 설정하세요. 게임을 템플릿으로 대체하지 않았습니다.")
        self.loop = asyncio.get_running_loop()
        run_id = uuid.uuid4().hex[:12]
        config = {"configurable": {"thread_id": run_id}, "recursion_limit": 200}
        selected_genre = request.genre.strip()
        selected_brief = request.brief.strip()
        display_genre = selected_genre if selected_genre and selected_genre != "auto" else "자동 기획"
        display_brief = selected_brief or "사용자 경험 미입력 — Agent 자동 기획"
        run = Run(
            id=run_id, genre=display_genre, brief=display_brief, model_id=request.model_id,
            engine=request.engine, config=config,
        )
        self.runs[run_id] = run
        payload = {
            "brief": (
                f"Requested genre: {display_genre}\n"
                f"Player brief: {selected_brief or '사용자 경험 없이 독자적으로 기획하세요.'}\n"
                f"Run seed: {run_id}"
            ),
            "output_dir": str(GAME_OUTPUT_ROOT),
            "workspace_dir": str((GAME_OUTPUT_ROOT / run_id).resolve()),
            "use_llm": True,
            "engine": request.engine,
            "model_id": request.model_id,
            "code_model_id": request.code_model_id,
            "generate_images": request.generate_images,
            "repair_attempts": 0,
            "trace_notes": [],
        }
        asyncio.create_task(asyncio.to_thread(self._drive, run, payload))
        return run

    async def revise(self, run_id: str, revision: RevisionRequest) -> Run:
        """Re-open a finished game at the code agent, carrying its own plan and workspace forward.

        Not a new run from a brief. The concept, the art direction and the approved contract all
        already exist and were already agreed, so re-deriving them would produce a different game -
        which is the opposite of "fix this one". The run enters at the stage where the artifact is
        written and goes on to verification and packaging exactly as a first build does.

        Seeded from the production manifest rather than from the checkpointer, for the same reason
        the run list is: the manifest is the record of what was delivered, it is one small file per
        run, and it outlives both the process and the checkpoint database.
        """
        if not await asyncio.to_thread(bedrock_credentials_configured):
            raise HTTPException(503, "Bedrock 인증이 없습니다. 보완 작업에도 코드 모델이 필요합니다.")
        if not _RUN_ID.match(run_id):
            raise HTTPException(404, "Invalid run ID")
        workspace = run_folder(run_id)
        manifest_path = workspace / "production-manifest.json"
        if not workspace.is_relative_to(GAME_OUTPUT_ROOT) or not manifest_path.is_file():
            raise HTTPException(404, "보완할 산출물이 없습니다.")
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise HTTPException(500, f"매니페스트를 읽지 못했습니다: {error}") from error
        if not manifest.get("implementation_plan") or not manifest.get("concept"):
            raise HTTPException(409, "이 산출물에는 기획과 계약 기록이 없어 보완을 시작할 수 없습니다.")
        if (live := self.runs.get(run_id)) and live.status == "running":
            raise HTTPException(409, "이 게임은 이미 작업 중입니다.")

        self.loop = asyncio.get_running_loop()
        engine = manifest.get("engine", "html5")
        # A fresh thread, because the checkpointer already holds a completed run under the old one
        # and invoking it again would resume that instead of starting this. The Run keeps the
        # original id: the folder is named after it, and /launch, /adopt and /games all resolve
        # through that name.
        thread_id = f"{run_id}-rev-{uuid.uuid4().hex[:6]}"
        concept = manifest["concept"]
        run = Run(
            id=run_id,
            genre=(manifest.get("implementation_plan") or {}).get("genre", "보완"),
            brief=f"보완: {revision.request}"[:120],
            model_id=manifest.get("code_model_id", ""),
            engine=engine,
            config={"configurable": {"thread_id": thread_id}, "recursion_limit": 200},
        )
        self.runs[run_id] = run
        run.event("revision", f"완성된 게임을 보완합니다: {revision.request[:120]}")

        # Asking for improvements is continuing to develop it, so the adoption metric records
        # itself here rather than through a separate button nobody would press. That button is what
        # this form replaced: it asked "이 게임을 이어서 개발합니까?" and then did nothing but write
        # the answer down, which read as an action and was not one.
        manifest["adoption"] = {"adopted": True, "note": revision.request[:500],
                                "decided_at": datetime.now(UTC).isoformat()}
        await asyncio.to_thread(
            manifest_path.write_text,
            json.dumps(manifest, indent=2, ensure_ascii=False), "utf-8")
        await asyncio.to_thread(_stage_draft_for_revision, workspace, engine)
        payload = {
            # stage "art" is what the supervisor reads as "art direction is settled", and its only
            # edge out is the code agent - so this is how a run enters at the build without
            # re-planning anything ahead of it.
            "stage": "art",
            "brief": manifest.get("brief") or concept.get("elevator_pitch", ""),
            "revision_request": revision.request,
            "concept": concept,
            "art": manifest.get("art") or {},
            "implementation_plan": manifest["implementation_plan"],
            # Already approved once, and the contract has not changed. Asking again would stop the
            # run at a gate whose question was answered before the player ever saw the game.
            "approval": {"decision": "approved", "comment": "보완 요청으로 재개"},
            "output_dir": str(GAME_OUTPUT_ROOT),
            "workspace_dir": str(workspace),
            "use_llm": True,
            "engine": engine,
            "model_id": manifest.get("code_model_id") or DEFAULT_MODEL_ID,
            "code_model_id": manifest.get("code_model_id") or DEFAULT_MODEL_ID,
            "generate_images": bool((manifest.get("art") or {}).get("asset_plan")),
            "repair_attempts": 0,
            "rethink_cycles": 0,
            "trace_notes": [],
        }
        asyncio.create_task(asyncio.to_thread(self._drive, run, payload))
        return run

    async def decide(self, run: Run, decision: ReviewDecision) -> None:
        if run.status != "waiting_approval":
            raise ValueError("This run is not waiting for a design decision.")
        run.status = "running"
        run.event("hitl-design-approval", f"Reviewer chose {decision.decision}")
        self.publish(run)
        command = Command(resume={"decision": decision.decision, "comment": decision.comment})
        asyncio.create_task(asyncio.to_thread(self._drive, run, command))


service = StudioService()


def _quiet_client_disconnects(loop: asyncio.AbstractEventLoop, context: dict[str, Any]) -> None:
    """Swallow the ConnectionResetError (WinError 10054) that Windows' proactor event loop raises
    from _call_connection_lost when a websocket client vanishes (tab closed, page refreshed, server
    stopped while the dashboard was open). It is pure shutdown noise - the socket is already gone -
    but it prints a full traceback that reads like a crash. Everything else is still reported."""
    error = context.get("exception")
    if isinstance(error, ConnectionResetError):
        return
    loop.default_exception_handler(context)


@asynccontextmanager
async def lifespan(_: FastAPI):
    load_dotenv(PROJECT_ROOT / ".env")
    loop = asyncio.get_running_loop()
    loop.set_exception_handler(_quiet_client_disconnects)
    service.loop = loop
    # After load_dotenv, so a GAME_OUTPUT_DIR set there is the folder actually scanned.
    restored = await asyncio.to_thread(service.restore)
    if restored:
        print(f"이전 실행 {restored}건을 복원했습니다.")
    yield


app = FastAPI(title="Game Studio Control Room", version="0.2.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=WEB_ROOT), name="static")


@app.middleware("http")
async def no_store_dashboard_assets(request, call_next):
    """The dashboard's HTML/JS/CSS change while the studio is being worked on. Without this, a
    browser can hold a cached index.html whose markup no longer matches the newer app.js, and the
    page silently stops updating.

    Vendored libraries are exempt: they never change, and mermaid alone is 3.5MB to re-fetch.
    """
    path = request.url.path
    response = await call_next(request)
    if path.startswith("/static/vendor/"):
        response.headers["Cache-Control"] = "public, max-age=604800, immutable"
    elif path == "/" or path.startswith("/static/"):
        response.headers["Cache-Control"] = "no-store, must-revalidate"
    return response


@app.get("/")
async def dashboard() -> FileResponse:
    return FileResponse(WEB_ROOT / "index.html")


@app.get("/api/runs")
async def list_runs() -> dict[str, list[dict[str, Any]]]:
    return {"runs": [run.public() for run in reversed(list(service.runs.values()))]}


@app.get("/api/runs/{run_id}")
async def get_run(run_id: str) -> dict[str, Any]:
    run = service.runs.get(run_id)
    if not run:
        raise HTTPException(404, "Run not found")
    return run.public()


@app.post("/api/runs", status_code=202)
async def create_run(request: CreateRun) -> dict[str, Any]:
    return (await service.start(request)).public()


@app.post("/api/runs/{run_id}/decision", status_code=202)
async def review_design(run_id: str, decision: ReviewDecision) -> dict[str, Any]:
    run = service.runs.get(run_id)
    if not run:
        raise HTTPException(404, "Run not found")
    try:
        await service.decide(run, decision)
    except ValueError as error:
        raise HTTPException(409, str(error)) from error
    return run.public()


@app.post("/api/runs/{run_id}/revise", status_code=202)
async def revise_game(run_id: str, revision: RevisionRequest) -> dict[str, Any]:
    """Rework a finished game from the player's own notes, starting at the code agent.

    The gap this closes: every check in this pipeline runs before anyone has played the result.
    Static QA proves the game runs, the design review proves it matches its contract, and neither
    can notice that the jump feels heavy. That feedback only exists after the game ships, and until
    now there was nothing to do with it - the only way back in was a new run from a brief, which
    produces a different game.
    """
    return (await service.revise(run_id, revision)).public()


@app.post("/api/runs/{run_id}/adopt", status_code=200)
async def mark_adoption(run_id: str, mark: AdoptionMark) -> dict[str, Any]:
    """Record whether a finished game is being taken forward.

    Written into the production manifest rather than into the run list, for the same reason the run
    list itself is rebuilt from manifests: the in-memory dict lasts as long as the process, and an
    adoption rate that resets on restart measures nothing. The manifest is already the record of
    what was delivered, it survives a wiped checkpoint database, and it is the file the folder
    carries with it.
    """
    if not _RUN_ID.match(run_id):
        raise HTTPException(404, "Invalid run ID")
    manifest_path = (run_folder(run_id) / "production-manifest.json").resolve()
    if (not manifest_path.is_relative_to(GAME_OUTPUT_ROOT) or not manifest_path.is_file()):
        raise HTTPException(404, "이 실행에는 채택 여부를 기록할 산출물이 없습니다.")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise HTTPException(500, f"매니페스트를 읽지 못했습니다: {error}") from error
    adoption = {"adopted": mark.adopted, "note": mark.note,
                "decided_at": datetime.now(UTC).isoformat()}
    manifest["adoption"] = adoption
    try:
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False),
                                 encoding="utf-8")
    except OSError as error:
        raise HTTPException(500, f"매니페스트를 저장하지 못했습니다: {error}") from error
    if run := service.runs.get(run_id):
        run.state["adoption"] = adoption
        service.publish(run)
    return adoption


@app.get("/api/adoption")
async def adoption_rate() -> dict[str, Any]:
    """The service's headline metric, counted off disk.

    Deliberately not "how many games passed QA". A prototype that runs and is not worth continuing
    is a successful run of this pipeline and a failed idea, and only the second number tells the
    studio anything. Undecided runs are reported separately rather than counted as rejections -
    the rate is over games somebody actually judged.
    """
    decided = adopted = finished = 0
    if GAME_OUTPUT_ROOT.is_dir():
        for manifest_path in GAME_OUTPUT_ROOT.glob("*/production-manifest.json"):
            if not run_id_of(manifest_path.parent.name):
                continue
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            finished += 1
            if (adoption := manifest.get("adoption")) is not None:
                decided += 1
                adopted += bool(adoption.get("adopted"))
    return {"finished": finished, "decided": decided, "adopted": adopted,
            "undecided": finished - decided,
            "rate": round(adopted / decided, 3) if decided else None}


@app.get("/api/graph")
async def graph_shape() -> dict[str, Any]:
    """The pipeline's real topology as mermaid, straight from the compiled graph.

    Generated rather than hand-maintained, so a change to the graph cannot leave the dashboard
    drawing a diagram that no longer matches what actually runs.
    """
    return {"mermaid": service.graph.get_graph().draw_mermaid()}


@app.get("/api/model-status")
async def model_status():
    from game_studio.godot import godot_available, godot_executable, godot_version

    ready = await asyncio.to_thread(godot_available)
    return {"configured": await asyncio.to_thread(bedrock_credentials_configured),
            "provider": "Amazon Bedrock", "model_access_verified": False,
            "comfyui_available": await asyncio.to_thread(comfyui_available),
            "comfyui_server": os.getenv("COMFYUI_SERVER", "http://127.0.0.1:8188"),
            "engines": GAME_ENGINES,
            "godot_available": ready,
            "godot_path": godot_executable() or "",
            "godot_version": await asyncio.to_thread(godot_version) if ready else ""}


@app.post("/api/runs/{run_id}/launch", status_code=202)
async def launch_godot_game(run_id: str) -> dict[str, Any]:
    """Start the finished Godot game on this machine, from the dashboard's run button.

    A browser cannot execute a .bat, so the button has to come back here. That makes this the one
    endpoint in the dashboard that starts a local process, and it is deliberately narrow: the only
    thing it can ever run is the run.bat this pipeline itself wrote, inside a run folder under
    GAME_OUTPUT_ROOT, named by an id that has already been pattern-checked. No path, argument or
    command reaches it from the request.
    """
    from game_studio.godot import LAUNCH_SCRIPT

    if not re.fullmatch(r"[a-zA-Z0-9_-]+", run_id):
        raise HTTPException(404, "Invalid run ID")
    root = run_folder(run_id)
    script = (root / LAUNCH_SCRIPT).resolve()
    # Runs live in memory but finished games outlive the dashboard, so this is answered from disk.
    if not root.is_relative_to(GAME_OUTPUT_ROOT) or script.parent != root or not script.is_file():
        raise HTTPException(404, "이 실행에는 Godot 실행 스크립트가 없습니다.")
    try:
        await asyncio.to_thread(_spawn_detached, script, root)
    except OSError as error:
        raise HTTPException(500, f"게임을 실행하지 못했습니다: {error}") from error
    return {"launched": True, "script": str(script)}


def _spawn_detached(script: Path, cwd: Path) -> None:
    """Launch the game and return.

    The dashboard must not wait on a window the player closes whenever they feel like it, so the
    child gets its own console and its own lifetime.

    CREATE_NEW_CONSOLE only. It and DETACHED_PROCESS are mutually exclusive on Windows - combining
    them is not "more detached", it is an invalid flag pair, and CreateProcess rejects it outright
    with ERROR_INVALID_PARAMETER (WinError 87) before the launcher ever runs.

    Nor are the child's streams sent to DEVNULL. run.bat pauses when the engine is missing or the
    game exits with an error, so its output is the only thing that explains a failed launch; with
    it discarded the player would get an invisible process waiting forever on a prompt nobody can
    see. It writes to the new console instead, which is what that console is for.
    """
    extra: dict[str, Any] = (
        {"creationflags": subprocess.CREATE_NEW_CONSOLE} if os.name == "nt"
        else {"start_new_session": True}
    )
    subprocess.Popen([str(script)], cwd=str(cwd), shell=False, close_fds=True, **extra)


@app.get("/games/{run_id}")
async def canonical_game_url(run_id: str):
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", run_id):
        raise HTTPException(404, "Invalid game ID")
    return RedirectResponse(f"/games/{run_id}/", status_code=307)


@app.get("/games/{run_id}/")
async def play_game(run_id: str) -> FileResponse:
    run = service.runs.get(run_id)
    # A QA-failed run still publishes its draft for review, so it has to be playable too.
    if run and run.status not in {"completed", "qa_failed"}:
        raise HTTPException(404, "Playable game not found")

    # Runs live in memory, but completed games must remain playable after the
    # dashboard restarts. The fallback path is constrained to GAME_OUTPUT_ROOT.
    game_path = (
        Path(run.state.get("game_path", "")).resolve()
        if run
        else (run_folder(run_id) / "index.html").resolve()
    )
    if not game_path.is_file() or not game_path.is_relative_to(GAME_OUTPUT_ROOT):
        raise HTTPException(404, "Game package not found")
    return FileResponse(game_path, media_type="text/html")


@app.get("/games/{run_id}/{asset_path:path}")
async def game_asset(run_id: str, asset_path: str):
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", run_id):
        raise HTTPException(404, "Invalid game ID")
    root = run_folder(run_id)
    path = (root / asset_path).resolve()
    if (not root.is_relative_to(GAME_OUTPUT_ROOT) or not path.is_relative_to(root)
            or not (root / "index.html").is_file() or not path.is_file()
            or path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp", ".js", ".css", ".wav", ".ogg", ".mp3"}):
        raise HTTPException(404, "Asset not found")
    return FileResponse(path)


@app.websocket("/ws")
async def live_updates(client: WebSocket) -> None:
    await service.connections.connect(client)
    await client.send_json({"type": "runs:initial", "runs": [run.public() for run in service.runs.values()]})
    try:
        while True:
            await client.receive_text()  # Keeps the connection alive; client may send ping.
    except WebSocketDisconnect:
        service.connections.disconnect(client)


def main() -> None:
    parser = argparse.ArgumentParser(description="Start the Game Studio HITL dashboard.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8765, type=int)
    args = parser.parse_args()
    uvicorn.run("game_studio.server:app", host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    main()
