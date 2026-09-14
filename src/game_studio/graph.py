"""LangGraph orchestration for a multi-agent browser-game build."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy, interrupt
from langsmith import traceable
from pydantic import ValidationError

from .agent_tools import GAME_TOOLS
from .code_agent import build_code_agent
from .agents import (
    CURRENT_STEP,
    _content_text,
    _model,
    _structured,
    static_qa,
    create_art,
    create_concept,
    qa_model_id,
    run_deep_director,
    stream_turn,
)
from .models import (
    ArtDirection,
    DesignReview,
    GameConcept,
    ImplementationPlan,
    StudioState,
    SupervisorDecision,
    game_output_dir,
)
from .prompts import CODE_SYSTEM, SUPERVISOR_ESCALATION_SYSTEM


# Escalation budget for a build that fails verification. The supervisor chooses what to spend it on,
# but it cannot overspend: a cheap text-only repair, then rethink cycles that hand the code agent a
# fresh tool loop, and once both are gone the run ends with a clean verdict.
#
# One rethink, not two. A rethink is the most expensive thing this pipeline can do - a full code
# agent loop (up to CODE_AGENT_MODEL_CALLS model calls at 16k tokens each) plus another audit - and
# a second one was routinely spent re-litigating an already playable game against reviewer nitpicks.
# Both are env-tunable for a run that genuinely wants to keep grinding.
MAX_REPAIR_ATTEMPTS = int(os.getenv("MAX_REPAIR_ATTEMPTS", "1"))
MAX_RETHINK_CYCLES = int(os.getenv("MAX_RETHINK_CYCLES", "1"))
# What the dashboard and the logs call the supervisor.
SUPERVISOR = "총괄 감독"
# Share of the implementation contract that has to be explicitly rejected before the design review
# blocks a release. A few disputed items out of a dozen are feedback, not a broken game: the build
# is published with them recorded rather than sent round another repair cycle. Only a review saying
# most of the contract is unimplemented means the run actually built the wrong thing.
QA_REJECT_TOLERANCE = float(os.getenv("QA_REJECT_TOLERANCE", "0.5"))
# How many advisory notes ("other bugs I noticed") are kept. They are not contract violations and
# they do not block a release, but an unbounded list of them became the repair instructions - one
# run shipped with 31 open items, nearly all of them style opinions. Keeping the first few keeps the
# report readable and the repair prompt focused on what actually failed.
ADVISORY_FINDING_LIMIT = int(os.getenv("ADVISORY_FINDING_LIMIT", "5"))
# The design review's answer grows with the contract. Now that only a rejected requirement has to
# carry evidence, a clean audit is a short list of booleans and this ceiling is headroom rather than
# a target - and _structured grows it by itself if an audit ever does run out of room.
DESIGN_REVIEW_MAX_TOKENS = int(os.getenv("DESIGN_REVIEW_MAX_TOKENS", "12000"))
# The escalation decision is a short verdict, not a document. It was sharing the 8k structured
# default and never needed a fraction of it.
ESCALATION_MAX_TOKENS = int(os.getenv("ESCALATION_MAX_TOKENS", "2000"))

_RETRYABLE_BEDROCK_CODES = {
    "ThrottlingException", "TooManyRequestsException", "ServiceUnavailableException",
    "InternalServerException", "ModelTimeoutException", "ModelNotReadyException",
}


def _is_transient(error: BaseException) -> bool:
    """Whether a node failure is worth another attempt rather than losing the run.

    LangGraph's default predicate does not retry botocore's ReadTimeoutError, and that single gap
    was enough to throw away a run that had already passed human approval: a long generation went
    quiet, the socket timed out, and the exception unwound the whole graph.
    """
    from botocore.exceptions import (
        ClientError,
        ConnectionClosedError,
        ConnectTimeoutError,
        EndpointConnectionError,
        ReadTimeoutError,
    )
    if isinstance(error, (ReadTimeoutError, ConnectTimeoutError, EndpointConnectionError,
                          ConnectionClosedError)):
        return True
    if isinstance(error, ClientError):
        return error.response.get("Error", {}).get("Code", "") in _RETRYABLE_BEDROCK_CODES
    return False


# Only the model-calling nodes get this. The tools node is deliberately excluded: its tools write
# files and generate images, so a retry there could duplicate real side effects.
MODEL_RETRY = RetryPolicy(
    max_attempts=3, initial_interval=2.0, backoff_factor=2.0, retry_on=_is_transient
)


def _concept(state: StudioState) -> GameConcept:
    return GameConcept.model_validate(state["concept"])


def _art(state: StudioState) -> ArtDirection:
    return ArtDirection.model_validate(state["art"])


def _workspace(state: StudioState, concept: GameConcept) -> Path:
    """Each run owns one isolated workspace; older callers retain title-based output."""
    return Path(state.get("workspace_dir") or game_output_dir(Path(state["output_dir"]), concept.title))


def _existing_sprites(state: StudioState) -> list[str]:
    """Sprites this run has already generated, so a later pass knows what it can reuse."""
    try:
        assets = _workspace(state, _concept(state)) / "assets"
    except (KeyError, ValueError):
        return []
    return sorted(p.name for p in assets.glob("*.png")) if assets.exists() else []


def _chunk_sink(step: str):
    """Return a callback that forwards live agent text to whoever is streaming this run.

    get_stream_writer() only works inside an active graph run (.invoke()/.stream()); a direct,
    out-of-graph call to a node function (as some unit tests do) has no such context, so this
    degrades to a no-op instead of raising.
    """
    try:
        writer = get_stream_writer()
    except RuntimeError:
        return None
    return lambda text: writer({"step": step, "text": text})


def _has_writer() -> bool:
    """Whether this code is running inside a graph run that can accept custom stream events."""
    try:
        get_stream_writer()
    except RuntimeError:
        return False
    return True


def _step(name: str) -> None:
    """Declare the executing step so token usage is billed to it rather than to whichever node
    happened to finish last."""
    CURRENT_STEP.set(name)


def _log(kind: str, **fields) -> None:
    """Emit one structured progress-log event: which agent/model is running, which tool it just
    called (with what arguments), or what text it answered with. This is the "누가 지금 뭘 하고
    있는지" feed the dashboard renders live. Safe to call from anywhere, including outside an
    active graph run (some unit tests invoke node functions directly without going through
    .invoke()/.stream()), in which case it silently does nothing.
    """
    try:
        writer = get_stream_writer()
    except RuntimeError:
        return
    writer({"kind": kind, **fields})


def _trim(value: object, limit: int = 300) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
    return text if len(text) <= limit else f"{text[:limit]}…({len(text)}자)"


def _stream_call(model, messages, agent: str, step: str):
    """Run one model turn in streaming mode instead of a single blocking .invoke().

    A plain .invoke() has nothing to report until the whole answer is finished, which is why the
    slowest calls in this pipeline (writing a whole game, repairing it) used to look frozen for a
    minute at a time. Streaming lets progress reach the dashboard while it happens, and the turn is
    returned as the same AIMessage the caller would otherwise have got.
    """
    sink = _chunk_sink(step)
    answer = stream_turn(
        model, messages,
        on_preview=(lambda acc: sink(preview) if (preview := acc.preview()) else None)
        if sink else None,
    )
    if answer.empty:
        raise RuntimeError(f"{agent} 모델이 빈 스트림을 반환했습니다.")
    # Running out of token budget mid-answer is silent otherwise: LangChain's partial-JSON parser
    # closes off a half-streamed tool argument and hands back what looks like a perfectly good
    # call, just with a truncated file in it. The stop reason is the only honest signal, so surface
    # it - write_game_file will reject the incomplete HTML, and this says why.
    if answer.truncated():
        _log("model_text", agent=agent, text=(
            f"토큰 예산({getattr(model, 'max_tokens', '?')})을 모두 써서 답변이 중간에 끊겼습니다. "
            "도구 인자가 잘렸을 수 있습니다."
        ))
    # A plain AIMessage: state, ToolNode and the routers all treat this as a finished assistant
    # turn, and an accumulated chunk carries streaming-only bookkeeping that must not be written
    # into the checkpoint.
    return answer.finish()


@traceable(name="idea-agent", run_type="chain")
def idea_node(state: StudioState) -> dict:
    _step("idea")
    model_id = state.get("model_id")
    _log("model_call", agent="기획 Agent", model=model_id)
    concept = create_concept(
        state["brief"], state.get("use_llm", True), model_id, on_chunk=_chunk_sink("idea"),
        production_brief=state.get("production_brief", ""),
    )
    _log("model_text", agent="기획 Agent", text=f"컨셉 '{concept.title}' 생성 완료")
    return {"concept": concept.model_dump(), "stage": "idea"}


@traceable(name="design-document", run_type="chain")
def design_document_node(state: StudioState) -> dict:
    """Create the reviewable production document before any asset/code work begins."""
    _step("design_document")
    concept = _concept(state)
    model_id = state.get("code_model_id") or state.get("model_id")
    _log("model_call", agent="기획 문서", model=model_id)
    plan = _structured(
        ImplementationPlan,
        "You are a game systems engineer. Convert this concept and user request into a concrete, "
        "feasible implementation contract. Preserve the requested genre. Specify unique mechanics, "
        "win/loss conditions, state transitions and 3-8 objectively testable acceptance criteria. "
        "Write in Korean. Do not replace it with a generic survival game.\n"
        "Every mechanic must say what the player does and what it costs or earns them, with the "
        "numbers a developer needs. Do not list mechanics that only describe internal machinery.\n"
        "The win condition must reward playing well, not merely finishing: include the measure the "
        "player is scored on (time, score, rank, distance) and how it is shown.\n"
        "Then check the incentives before you answer: if playing slowly, passively or defensively "
        "is the safest way to win, the contract is wrong. Adjust the rules - a clock, a decaying "
        "score, a pursuing rival, a resource that only refills through risk - until cautious play "
        "measurably loses, and make one acceptance test verify exactly that.",
        f"User request: {state['brief']}\nConcept: {concept.model_dump_json()}"
        # Carry the borrowed mechanics forward so the contract stays inside the genre the concept
        # anchored on, instead of drifting once it is turned into systems.
        + (f"\n참조 게임(이 메커닉을 알아볼 수 있게 유지하세요): {'; '.join(concept.reference_games)}"
           if concept.reference_games else "")
        + (f"\nProduction director's brief: {state['production_brief']}" if state.get("production_brief") else ""),
        model_id,
        on_chunk=_chunk_sink("design_document"),
    )
    _log("model_text", agent="기획 문서", text=f"구현 계획 생성 완료 (장르: {plan.genre}, 메커닉 {len(plan.mechanics)}개)")
    document = {
        "title": concept.title,
        "summary": concept.elevator_pitch,
        "player_goal": concept.player_goal,
        "controls": concept.controls,
        "core_loop": concept.core_loop,
        "difficulty_curve": concept.difficulty_curve,
        "visual_direction": concept.visual_direction,
        "reference_games": concept.reference_games,
        "delivery_scope": "외부 네트워크 의존성 없이 단독 실행되는 오리지널 HTML5 Canvas 게임 1개",
        "approval_question": "이 기획안을 승인하고 이미지·코드·QA 제작을 시작할까요?",
        "implementation_plan": plan.model_dump(),
    }
    return {"design_document": document, "implementation_plan": plan.model_dump(),
            "stage": "design_document"}


@traceable(name="hitl-design-approval", run_type="chain")
def approval_node(state: StudioState) -> dict:
    """Durable approval boundary. Execution resumes only with an explicit Command."""
    response = interrupt({
        "type": "design_approval",
        "document": state["design_document"],
        "instruction": "Choose approve to start production or reject to stop this run.",
    })
    decision = response if isinstance(response, dict) else {"decision": str(response)}
    accepted = decision.get("decision") == "approve"
    return {"approval": {
        "decision": "approved" if accepted else "rejected",
        "comment": str(decision.get("comment", "")),
    }, "stage": "approval"}


@traceable(name="art-direction", run_type="chain")
def art_node(state: StudioState) -> dict:
    """Plan the visual system only. Nothing is rendered here: the art direction's palette, canvas
    effects, backdrop image_prompt and per-object asset_plan are the menu the code agent works
    from, and the code agent generates whatever raster art it actually needs while it builds."""
    _step("art")
    concept = _concept(state)
    model_id = state.get("model_id")
    # Reached a second time only when a rethink decided the plan itself was the problem.
    revising = bool(state.get("art_revision_needed"))
    findings = state.get("qa", {}).get("findings", []) if revising else []
    _log("model_call", agent="아트 기획", model=model_id,
         note="QA 지적을 반영해 아트 방향을 다시 세웁니다." if revising else "")
    art = create_art(
        concept, state.get("use_llm", True), model_id,
        findings=findings, existing_sprites=_existing_sprites(state) if revising else None,
    )
    _log("model_text", agent="아트 기획",
         text=(f"아트 방향 {'재수립' if revising else '생성'} 완료 (에셋 후보 {len(art.asset_plan)}개)"
               " — 실제 이미지 생성은 코드 Agent가 필요할 때 수행"))
    return {
        "art": art.model_dump(),
        "stage": "art",
        "art_revision_needed": False,
        "art_revised": revising or bool(state.get("art_revised")),
        "trace_notes": state.get("trace_notes", [])
        + [("Art plan revised after QA." if revising else "Canvas-first art plan generated.")],
    }


def _code_system_prompt(state: StudioState) -> str:
    """The standing contract for the build, including whether raster art is on for this run."""
    art = _art(state)
    if state.get("generate_images", False):
        plan_items = "; ".join(art.asset_plan[:8]) or "(아트 계획에 객체 목록이 없습니다)"
        image_guidance = (
            "\nYou own art generation for this build end to end - no separate image agent runs "
            "before or after you, and nothing is generated unless you ask for it. "
            "generate_comfyui_image is one of your tools and raster generation IS enabled for this "
            "run, so use it. Requirement: generate a sprite for the main gameplay objects the art "
            "direction planned - at minimum the player's own object and its primary opponent or "
            "obstacle - and draw those sprites in the game. Do not ship a Canvas-only rectangle "
            "for the player when a sprite was planned and the budget allows one.\n"
            f"Planned objects: {plan_items}\n"
            "Call list_game_assets first so you never regenerate something that already exists, "
            'pass a short asset_name (e.g. "player-car", "rival-car", "cone", "backdrop") so the '
            "file is easy to reference and re-generating the same object replaces its old file, and "
            "draw the returned assets/<asset_name>.png with a Canvas fallback in case it is ever "
            "missing.\n"
            "Prompt each sprite with the OBJECT ONLY - its shape, colours and style. Never describe "
            "a background, scene, floor or shadow: the staging is added for you, and anything you "
            "put behind the object survives the background cut and ships as an opaque box over your "
            "game.\n"
            "Every sprite is generated facing one direction you choose, and you must draw it with "
            "exactly the matching rotation - the tool tells you which, and list_game_assets repeats "
            "it for every file. facing=\"right\" for anything that travels (ships, cars, creatures, "
            "projectiles): draw with ctx.rotate(Math.atan2(vy, vx)) and no extra offset, or mirror "
            "with ctx.scale(-1, 1) in a game that only moves left and right. facing=\"up\" for "
            "top-down art that reads nose-up: add Math.PI / 2. facing=\"none\" for items, coins, "
            "blocks and obstacles: do not rotate them at all. Do not invent your own offset and do "
            "not rotate a facing=\"none\" sprite - a sprite pointing the wrong way is the single "
            "most obvious defect in a finished game.\n"
            "Sprites come back already trimmed to the art with a transparent background, so draw "
            "them at the entity's own size with ctx.drawImage and do not add your own inset. Use "
            "kind=\"backdrop\" only for a full-frame background image; it keeps its background.\n"
            "There is a small per-run image budget; spend it on the objects the player looks at "
            "most and fall back to Canvas drawing for the rest. Interleave art with code freely: "
            "write part of the game, generate the sprite you just discovered you need, wire it in, "
            "keep building."
        )
    else:
        image_guidance = (
            "\nRaster image generation is disabled for this run; use Canvas-only art for every object."
        )
    return (
        CODE_SYSTEM
        + image_guidance
        + "\nUse the provided tools autonomously. First call list_game_assets. "
        "Then write_game_file, then run_static_qa. "
        "If repair is needed, read the draft narrowly before rewriting it: "
        "read_game_file(outline=True) gives a line map, then read_game_file(start_line, "
        "end_line) returns just the section at fault. Everything you read back stays in this "
        "conversation and is re-sent on every later turn, so never pull the whole file when a "
        "section will do. Then repair_html and QA again. "
        "Call generate_asset once to record the final art plan. Stop when static QA passes."
    )


def _code_task(state: StudioState) -> str:
    """The one human turn that starts the agent: the approved design it has to build."""
    concept, art = _concept(state), _art(state)
    task = (
        f"Build this game.\nOriginal request: {state['brief']}\n"
        f"Concept: {concept.model_dump_json()}\nArt: {art.model_dump_json()}\n"
        f"Implementation contract: {json.dumps(state['implementation_plan'], ensure_ascii=False)}\n"
        f"Review comment: {state.get('approval', {}).get('comment', '')}"
    )
    if state.get("qa_guidance"):
        task += (
            f"\n\n{SUPERVISOR}의 수정 지침 (QA 미통과 후 재검토, 최우선으로 반영하세요):\n"
            f"{state['qa_guidance']}"
        )
    return task

@traceable(name="autonomous-code-agent", run_type="chain")
def code_node(state: StudioState) -> dict:
    """Run the code agent to completion.

    The write/inspect/repair loop, its call budget, retries and history compaction all live inside
    the create_agent graph now (see code_agent.py), so this node hands it the task and waits. The
    agent is streamed rather than invoked so the dashboard keeps its live view of the build.
    """
    _step("code")
    if not state.get("use_llm", True):
        raise RuntimeError("코드 모델이 연결되지 않았습니다. 고정 게임을 대신 생성하지 않습니다.")
    agent = build_code_agent(
        state.get("code_model_id") or state.get("model_id"),
        GAME_TOOLS,
        _code_system_prompt(state),
        _is_transient,
    )
    # Only the fields the game tools read through InjectedState, plus the task itself.
    payload = {
        "messages": [("human", _code_task(state))],
        "brief": state["brief"],
        "concept": state["concept"],
        "art": state["art"],
        "implementation_plan": state["implementation_plan"],
        "approval": state.get("approval", {}),
        "qa_guidance": state.get("qa_guidance", ""),
        "output_dir": state.get("output_dir", ""),
        "workspace_dir": state.get("workspace_dir", ""),
        "generate_images": state.get("generate_images", False),
        "model_id": state.get("model_id", ""),
        "code_model_id": state.get("code_model_id", ""),
    }
    # The agent is a graph of its own, so its middleware writes to *its* custom stream, not this
    # run's. Without relaying, everything the dashboard shows about the build - the model calls,
    # the tool calls and results, the live generation preview - silently goes nowhere.
    relay = get_stream_writer() if _has_writer() else None

    def run(task: str) -> list:
        collected: list = []
        for mode, chunk in agent.stream(
            {**payload, "messages": [("human", task)]},
            {"recursion_limit": 120},
            stream_mode=["updates", "custom"],
        ):
            if mode == "custom":
                if relay is not None:
                    relay(chunk)
                continue
            for update in (chunk or {}).values():
                collected.extend((update or {}).get("messages", []) or [])
        return collected

    produced = run(_code_task(state))
    # create_agent stops the moment a turn comes back without tool calls, so one chatty answer can
    # end the build with nothing written. Say so plainly and give it one more go before the run
    # walks into QA and fails on a draft that was never created.
    draft = _workspace(state, _concept(state)) / "draft.html"
    if not draft.exists():
        _log("model_text", agent="코드 Agent",
             text="도구를 호출하지 않고 종료했습니다. write_game_file을 요구하며 한 번 더 시도합니다.")
        produced += run(
            _code_task(state)
            + "\n\n이전 시도는 도구를 호출하지 않아 아무 파일도 만들지 못했습니다. 설명하지 말고 "
              "지금 즉시 write_game_file로 완성된 게임 HTML 전체를 저장한 뒤 run_static_qa를 호출하세요."
        )
    return {"messages": produced, "stage": "code"}


@traceable(name="qa-agent", run_type="chain")
def qa_node(state: StudioState) -> dict:
    _step("qa")
    concept = _concept(state)
    if not state.get("game_html"):
        draft = _workspace(state, concept) / "draft.html"
        if not draft.exists():
            raise RuntimeError("코드 Agent가 draft.html을 작성하지 않았습니다. 제작 실패입니다.")
        html = draft.read_text(encoding="utf-8")
    else:
        html = state["game_html"]
    report = static_qa(html)
    if report.status != "pass":
        _log("model_text", agent="QA 검증", text=f"정적 QA 실패: {_trim('; '.join(report.findings), 300)}")
        return {"game_html": html, "qa": report.model_dump(), "stage": "qa"}
    plan = ImplementationPlan.model_validate(state["implementation_plan"])
    requirements = plan.mechanics + [plan.win_condition, plan.loss_condition] + plan.acceptance_tests
    # The design review sends the whole game source, so it is one of the most expensive calls in
    # the pipeline. Repair and rethink cycles brought QA back here five times in one measured run;
    # re-auditing byte-identical HTML just pays for the same answer again.
    fingerprint = hashlib.sha256(html.encode("utf-8")).hexdigest()
    if state.get("design_review") and state.get("design_review_hash") == fingerprint:
        _log("model_text", agent="QA 검증", text="소스가 직전 감사와 동일해 설계 감사를 건너뜁니다.")
        return {"game_html": html, "qa": state["qa"], "design_review": state["design_review"],
                "stage": "qa"}
    model_id = qa_model_id(state.get("code_model_id") or state.get("model_id"))
    _log("model_call", agent="QA 검증", model=model_id, note=f"요구사항 {len(requirements)}개 대조 감사 중")
    review = _structured(
        DesignReview,
        # Asking for concrete evidence on every requirement made the reviewer write an essay about
        # code it was only meant to check: the answer grew with the contract until it ran out of
        # output budget mid-array, and the prose it did produce was mostly justifying things that
        # already worked. It is asked for a verdict now, and for reasons only where it says no.
        "Check this game's SOURCE against the listed requirements, in order. Return one check per "
        "requirement, reusing the requirement string exactly as given.\n"
        "For a requirement that is met, set passed=true and leave evidence empty. Write evidence "
        "ONLY for a requirement you reject, in one short sentence (under 120 characters) naming the "
        "function or variable that is missing or wrong.\n"
        "Reject a requirement only when the game genuinely does not implement it - no code for the "
        "mechanic, a win/loss condition that can never trigger, controls that are wired to nothing, "
        "or the wrong genre entirely. A mechanic that is implemented differently than you would "
        "have written it, or is simpler than you would like, passes. Do not reject on style, "
        "polish, balance, naming, or code you would refactor. Judge the logic, not keywords, and "
        "never claim to have run the game.\n"
        "findings is for launch-blocking bugs you noticed outside the requirement list - crashes, "
        "infinite loops, unreachable states. Leave it empty unless the game is actually broken; it "
        "is not a place for suggestions.",
        f"Concept: {concept.model_dump_json()}\nRequirements: {json.dumps(requirements, ensure_ascii=False)}\nHTML: {html}",
        model_id,
        # Streamed because this is the longest single call in the pipeline and a blocking invoke
        # shows nothing while it runs; streaming also exposes the stop reason, so a truncated audit
        # says so instead of coming back as an unexplained schema violation.
        on_chunk=_chunk_sink("qa"),
        max_tokens=DESIGN_REVIEW_MAX_TOKENS,
    )
    verdict = _verdict(review, requirements)
    report.status = "repair" if verdict.blocking else "pass"
    report.findings = verdict.findings
    # Only the unmet terms of the contract are worth paying a repair for. The advisories ride along
    # in findings so they stay visible, but they must not steer the rewrite.
    report.repair_instructions = "\n".join(verdict.unmet)
    if verdict.unreviewed:
        _log("model_text", agent="QA 검증",
             text=f"검토되지 않은 요구사항 {len(verdict.unreviewed)}건: {_trim('; '.join(verdict.unreviewed), 200)}")
    _log("model_text", agent="QA 검증", text=(
        "설계 감사 통과" if not verdict.findings else
        f"설계 감사: 계약 미충족 {len(verdict.unmet)}/{len(requirements)}건"
        f"{f', 참고 {len(verdict.advisories)}건' if verdict.advisories else ''}"
        f" — {'재수정 필요' if verdict.blocking else '배포 가능(기록만 남김)'}"
        f": {_trim('; '.join(verdict.findings), 300)}"))
    return {"game_html": html, "qa": report.model_dump(), "design_review": review.model_dump(),
            "design_review_hash": fingerprint, "stage": "qa"}


def _normalize_requirement(text: str) -> str:
    """Match requirements on their letters alone, so a reviewer's spacing or punctuation does not
    turn a met requirement into an unmet one."""
    return "".join(ch for ch in text.lower() if ch.isalnum())


@dataclass(frozen=True)
class _Verdict:
    """What the design review actually established, separated from how it phrased it."""

    unmet: list[str]          # requirements from OUR contract the reviewer explicitly rejected
    advisories: list[str]     # other bugs it noticed; recorded, never blocking on their own
    unreviewed: list[str]     # requirements it did not mention at all
    blocking: bool

    @property
    def findings(self) -> list[str]:
        return self.unmet + self.advisories


def _verdict(review: DesignReview, requirements: list[str]) -> _Verdict:
    """Turn one design review into a release decision.

    Two things are deliberately not counted as failures. A requirement the reviewer paraphrased,
    merged or skipped is not evidence of a broken game - matching on exact strings sent working
    builds back to repair over the reviewer's wording. And a rejection that matches no requirement
    we actually asked about is the reviewer inventing its own terms: those used to inflate the
    rejection ratio and block releases whose real contract was fully met, which is how a playable
    game ended up shipping with dozens of open items against it.
    """
    rejected = {_normalize_requirement(c.requirement) for c in review.checks if not c.passed}
    reviewed = {_normalize_requirement(c.requirement) for c in review.checks}
    unmet, unreviewed = [], []
    for requirement in requirements:
        key = _normalize_requirement(requirement)
        if key in rejected:
            unmet.append(f"미충족: {requirement}")
        elif key not in reviewed:
            unreviewed.append(requirement)
    advisories = [str(note) for note in review.findings][:ADVISORY_FINDING_LIMIT]
    # Tolerating partial coverage is not the same as accepting no audit at all: a review that
    # checked nothing is a broken review, and must not read as a clean bill of health.
    if not review.checks:
        return _Verdict(
            unmet=["설계 감사 결과가 비어 있습니다. 요구사항별 검증을 다시 수행해야 합니다."],
            advisories=advisories, unreviewed=unreviewed, blocking=True,
        )
    # Measured against the contract we asked about, not against everything the reviewer chose to
    # say. Below the tolerance the build ships with its open items recorded rather than paying for
    # another repair cycle over a handful of disputed mechanics.
    return _Verdict(
        unmet=unmet, advisories=advisories, unreviewed=unreviewed,
        blocking=bool(unmet) and (len(unmet) / max(1, len(requirements))) > QA_REJECT_TOLERANCE,
    )


@traceable(name="repair-agent", run_type="chain")
def repair_node(state: StudioState) -> dict:
    _step("repair")
    from .agents import normalize_html
    prompt = f"Repair the entire game to satisfy the approved design.\nConcept: {json.dumps(state['concept'])}\nContract: {json.dumps(state['implementation_plan'])}\nArt: {json.dumps(state['art'])}\nReview: {json.dumps(state['qa'])}\nSource: {state['game_html']}"
    # The supervisor chose this repair and said what it wants changed; without this the instructions
    # it just paid a model call to write would only ever reach the code agent.
    if state.get("qa_guidance"):
        prompt += f"\nSupervisor's instructions (follow these first):\n{state['qa_guidance']}"
    model_id = state.get("code_model_id") or state.get("model_id")
    _log("model_call", agent="자동 수정", model=model_id)
    response = _stream_call(
        _model(model_id, max_tokens=16000),
        [("system", CODE_SYSTEM), ("human", prompt)],
        "자동 수정",
        "repair",
    )
    html = normalize_html(_content_text(response))
    lowered = html.lower()
    if "<canvas" not in lowered or "</html>" not in lowered:
        # The repair call did not return a complete standalone game (truncation, refusal, a
        # conversational reply, ...). Keep the previously written draft intact instead of
        # clobbering working output with garbage, and surface the real cause so the next repair
        # attempt (or the final failure message) is honest about what went wrong.
        qa = dict(state.get("qa", {}))
        qa["status"] = "repair"
        qa["findings"] = list(qa.get("findings", [])) + [
            "Repair model did not return complete HTML (missing <canvas> or </html>); previous draft was kept."
        ]
        _log("model_text", agent="자동 수정", text="수정 응답이 불완전한 HTML이라 기존 초안을 유지했습니다.")
        return {"qa": qa, "repair_attempts": state.get("repair_attempts", 0) + 1, "stage": "repair"}
    (_workspace(state, _concept(state)) / "draft.html").write_text(html, encoding="utf-8")
    _log("model_text", agent="자동 수정", text="게임 초안을 다시 작성했습니다.")
    return {"game_html": html, "repair_attempts": state.get("repair_attempts", 0) + 1,
            "stage": "repair"}


@traceable(name="package-game", run_type="chain")
def package_node(state: StudioState) -> dict:
    if state.get("qa", {}).get("status") != "pass" or not state.get("design_review"):
        raise RuntimeError("기획 일치 검증 실패: " + "; ".join(state.get("qa", {}).get("findings", [])))
    concept = _concept(state)
    target = _workspace(state, concept)
    target.mkdir(parents=True, exist_ok=True)
    game_path = target / "index.html"
    game_path.write_text(state["game_html"], encoding="utf-8")
    manifest = {
        "concept": concept.model_dump(), "art": state["art"], "qa": state["qa"],
        "trace_notes": state.get("trace_notes", []),
        "implementation_plan": state["implementation_plan"],
        "design_review": state["design_review"],
        "code_model_id": state.get("code_model_id") or state.get("model_id"),
        "generation_mode": "model_generated",
    }
    (target / "production-manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {"game_path": str(game_path)}


@traceable(name="qa-not-passed", run_type="chain")
def abandoned_node(state: StudioState) -> dict:
    """Terminal state for a build that never passed verification.

    The run ends cleanly with a verdict and an inspectable report instead of throwing, but it still
    refuses to publish index.html - shipping an unverified game was never the fallback.
    """
    concept = _concept(state)
    findings = state.get("qa", {}).get("findings", [])
    target = _workspace(state, concept)
    target.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "qa_failed",
        "findings": findings,
        "repair_attempts": state.get("repair_attempts", 0),
        "rethink_cycles": state.get("rethink_cycles", 0),
        "concept": concept.model_dump(),
        "implementation_plan": state.get("implementation_plan", {}),
        "design_review": state.get("design_review", {}),
        "last_guidance": state.get("qa_guidance", ""),
    }
    (target / "qa-report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    # Publish the best draft anyway. A playable game the reviewer can judge beats an empty folder,
    # and whole runs were ending with nothing over a single unmet check. What this does not do is
    # pretend it passed: the run reports QA 미통과 and the manifest carries every open finding.
    html = state.get("game_html") or ""
    if not html:
        draft = target / "draft.html"
        html = draft.read_text(encoding="utf-8") if draft.exists() else ""
    published = None
    if html:
        game_path = target / "index.html"
        game_path.write_text(html, encoding="utf-8")
        manifest = {
            "concept": concept.model_dump(), "art": state.get("art", {}), "qa": state.get("qa", {}),
            "trace_notes": state.get("trace_notes", []),
            "implementation_plan": state.get("implementation_plan", {}),
            "design_review": state.get("design_review", {}),
            "code_model_id": state.get("code_model_id") or state.get("model_id"),
            "generation_mode": "model_generated_qa_failed",
            "qa_outstanding": findings,
        }
        (target / "production-manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        published = str(game_path)

    _log("model_text", agent="QA 검증", text=(
        f"수정·재검토 예산을 모두 사용했습니다. 미해결 {len(findings)}건이 남은 상태로 게임을 배포합니다."
        if published else
        f"수정·재검토 예산을 모두 사용했고 배포할 초안도 없습니다 (미해결 {len(findings)}건)."
    ))
    return {
        "qa_report_path": str(target / "qa-report.json"),
        **({"game_path": published} if published else {}),
        "trace_notes": state.get("trace_notes", [])
        + [f"QA never passed ({len(findings)} findings); published anyway for review."],
    }


_ART_FINDING_MARKERS = ("sprite", "assets/", "drawimage", "asset_name", "이미지", ".png")


def _needs_tools(findings: list[str]) -> bool:
    """Whether fixing this requires the code agent's tools rather than a text rewrite.

    repair_node is a single model call with no tools: it can rewrite HTML and nothing else. Art
    problems - a sprite that was generated but never drawn, an object that still needs one - can
    only be fixed by something that can call generate_comfyui_image and list_game_assets, which is
    the code agent. Sending those to repair first just burns a whole-file rewrite for nothing.
    """
    joined = " ".join(findings).lower()
    return any(marker in joined for marker in _ART_FINDING_MARKERS)


@traceable(name="rejected-by-reviewer", run_type="chain")
def rejected_node(state: StudioState) -> dict:
    return {"trace_notes": state.get("trace_notes", []) + ["Production stopped: design rejected by reviewer."]}


# The production ladder the supervisor walks when a stage simply finished its job. Approval and QA
# are missing on purpose: one comes back with a human decision and the other with a verdict, so the
# supervisor decides those itself rather than looking them up.
_NEXT_AFTER = {
    "idea": "design_document",
    "design_document": "approval",
    # Art direction feeds straight into the code agent. Raster art is not a separate stage that has
    # to finish first: generate_comfyui_image is one of the code agent's own tools, so it generates
    # each object's image at the point it decides that object needs one.
    "art": "code",
    # The write/inspect/repair loop, its call budget and its tool routing all live inside the code
    # agent (code_agent.py), so coding reports back ready for verification.
    "code": "qa",
    "repair": "qa",
}

# Which workers report back to the supervisor instead of ending the run.
WORKERS = ("idea", "design_document", "approval", "art", "code", "qa", "repair")
# Where the supervisor is allowed to send a run.
DESTINATIONS = (*WORKERS, "package", "abandoned", "rejected")


def _production_brief(state: StudioState) -> dict:
    """Open the run: a Deep Agents pass that settles the production plan the planners work from."""
    model_id = state.get("model_id")
    _log("model_call", agent=SUPERVISOR, model=model_id,
         note="서브에이전트(기획·아트·코드·QA)와 프로덕션 계획을 조율하는 중입니다.")
    brief = run_deep_director(
        state["brief"], state.get("use_llm", True), model_id, on_step=_log_supervisor_step
    )
    _log("model_text", agent=SUPERVISOR, text=_trim(brief, 400))
    return {
        "production_brief": brief,
        "trace_notes": state.get("trace_notes", []) + [brief],
    }


def _log_supervisor_step(node: str, tools: str, text: str) -> None:
    """Turn one internal deepagents step into a log line, so the supervisor's delegations are
    visible while they happen instead of being a single opaque multi-minute call."""
    if tools:
        _log("tool_call", agent=SUPERVISOR, name=tools, args={"node": node})
    elif text:
        _log("model_text", agent=SUPERVISOR, text=f"[{node}] {_trim(text, 300)}")


def _affordable_actions(state: StudioState) -> list[str]:
    """The escalation moves the remaining budget still pays for.

    The supervisor picks from this rather than from the full menu, so an over-eager decision cannot
    spend a repair or a rethink cycle the run does not have.
    """
    actions = []
    if state.get("rethink_cycles", 0) < MAX_RETHINK_CYCLES:
        actions.append("code")
        # Re-planning the art costs a model call to hear the same plan back a second time, so the
        # supervisor gets one chance at it per run.
        if not state.get("art_revised", False):
            actions.append("art")
    if state.get("repair_attempts", 0) < MAX_REPAIR_ATTEMPTS:
        actions.append("repair")
    return actions


def _fallback_action(findings: list[str], affordable: list[str]) -> str:
    """What to do when the supervisor names a move the budget cannot pay for."""
    preferred = ("code", "repair") if _needs_tools(findings) else ("repair", "code")
    return next((action for action in preferred if action in affordable), affordable[0])


def _escalate(state: StudioState) -> dict:
    """Decide what a failed verification costs next, and write the instructions to carry it out.

    This is the decision the graph used to make with a hard-coded router plus a separate rethink
    node. Both are gone: the supervisor that planned the production is the one that re-plans it, it
    sees the findings and the art that already exists, and it says which worker gets the work.
    """
    findings = state.get("qa", {}).get("findings", [])
    affordable = _affordable_actions(state)
    if not affordable:
        _log("model_text", agent=SUPERVISOR,
             text=f"수정·재검토 예산을 모두 사용했습니다. 미해결 {len(findings)}건으로 실행을 종료합니다.")
        return {"next_step": "abandoned"}
    model_id = state.get("model_id")
    cycles = state.get("rethink_cycles", 0)
    # Report the model the decision will actually run on, not the run's planning model.
    _log("model_call", agent=SUPERVISOR, model=qa_model_id(model_id),
         note=f"QA 미통과 {len(findings)}건을 검토해 다음 조치를 결정합니다 "
              f"(선택 가능: {', '.join(affordable)} · 재검토 {cycles}/{MAX_RETHINK_CYCLES}).")
    decision = _decide(state, findings, affordable, model_id, cycles)
    action = decision.action if decision.action in affordable else _fallback_action(findings, affordable)
    guidance = decision.instructions.strip() or "QA 지적 사항을 우선순위대로 직접 수정하세요."
    _log("model_text", agent=SUPERVISOR,
         text=f"[{action}] {_trim(decision.reason, 200)}\n{_trim(guidance, 400)}")
    if action == "abandon":
        return {"next_step": "abandoned",
                "trace_notes": state.get("trace_notes", []) + [f"Supervisor abandoned: {decision.reason[:200]}"]}
    if action == "repair":
        return {
            "next_step": "repair",
            "qa_guidance": guidance,
            "trace_notes": state.get("trace_notes", []) + [f"Supervisor chose repair: {guidance[:200]}"],
        }
    cycle = cycles + 1
    return {
        "next_step": "art" if action == "art" else "code",
        "qa_guidance": guidance,
        "rethink_cycles": cycle,
        "art_revision_needed": action == "art",
        # Give the code agent a fresh tool budget and drop the stale html so QA re-reads whatever
        # the agent writes next instead of re-judging the previous draft.
        "repair_attempts": 0,
        "tool_iterations": 0,
        "game_html": "",
        "trace_notes": state.get("trace_notes", []) + [f"Supervisor rethink {cycle}: {guidance[:200]}"],
    }


def _decide(
    state: StudioState, findings: list[str], affordable: list[str], model_id: str | None, cycles: int
) -> SupervisorDecision:
    """Ask the supervisor which move to make, falling back to the cheapest safe one it can afford.

    A schema the model never manages to fill would otherwise end an approved run at its last stage,
    which is the one outcome this pipeline refuses: the findings are already concrete enough to act
    on, so an unanswered decision becomes the heuristic choice plus the findings themselves.
    """
    try:
        return _structured(
            SupervisorDecision,
            SUPERVISOR_ESCALATION_SYSTEM,
            f"Concept: {_concept(state).model_dump_json()}\n"
            f"Contract: {json.dumps(state.get('implementation_plan', {}), ensure_ascii=False)}\n"
            f"QA findings: {json.dumps(findings, ensure_ascii=False)}\n"
            f"Actions still available to you: {json.dumps(affordable)}\n"
            f"Rethink cycles spent: {cycles}/{MAX_RETHINK_CYCLES}\n"
            f"Raster generation enabled: {bool(state.get('generate_images', False))}\n"
            f"Sprites already generated: {json.dumps(_existing_sprites(state), ensure_ascii=False)}\n"
            f"Art plan (candidate objects): {json.dumps(_art(state).asset_plan, ensure_ascii=False)}\n"
            f"Current draft (truncated): {str(state.get('game_html', ''))[:2000]}",
            # Verification and the decision about verification run on the same model, so a run
            # cannot audit on one and escalate on another.
            qa_model_id(model_id),
            on_chunk=_chunk_sink("supervisor"),
            max_tokens=ESCALATION_MAX_TOKENS,
        )
    except (ValidationError, RuntimeError) as error:
        _log("model_text", agent=SUPERVISOR,
             text=f"조치 결정을 구조화하지 못해 기본 경로로 진행합니다: {_trim(str(error), 200)}")
        return SupervisorDecision(
            action=_fallback_action(findings, affordable),
            reason="결정 응답을 받지 못해 지적 사항 기반 기본 경로를 선택했습니다.",
            instructions="\n".join(f"{index}. {finding}" for index, finding in enumerate(findings, 1)),
        )


@traceable(name="supervisor", run_type="chain")
def supervisor_node(state: StudioState) -> dict:
    """The single node that decides what the run does next.

    Every worker returns here with the stage it just finished, and this is the only place an edge
    is chosen - so the whole topology is a hub, not a chain with escalation bolted onto the side.
    Its first visit opens the run with the Deep Agents production pass; later visits are free
    unless there is a real decision to make, and the only real decision is what a failed
    verification costs next.
    """
    _step("supervisor")
    stage = state.get("stage", "")
    if not stage:
        return {**_production_brief(state), "next_step": "idea"}
    if stage == "approval":
        approved = state.get("approval", {}).get("decision") == "approved"
        _log("model_text", agent=SUPERVISOR,
             text="기획이 승인되어 아트 기획부터 제작에 들어갑니다." if approved
             else "기획이 거부되어 제작을 중단합니다.")
        return {"next_step": "art" if approved else "rejected"}
    if stage == "qa":
        if state.get("qa", {}).get("status") == "pass":
            _log("model_text", agent=SUPERVISOR, text="검증을 통과했습니다. 패키징으로 넘깁니다.")
            return {"next_step": "package"}
        return _escalate(state)
    return {"next_step": _NEXT_AFTER[stage]}


def route_from_supervisor(state: StudioState) -> str:
    return state["next_step"]


def build_graph(checkpointer=None):
    graph = StateGraph(StudioState)
    graph.add_node("supervisor", supervisor_node, retry_policy=MODEL_RETRY)
    graph.add_node("idea", idea_node, retry_policy=MODEL_RETRY)
    graph.add_node("design_document", design_document_node, retry_policy=MODEL_RETRY)
    graph.add_node("approval", approval_node)
    graph.add_node("art", art_node, retry_policy=MODEL_RETRY)
    graph.add_node("code", code_node, retry_policy=MODEL_RETRY)
    graph.add_node("qa", qa_node, retry_policy=MODEL_RETRY)
    graph.add_node("repair", repair_node, retry_policy=MODEL_RETRY)
    graph.add_node("package", package_node)
    graph.add_node("abandoned", abandoned_node)
    graph.add_node("rejected", rejected_node)
    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges(
        "supervisor", route_from_supervisor, {name: name for name in DESTINATIONS}
    )
    for worker in WORKERS:
        graph.add_edge(worker, "supervisor")
    graph.add_edge("package", END)
    graph.add_edge("abandoned", END)
    graph.add_edge("rejected", END)
    return graph.compile(checkpointer=checkpointer)


# Standalone script/CLI usage (game_studio.cli) manages its own run via this in-memory-backed graph.
compiled_graph = build_graph(InMemorySaver())

# LangGraph API server usage (`langgraph dev`, LangGraph Platform) is wired through langgraph.json.
# The platform supplies its own persistence and rejects a graph compiled with a custom checkpointer,
# so this variant is intentionally compiled without one.
api_graph = build_graph()
