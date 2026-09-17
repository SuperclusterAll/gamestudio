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
from .godot import (
    check_scripts,
    static_project_qa,
    export_web,
    godot_available,
    godot_version,
    run_project,
    write_launch_script,
)
from .godot_tools import GODOT_TOOLS
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
    run_director,
    stream_turn,
)
from .models import (
    CONTRACT_ITEM_CHARS,
    CONTRACT_MAX_ITEMS,
    ArtDirection,
    QAReport,
    DesignReview,
    GameConcept,
    ImplementationPlan,
    StudioState,
    SupervisorDecision,
    game_output_dir,
    workspace_name,
)
from .prompts import CODE_SYSTEM, GODOT_CODE_SYSTEM, SUPERVISOR_ESCALATION_SYSTEM
from .agent_tools import SPRITE_MANIFEST
from .required_art import (
    asset_slug,
    missing_required,
    missing_required_finding,
    required_assets,
)
from .sprites import DEFAULT_FACING


# Escalation budget for a build that fails verification. The supervisor chooses what to spend it on,
# but it cannot overspend: a cheap text-only repair, then rethink cycles that hand the code agent a
# fresh tool loop, and once both are gone the run ends with a clean verdict.
#
# Two of each, raised from one. The argument for a single rethink was that the second one got spent
# re-litigating an already playable game against reviewer nitpicks - and that argument has since
# been answered at the source: _verdict counts only requirements from our own contract, advisories
# are capped and non-blocking, and keyword guesses no longer travel alone. What survives to block a
# release now is a real defect.
#
# And a real defect is worth another pass, because the shape they take is mechanical. A Godot run
# shipped referencing four resources it never created - three .tscn scenes whose .gd scripts were
# right there, and a sprite under a mangled name - each one a single file away from running. It
# spent its one cycle and ended with all four open. A rethink is still the most expensive thing
# this pipeline can do (a full code agent loop, up to CODE_AGENT_MODEL_CALLS calls at 16k tokens
# each, plus another audit), so this roughly doubles the worst case of a failing run and costs a
# passing run nothing. Both are env-tunable in either direction.
MAX_REPAIR_ATTEMPTS = int(os.getenv("MAX_REPAIR_ATTEMPTS", "2"))
MAX_RETHINK_CYCLES = int(os.getenv("MAX_RETHINK_CYCLES", "2"))
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
DESIGN_REVIEW_MAX_TOKENS = int(os.getenv("DESIGN_REVIEW_MAX_TOKENS", "16000"))
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


# How many trace entries a run keeps. The trail is a human-readable record of what the supervisor
# decided, not a log: it rides along in every checkpoint and in the published manifest, so an
# unbounded list of model output is pure weight. The oldest entries are the ones a reader stops
# caring about first.
TRACE_NOTE_LIMIT = int(os.getenv("TRACE_NOTE_LIMIT", "24"))


def _note_trail(state: StudioState, *entries: str) -> list[str]:
    """Append to the run's trace, keeping only the most recent entries."""
    return [*state.get("trace_notes", []), *entries][-TRACE_NOTE_LIMIT:]


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


def _trace_run(state: StudioState) -> None:
    """Tag this trace with what the run actually is, so traces are comparable across runs.

    Without it every trace looks alike in the LangSmith list: the interesting axes - which engine,
    which models, whether raster art was on - live in state and never reach the trace metadata.
    """
    try:
        from langsmith.run_helpers import get_current_run_tree

        if (run := get_current_run_tree()) is not None:
            run.extra.setdefault("metadata", {}).update({
                "engine": _engine(state),
                "model_id": state.get("model_id", ""),
                "code_model_id": state.get("code_model_id", ""),
                "qa_model_id": qa_model_id(state.get("model_id")),
                "generate_images": bool(state.get("generate_images", False)),
                "rethink_cycles": state.get("rethink_cycles", 0),
            })
    except Exception:
        return


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
        # The games this studio already shipped, so the agent stops re-inventing the last one.
        output_root=state.get("output_dir", ""),
    )
    _log("model_text", agent="기획 Agent", text=f"컨셉 '{concept.title}' 생성 완료")
    return {"concept": concept.model_dump(), "stage": "idea",
            **_named_workspace(state, concept)}


def _named_workspace(state: StudioState, concept: GameConcept) -> dict:
    """Move this run's output folder to "<제목>_<엔진>_<런 id>", now that there is a title.

    Nothing is renamed: the folder is only created by the first thing that writes into it, and
    nothing writes before the art stage. The title simply does not exist when the run starts - the
    server has to name the workspace before anyone has decided what the game is - so this is the
    first moment the name can be right, and the last moment it is free to change.

    A run that was started some other way, or is being revised in a folder that already holds a
    game, keeps the workspace it was given.
    """
    current = Path(state.get("workspace_dir") or "")
    run_id = current.name
    if not current.name or current.exists() or state.get("revision_request"):
        return {}
    named = current.with_name(workspace_name(concept.title, _engine(state), run_id))
    return {"workspace_dir": str(named)} if named != current else {}


def _implementation_plan(concept: GameConcept, brief: str, production_brief: str,
                        model_id: str | None, on_chunk) -> ImplementationPlan:
    """Turn an approved concept into the contract QA will judge the build against.

    Split out of design_document_node so the eval harness can exercise the prompt that
    produces this contract without standing up a graph run - see game_studio.evaluate.
    """
    return _structured(
        ImplementationPlan,
        "You are a game systems engineer. Convert this concept and user request into a concrete, "
        "feasible implementation contract. Preserve the requested genre. Specify unique mechanics, "
        f"win/loss conditions, state transitions and 3-{CONTRACT_MAX_ITEMS} objectively testable "
        "acceptance criteria. Write in Korean. Do not replace it with a generic survival game.\n"
        # The schema refuses more, but a model that planned twelve and had six accepted delivers
        # six arbitrary ones. Told the budget up front, it chooses which six carry the game.
        f"At most {CONTRACT_MAX_ITEMS} mechanics and {CONTRACT_MAX_ITEMS} acceptance tests. This is "
        "the build budget, not a preference: one agent implements every item in a single session, "
        "and a contract that does not fit ships half-built. Choose the ones without which this is "
        "not the game - fold the rest into them or leave them out. A mechanic that only decorates "
        "a mechanic already listed is not a separate item.\n"
        # The failure this is here to stop: a Mario-like contract asked for running acceleration
        # curves, ? blocks, coin 1-ups, timer bonuses, flagpole scoring tiers and damage states.
        # The build spent everything on the list and shipped a game that did not start.
        "mechanics is ordered and the order is the build order. The first two items together must "
        "make a game that is already playable on its own: it starts straight into play with no "
        "menu and no extra click, one control visibly moves something, and there is a way to lose "
        "and a way to start again. Everything after those two is an addition to a game that "
        "already works. Write them so that a build which runs out of budget half way down the "
        "list is still a game somebody can play.\n"
        f"Each mechanic and each acceptance test is one sentence, at most {CONTRACT_ITEM_CHARS} "
        "characters. Say what the mechanic IS and what it costs or earns the player, not how it is "
        "tuned: '방향키로 좌우 이동하고 점프로 적을 밟아 처치한다' is a mechanic, a paragraph of "
        "per-frame acceleration, friction and boost values is a specification and does not belong "
        "in a contract.\n"
        # Acceptance tests that need instrumentation are unverifiable here: nothing in this
        # pipeline reads frame logs, so they are scored by a model reading source and always come
        # back disputed. A measured contract asked for "프레임 단위 로그로 확인한다" six times.
        "Every acceptance test must be checkable by one person playing for sixty seconds and "
        "watching the screen. No frame counts, no coordinate logs, no internal variables, no "
        "measuring what a value is on frame 121. Make the first acceptance test 'the game starts "
        "and can be played': what is on screen at the start, which key does what, and what the "
        "player sees when they lose.\n"
        "Every mechanic must say what the player does and what it costs or earns them, with the "
        "numbers a developer needs. Do not list mechanics that only describe internal machinery.\n"
        "The win condition must reward playing well, not merely finishing: include the measure the "
        "player is scored on (time, score, rank, distance) and how it is shown.\n"
        "Then check the incentives before you answer: if playing slowly, passively or defensively "
        "is the safest way to win, the contract is wrong. Adjust the rules - a clock, a decaying "
        "score, a pursuing rival, a resource that only refills through risk - until cautious play "
        "measurably loses, and make one acceptance test verify exactly that.",
        f"User request: {brief}\nConcept: {concept.model_dump_json()}"
        # Carry the borrowed mechanics forward so the contract stays inside the genre the concept
        # anchored on, instead of drifting once it is turned into systems.
        + (f"\n참조 게임(이 메커닉을 알아볼 수 있게 유지하세요): {'; '.join(concept.reference_games)}"
           if concept.reference_games else "")
        + (f"\nProduction director's brief: {production_brief}" if production_brief else ""),
        model_id,
        on_chunk=on_chunk,
    )


@traceable(name="design-document", run_type="chain")
def design_document_node(state: StudioState) -> dict:
    """Create the reviewable production document before any asset/code work begins."""
    _step("design_document")
    concept = _concept(state)
    model_id = state.get("code_model_id") or state.get("model_id")
    _log("model_call", agent="기획 문서", model=model_id)
    plan = _implementation_plan(concept, state["brief"], state.get("production_brief", ""),
                                model_id, _chunk_sink("design_document"))
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


def _persist_art_plan(workspace: Path, art: ArtDirection) -> None:
    """Keep the art direction on disk beside the game, for review.

    This was a tool the code agent had to be told to call ("generate_asset"), which meant a slot in
    every tool list, its description re-sent on every model call, and a turn spent invoking it -
    for a file write that needs no model at all and that the agent gained nothing from doing.
    """
    try:
        target = workspace / "assets" / "canvas-art-plan.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(art.model_dump(), indent=2, ensure_ascii=False),
                          encoding="utf-8")
    except OSError:
        return


def _established_facing(workspace: Path) -> str:
    """The orientation this run's existing sprites were drawn in.

    A revision happens after a game already exists, so the camera is settled: a top-down build's
    sprites face up and a side-on build's face right. Generating the new object in whatever
    direction the others use keeps the rotation contract consistent across the game, which is
    something the first pass genuinely cannot know and this one can just read.
    """
    manifest = workspace / "assets" / SPRITE_MANIFEST
    try:
        entries = json.loads(manifest.read_text(encoding="utf-8")) if manifest.is_file() else {}
    except (OSError, ValueError):
        return DEFAULT_FACING
    seen = [str(entry.get("facing", "")) for entry in entries.values()
            if isinstance(entry, dict) and entry.get("kind") != "backdrop"]
    directional = [facing for facing in seen if facing in {"right", "up"}]
    return max(set(directional), key=directional.count) if directional else DEFAULT_FACING


def _generate_required_art(state: StudioState, workspace: Path, required: list[str],
                           asset_plan: list[str]) -> list[str]:
    """Produce the sprites this revision made mandatory, here, without spending a model call.

    A first pass leaves generation to the code agent on purpose: it discovers what the game needs
    while building it, and art planned before any code exists would spend the image budget on
    guesses. A revision is the opposite case. The object is already decided - verification named
    it - so there is nothing left to discover, and generating it in the code agent's loop only
    competes with the coding: a measured run spent four of its calls making sprites and then had
    none left to wire them in.

    This is a plain function call, not a tool, so it costs no model call at all.
    """
    from .agent_tools import _generate_comfyui_image

    descriptions = {asset_slug(entry): str(entry) for entry in asset_plan or []}
    facing = _established_facing(workspace)
    payload = {"concept": state["concept"], "output_dir": state.get("output_dir", ""),
               "workspace_dir": str(workspace), "generate_images": True}
    produced = []
    for slug in required:
        _log("tool_call", agent="아트 기획", name="generate_comfyui_image",
             args={"asset_name": slug, "facing": facing})
        result = _generate_comfyui_image(
            prompt=descriptions.get(slug, slug), state=payload, asset_name=slug,
            # Deterministic per object, so a re-run of the same revision is reproducible and two
            # sprites in one revision do not come back as near-identical images.
            seed=int(hashlib.sha256(slug.encode("utf-8")).hexdigest()[:8], 16) % (2**31),
            width=768, height=768, kind="sprite", facing=facing,
        )
        _log("tool_result", agent="아트 기획", name="generate_comfyui_image", text=_trim(result, 200))
        produced.append(slug)
    return produced


@traceable(name="art-direction", run_type="chain")
def art_node(state: StudioState) -> dict:
    """Plan the visual system, and on a revision produce the art that verification said was missing.

    A first pass plans only: palette, canvas effects, a backdrop prompt and a per-object asset_plan
    that is a menu for the code agent, which generates whatever it actually turns out to need while
    it builds. A revision also generates, because by then the objects are not guesses - see
    _generate_required_art.
    """
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
    # On the first pass the asset plan is a menu: the code agent decides which entries are worth
    # spending the image budget on. A revision is not a menu. It exists because QA said art was
    # missing, so the objects it names that still have no file are mandatory from here on - without
    # that, re-planning changed a list nobody was obliged to act on and the same finding came back.
    #
    # A run with raster generation switched off can never satisfy such a mandate, so it is not given
    # one: requiring an image nothing is able to produce would block every remaining cycle on a
    # finding that has no fix.
    can_generate = bool(state.get("generate_images", False))
    required = (required_assets(art.asset_plan, _existing_sprites(state))
                if revising and can_generate else [])
    workspace = _workspace(state, concept)
    _persist_art_plan(workspace, art)
    if required:
        _generate_required_art(state, workspace, required, art.asset_plan)
    _log("model_text", agent="아트 기획",
         text=(f"아트 방향 {'재수립' if revising else '생성'} 완료 (에셋 후보 {len(art.asset_plan)}개)"
               + (f" — 필수 스프라이트 {len(required)}개 생성 완료: {', '.join(required)}"
                  " · 코드 Agent는 이것을 게임에 그려 넣기만 하면 됩니다" if required
                  else " — 실제 이미지 생성은 코드 Agent가 필요할 때 수행")))
    return {
        "art": art.model_dump(),
        "stage": "art",
        "art_revision_needed": False,
        "art_revised": revising or bool(state.get("art_revised")),
        **({"required_assets": required} if revising else {}),
        "trace_notes": _note_trail(state, "Art plan revised after QA; required assets: "
                                   + ", ".join(required) if revising
                                   else "Canvas-first art plan generated."),
    }


# The two engines a run can target. Every planning stage is shared - the concept, the implementation
# contract, the human approval, the art direction - and only the three stages that touch the
# artifact itself read this: building it, verifying it, and packaging it.
HTML5, GODOT = "html5", "godot"


def _engine(state: StudioState) -> str:
    """Which engine this run builds for. Anything unrecognised builds the standalone HTML game, so
    a checkpoint written before this option existed, or a hand-made payload, keeps working."""
    return GODOT if str(state.get("engine", "")).lower() == GODOT else HTML5


def _is_godot(state: StudioState) -> bool:
    return _engine(state) == GODOT


def _required_art_clause(state: StudioState) -> str:
    """The generations this build owes, if any.

    Nothing on a first pass: the asset plan is a menu there and the agent owns the image budget.
    After a re-plan that verification asked for, these are the answer to the finding and skipping
    one is what made the identical finding come back a cycle later.
    """
    required = list(state.get("required_assets") or [])
    if not required:
        return ""
    names = ", ".join(f'"{name}"' for name in required)
    return (
        "\nMANDATORY ART. The art direction was re-planned because verification reported missing "
        f"art, and these objects have no image yet: {names}. Generate every one of them with "
        "generate_comfyui_image(asset_name=\"<name>\", ...) using exactly those names, and draw "
        "each into the game. This is not the optional part of the asset plan - your own "
        "verification tool will refuse to pass while any of them is missing, so do these first."
    )


def _godot_system_prompt(state: StudioState) -> str:
    """The standing contract for a Godot build, including whether raster art is on for this run."""
    art = _art(state)
    if state.get("generate_images", False):
        plan_items = "; ".join(art.asset_plan[:8]) or "(아트 계획에 객체 목록이 없습니다)"
        image_guidance = (
            "\nYou own art generation for this build end to end, and raster generation IS enabled "
            "for this run. generate_comfyui_image writes into res://assets/, so a sprite is "
            "referenced by the path it was created at. Generate one for the player's own object and "
            "its primary opponent or obstacle at minimum, load it into a Sprite2D, and keep a drawn "
            "fallback so a missing texture never leaves an invisible object.\n"
            f"Planned objects: {plan_items}\n"
            "Call list_game_assets first so you never regenerate something that already exists, and "
            "respect the facing each sprite reports - it tells you the exact rotation to apply."
        )
    else:
        image_guidance = (
            "\nRaster image generation is disabled for this run. Build every visual from Godot "
            "nodes and drawing calls - ColorRect, Polygon2D, Line2D, _draw() - and do not "
            "reference any res://assets/ texture."
        )
    return (
        GODOT_CODE_SYSTEM
        + _required_art_clause(state)
        + image_guidance
        + "\nWork through the tools autonomously. First call list_godot_files to see what already "
        "exists. Write complete files with write_godot_file. When a repair is needed, read the "
        "file narrowly before rewriting it: read_godot_file(path, outline=True) gives a map, then "
        "start_line/end_line returns only the section at fault. Everything you read back stays in "
        "this conversation and is re-sent on every later turn, so never pull a whole file when a "
        "section will do. Stop when run_godot_qa passes."
    )


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
        + _required_art_clause(state)
        + image_guidance
        + "\nUse the provided tools autonomously. First call list_game_assets, then "
        "write_game_file. Every write and every repair answers with the static QA verdict on what "
        "you just wrote, so do not spend a turn asking for it - read the verdict in the result and "
        "act on it. run_static_qa is only for re-checking a draft you did not just write.\n"
        "When a repair is needed, read the draft narrowly before rewriting it: "
        "read_game_file(outline=True) gives a line map, then read_game_file(start_line, "
        "end_line) returns just the section at fault. Everything you read back stays in this "
        "conversation and is re-sent on every later turn, so never pull the whole file when a "
        "section will do. Then repair_html. Stop when the verdict says static QA passed."
    )


def _art_brief(art: ArtDirection, engine: str) -> str:
    """The art direction, minus the half that belongs to the other engine.

    This message opens the agent's history and is re-sent on every one of its turns, so anything
    unusable in it is paid for twenty times over. canvas_effects is the largest field the art
    director writes - a measured run put 1,310 characters of ctx.fillRect recipes in it - and on a
    Godot run it describes an API the agent cannot call. The palette and the asset plan are what
    both engines actually build from.
    """
    fields = {"palette": art.palette, "image_prompt": art.image_prompt,
              "asset_plan": art.asset_plan}
    if engine != GODOT:
        fields["canvas_effects"] = art.canvas_effects
    # Compact separators, matching model_dump_json: the default ", " / ": " costs a character
    # per key on a payload that is re-sent every turn.
    return json.dumps(fields, ensure_ascii=False, separators=(",", ":"))


def _code_task(state: StudioState) -> str:
    """The one human turn that starts the agent: the approved design it has to build."""
    concept, art = _concept(state), _art(state)
    task = (
        f"Build this game.\nOriginal request: {state['brief']}\n"
        f"Concept: {concept.model_dump_json()}\n"
        f"Art: {_art_brief(art, _engine(state))}\n"
        f"Implementation contract: {json.dumps(state['implementation_plan'], ensure_ascii=False)}\n"
        "The contract's mechanics are in build order: make the first two playable and saved before "
        "you start the third.\n"
        f"Review comment: {state.get('approval', {}).get('comment', '')}"
    )
    # A revision opens on a game that already shipped, so the player has actually played it. That
    # outranks the contract: the contract is what they were promised, and this is what they think
    # of it. Stated before the QA guidance because a revision run has none.
    if request := state.get("revision_request", "").strip():
        task += (
            "\n\n플레이어가 완성된 게임을 직접 해 보고 보완을 요청했습니다. "
            "이 요청이 위의 계약보다 우선합니다:\n"
            f"{request}\n"
            "작업 폴더에 지금 돌아가는 게임이 이미 있습니다. 처음부터 다시 쓰지 말고 읽어서 "
            "고치세요 — 요청과 무관한 부분은 그대로 두는 것이 이 작업의 요구사항입니다. "
            "요청이 새 메커닉을 뜻하면 더하되, 기존에 돌아가던 것을 망가뜨리지 마세요."
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
    # The engine decides what the agent can touch and what contract it works to. Nothing else about
    # this node changes: the same budgets, the same streaming, the same relay to the dashboard.
    godot = _is_godot(state)
    agent = build_code_agent(
        state.get("code_model_id") or state.get("model_id"),
        GODOT_TOOLS if godot else GAME_TOOLS,
        _godot_system_prompt(state) if godot else _code_system_prompt(state),
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
        "required_assets": list(state.get("required_assets") or []),
        "engine": _engine(state),
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
    # walks into QA and fails on an artifact that was never created.
    workspace = _workspace(state, _concept(state))
    wrote = (workspace / "project.godot").exists() if godot else (workspace / "draft.html").exists()
    if not wrote:
        demand = ("지금 즉시 write_godot_file로 project.godot과 main.tscn, main.gd를 만든 뒤 "
                  "run_godot_qa를 호출하세요." if godot else
                  "지금 즉시 write_game_file로 완성된 게임 HTML 전체를 저장한 뒤 run_static_qa를 호출하세요.")
        _log("model_text", agent="코드 Agent",
             text="도구를 호출하지 않고 종료했습니다. 파일 작성을 요구하며 한 번 더 시도합니다.")
        produced += run(
            _code_task(state)
            + "\n\n이전 시도는 도구를 호출하지 않아 아무 파일도 만들지 못했습니다. 설명하지 말고 "
            + demand
        )
    # Deliberately not returned into graph state. The code agent builds its own payload from
    # scratch on every entry and nothing reads this transcript back, but writing it meant each
    # checkpoint re-serialised every whole-game tool argument the loop had produced - repeatedly,
    # and growing with each rethink. What the run needs from the build is on disk.
    _log("model_text", agent="코드 Agent",
         text=f"빌드 턴 {len(produced)}개 완료. 산출물은 워크스페이스에 기록되었습니다.")
    return {"stage": "code"}


GODOT_SOURCE_LIMIT = int(os.getenv("GODOT_SOURCE_LIMIT", "60000"))


def _godot_source(project: Path) -> str:
    """The whole project as one annotated listing, so the design review can read it the way it
    reads a single HTML file. Paths are kept because a finding without a file is not actionable."""
    parts = []
    for path in _project_source_files(project):
        body = path.read_text(encoding="utf-8", errors="replace")
        parts.append(f"--- res://{path.relative_to(project).as_posix()} ---\n{body}")
    listing = "\n\n".join(parts)
    return listing if len(listing) <= GODOT_SOURCE_LIMIT else listing[:GODOT_SOURCE_LIMIT] + "\n…(생략)"


def _project_source_files(project: Path) -> list[Path]:
    from .godot_tools import _project_files

    return _project_files(project.resolve())


def _godot_report(project: Path) -> QAReport:
    """Deterministic verification for a Godot build: compile every script, then actually run it.

    This is what the HTML path cannot do. A missing scene, a bad node path, a null dereference in
    _ready - the engine reports each with a file and a line, before a model is asked to read
    anything. A machine without the engine gets an honest skip rather than a false pass.
    """
    # Cheapest first, and the only stage that can see what the engine cannot: a project that
    # throws nothing is not the same as a game. A scene whose script is `func _ready(): pass`
    # imports cleanly, runs its five seconds, exits zero - and used to be reported as a pass.
    sprites = sorted(p.name for p in (project / "assets").glob("*.png"))         if (project / "assets").is_dir() else []
    structure = static_project_qa(project, sprites)
    if not structure.ok:
        return QAReport(status="repair", findings=structure.findings,
                        repair_instructions="프로젝트 구조 문제를 먼저 고치세요.")
    if not godot_available():
        return QAReport(status="pass", findings=[
            "Godot 실행 파일을 찾을 수 없어 엔진 검증을 건너뛰었습니다 (구조 검사와 설계 감사만 수행).",
            *structure.findings,
        ], repair_instructions="")
    scripts = check_scripts(project)
    if not scripts.ok:
        return QAReport(status="repair", findings=scripts.findings,
                        repair_instructions="스크립트 컴파일 오류를 먼저 고치세요.")
    played = run_project(project)
    return QAReport(
        status="pass" if played.ok else "repair",
        findings=played.findings + structure.findings,
        repair_instructions="헤드리스 실행에서 보고된 오류를 파일과 줄 번호대로 고치세요.",
    )


@traceable(name="qa-agent", run_type="chain")
def qa_node(state: StudioState) -> dict:
    _step("qa")
    concept = _concept(state)
    godot = _is_godot(state)
    carry: dict = {}
    if godot:
        project = _workspace(state, concept)
        if not (project / "project.godot").is_file():
            raise RuntimeError("코드 Agent가 project.godot을 작성하지 않았습니다. 제작 실패입니다.")
        report = _godot_report(project)
        source = _godot_source(project)
    else:
        if not state.get("game_html"):
            draft = _workspace(state, concept) / "draft.html"
            if not draft.exists():
                raise RuntimeError("코드 Agent가 draft.html을 작성하지 않았습니다. 제작 실패입니다.")
            source = draft.read_text(encoding="utf-8")
        else:
            source = state["game_html"]
        report = static_qa(source)
        carry = {"game_html": source}
    # Backstop for the mandatory generations a re-planned art direction asked for. The code agent's
    # own tool refuses to pass while one is missing, but the agent can also run out of calls and
    # stop - and a build that quietly shipped without the art QA had already asked for is exactly
    # the loop this whole mechanism exists to close.
    required = list(state.get("required_assets") or [])
    if required and (missing := missing_required(required, _workspace(state, concept) / "assets")):
        report.status = "repair"
        report.findings = [missing_required_finding(missing), *report.findings]
        report.repair_instructions = missing_required_finding(missing)
    if report.status != "pass":
        label = "엔진 검증" if godot else "정적 QA"
        _log("model_text", agent="QA 검증",
             text=f"{label} 실패: {_trim('; '.join(report.findings), 300)}")
        return {**carry, "qa": report.model_dump(), "stage": "qa"}
    plan = ImplementationPlan.model_validate(state["implementation_plan"])
    requirements = plan.mechanics + [plan.win_condition, plan.loss_condition] + plan.acceptance_tests
    # The design review sends the whole game source, so it is one of the most expensive calls in
    # the pipeline. Repair and rethink cycles brought QA back here five times in one measured run;
    # re-auditing byte-identical source just pays for the same answer again.
    fingerprint = hashlib.sha256(source.encode("utf-8")).hexdigest()
    if state.get("design_review") and state.get("design_review_hash") == fingerprint:
        _log("model_text", agent="QA 검증", text="소스가 직전 감사와 동일해 설계 감사를 건너뜁니다.")
        return {**carry, "qa": state["qa"], "design_review": state["design_review"],
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
        f"Concept: {concept.model_dump_json()}\n"
        f"Requirements: {json.dumps(requirements, ensure_ascii=False)}\n"
        f"{'GODOT PROJECT SOURCE' if godot else 'HTML'}: {source}",
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
    return {**carry, "qa": report.model_dump(), "design_review": review.model_dump(),
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


# Manifest keys that are written by somebody other than the stage that builds the manifest, and so
# would be destroyed by rebuilding it. Packaging composes a fresh dict from the run's own state and
# writes it over whatever was there, which quietly erased both of these:
#
#   adoption  - written by the dashboard when a revision starts, so a successful rework deleted the
#               record of having been reworked. Every finished revision lost its own evidence.
#   usage     - the run's token and call totals, which only the server sees (the graph never holds
#               them) and which are therefore added after this file is written.
#
# The run's own fields still win: this only restores what nothing in this run produced.
_CARRIED_MANIFEST_KEYS = ("adoption", "usage")


def _write_manifest(target: Path, manifest: dict) -> None:
    """Write the production manifest, keeping the fields this run did not author."""
    path = target / "production-manifest.json"
    try:
        previous = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        previous = {}
    for key in _CARRIED_MANIFEST_KEYS:
        if key not in manifest and isinstance(previous, dict) and key in previous:
            manifest[key] = previous[key]
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def _package_godot(state: StudioState, target: Path, manifest: dict) -> dict:
    """Everything a finished Godot folder needs, whether or not verification was satisfied.

    Shared because the two exits used to diverge, and the divergence was invisible from the
    outside: a run that passed got a web build attempt and a run that did not was never even
    offered one. The folder looked identically packaged - project, launcher, manifest all present -
    so the missing build read as "export templates are absent" rather than "this path never asked".

    The project IS the deliverable either way. A reviewer can only judge a game they can start, and
    an unmet check is not a reason to withhold the thing they are meant to review.
    """
    manifest["godot_version"] = godot_version()
    manifest["project_path"] = str(target / "project.godot")
    produced: dict = {"godot_project_path": str(target / "project.godot")}
    # A double-clickable launcher, so the finished folder plays without the dashboard, this
    # repository, or a Python environment. The dashboard's run button executes this same file.
    #
    # Everything below is wrapped because packaging runs last, on a project that is already
    # complete on disk. There is no failure here worth converting a finished game into a failed
    # run: a missing engine, a locked file, a full disk or an export that dies in a way nobody
    # anticipated all mean "no launcher" or "no web build", which the dashboard already knows how
    # to show. The deliverable is the project, and it is already delivered.
    try:
        launcher = write_launch_script(target)
    except OSError as error:
        launcher, note = None, f"실행 스크립트를 쓰지 못했습니다: {error}"
        _log("model_text", agent="패키징", text=note)
    manifest["launch_script"] = str(launcher) if launcher else ""
    if launcher:
        produced["launch_script_path"] = str(launcher)
    # A web build is a bonus that lets the dashboard embed the game, and it needs export templates
    # Godot only ships inside a ~1GB all-platform archive - so its absence is reported, never
    # treated as a failure.
    try:
        exported, note = export_web(target, target / "build")
    except Exception as error:  # see above: never fail a run over a project already delivered
        exported, note = False, f"웹 빌드 중 예기치 못한 오류: {type(error).__name__}: {error}"
    manifest["web_export"] = {"ok": exported, "detail": note}
    _log("model_text", agent="패키징", text=note)
    if exported:
        produced["game_path"] = str(target / "build" / "index.html")
    return produced


@traceable(name="package-game", run_type="chain")
def package_node(state: StudioState) -> dict:
    if state.get("qa", {}).get("status") != "pass" or not state.get("design_review"):
        raise RuntimeError("기획 일치 검증 실패: " + "; ".join(state.get("qa", {}).get("findings", [])))
    concept = _concept(state)
    target = _workspace(state, concept)
    target.mkdir(parents=True, exist_ok=True)
    manifest = {
        "engine": _engine(state),
        "concept": concept.model_dump(), "art": state["art"], "qa": state["qa"],
        "trace_notes": state.get("trace_notes", []),
        "implementation_plan": state["implementation_plan"],
        "design_review": state["design_review"],
        "code_model_id": state.get("code_model_id") or state.get("model_id"),
        "generation_mode": "model_generated",
    }
    produced: dict = {}
    if _is_godot(state):
        produced = _package_godot(state, target, manifest)
    else:
        game_path = target / "index.html"
        game_path.write_text(state["game_html"], encoding="utf-8")
        produced["game_path"] = str(game_path)
    _write_manifest(target, manifest)
    return produced


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
    manifest = {
        "engine": _engine(state),
        "concept": concept.model_dump(), "art": state.get("art", {}), "qa": state.get("qa", {}),
        "trace_notes": state.get("trace_notes", []),
        "implementation_plan": state.get("implementation_plan", {}),
        "design_review": state.get("design_review", {}),
        "code_model_id": state.get("code_model_id") or state.get("model_id"),
        "generation_mode": "model_generated_qa_failed",
        "qa_outstanding": findings,
    }
    published: dict = {}
    if _is_godot(state) and (target / "project.godot").is_file():
        # Packaged exactly as a passing run is - launcher and web build included. Withholding the
        # build here only meant the reviewer could not open the game in the browser to see the very
        # findings they were asked to judge.
        published = _package_godot(state, target, manifest)
    elif not _is_godot(state):
        html = state.get("game_html") or ""
        if not html:
            draft = target / "draft.html"
            html = draft.read_text(encoding="utf-8") if draft.exists() else ""
        if html:
            game_path = target / "index.html"
            game_path.write_text(html, encoding="utf-8")
            published["game_path"] = str(game_path)
    if published:
        _write_manifest(target, manifest)

    _log("model_text", agent="QA 검증", text=(
        f"수정·재검토 예산을 모두 사용했습니다. 미해결 {len(findings)}건이 남은 상태로 게임을 배포합니다."
        if published else
        f"수정·재검토 예산을 모두 사용했고 배포할 초안도 없습니다 (미해결 {len(findings)}건)."
    ))
    return {
        "qa_report_path": str(target / "qa-report.json"),
        **published,
        "trace_notes": _note_trail(state, f"QA never passed ({len(findings)} findings); published anyway for review."),
    }


_ART_FINDING_MARKERS = ("sprite", "assets/", "drawimage", "asset_name", "이미지", ".png")
# Findings that say art is absent, rather than present-but-wrong. The distinction decides where the
# run goes next: a sprite that exists and is not drawn is a coding problem, but an object the game
# needs and nobody ever planned is a planning problem, and only the art stage can add it to the
# list the code agent works from.
_MISSING_ART_MARKERS = (
    "스프라이트가 생성되지 않았습니다", "생성되지 않", "이미지가 없", "스프라이트가 없",
    "missing sprite", "no sprite", "not generated", "never generated", "needs a sprite",
)


def _needs_new_art(findings: list[str]) -> bool:
    """Whether the findings are asking for art that does not exist yet."""
    joined = " ".join(findings).lower()
    return any(marker.lower() in joined for marker in _MISSING_ART_MARKERS)


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
    return {"trace_notes": _note_trail(state, "Production stopped: design rejected by reviewer.")}


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
    brief = run_director(
        state["brief"], state.get("use_llm", True), model_id,
        on_step=_log_supervisor_step, engine=_engine(state),
    )
    if brief:
        _log("model_text", agent=SUPERVISOR, text=_trim(brief, 400))
    return {
        "production_brief": brief,
        # Trimmed: the full brief is already kept in production_brief and the manifest, and an
        # unbounded log of verbatim model output is not a trace.
        "trace_notes": _note_trail(state, f"Director brief: {_trim(brief, 300)}"),
    }


def _log_supervisor_step(node: str, tools: str, text: str) -> None:
    """Relay whatever the director had to say about itself - in practice only a failure, since
    the brief itself is logged by the caller."""
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
    # repair is one model call that answers with the whole game as a single blob of text. That is a
    # complete artifact for HTML and a meaningless one for Godot, where the game is project.godot
    # plus scenes plus scripts - so a Godot build is repaired by the only thing that can write more
    # than one file, the code agent's tool loop.
    if not _is_godot(state) and state.get("repair_attempts", 0) < MAX_REPAIR_ATTEMPTS:
        actions.append("repair")
    return actions


def _fallback_action(findings: list[str], affordable: list[str]) -> str:
    """What to do when the supervisor names a move the budget cannot pay for.

    Art that does not exist leads the order. Sending that to the code agent asks it to draw an
    object nothing ever planned, and sending it to repair - a model call with no tools at all -
    cannot produce an image under any circumstances; only the art stage can add the object to the
    list, and it is the art stage that makes the generation mandatory afterwards.
    """
    if _needs_new_art(findings):
        preferred = ("art", "code", "repair")
    elif _needs_tools(findings):
        preferred = ("code", "repair", "art")
    else:
        preferred = ("repair", "code", "art")
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
    # Art that does not exist is routed by the findings, not by preference. A model asked to fix
    # "the enemy has no sprite" will reliably choose to write code - it is the move that always
    # looks productive - and the code agent then draws a rectangle for an object nothing planned,
    # so the same finding returns next cycle. Only the art stage can add the object to the list,
    # and only a re-plan makes generating it mandatory afterwards.
    if action != "art" and "art" in affordable and _needs_new_art(findings):
        _log("model_text", agent=SUPERVISOR,
             text=f"지적 사항이 '없는 그림'을 요구하므로 [{action}] 대신 아트 재수립으로 보냅니다.")
        action = "art"
    guidance = decision.instructions.strip() or "QA 지적 사항을 우선순위대로 직접 수정하세요."
    _log("model_text", agent=SUPERVISOR,
         text=f"[{action}] {_trim(decision.reason, 200)}\n{_trim(guidance, 400)}")
    if action == "abandon":
        return {"next_step": "abandoned",
                "trace_notes": _note_trail(state, f"Supervisor abandoned: {decision.reason[:200]}")}
    if action == "repair":
        return {
            "next_step": "repair",
            "qa_guidance": guidance,
            "trace_notes": _note_trail(state, f"Supervisor chose repair: {guidance[:200]}"),
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
        "trace_notes": _note_trail(state, f"Supervisor rethink {cycle}: {guidance[:200]}"),
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
    _trace_run(state)
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
