"""Schemas shared by the specialist agents and LangGraph state."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field, StringConstraints


class GameConcept(BaseModel):
    title: str = Field(description="Short, browser-game-safe title")
    elevator_pitch: str
    player_goal: str
    controls: list[str] = Field(min_length=2, max_length=5)
    core_loop: list[str] = Field(min_length=3, max_length=5)
    difficulty_curve: str
    visual_direction: str
    # Real games whose proven mechanics this design borrows, each with what is taken from it.
    # Grounds the concept in a loop that is known to work instead of inventing one from scratch;
    # the borrowed mechanic is the point, never the name, characters or art.
    reference_games: list[str] = Field(default_factory=list, max_length=3)


class ArtDirection(BaseModel):
    # Collections a model legitimately leaves out when it has nothing to put in them. Without
    # defaults, "no extra effects" comes back as a missing field and fails validation instead.
    palette: dict[str, str] = Field(default_factory=dict)
    image_prompt: str
    asset_plan: list[str] = Field(default_factory=list)
    canvas_effects: list[str] = Field(default_factory=list)


# How many items the implementation contract may carry per list, and the build budget written as a
# number. The code agent builds the whole game inside CODE_AGENT_MODEL_CALLS model calls; the first
# measured Tetris contract came back at the old ceiling of eight mechanics and eight acceptance
# tests and there is no version of that build that finishes in twenty calls.
#
# Enforced by the schema rather than asked for in the prompt, because a ceiling the model may
# quietly exceed is not a budget. Held here so the three places that have to agree - this schema,
# the prompt that fills it, and the eval that scores it - read the same value instead of each
# carrying its own literal. They disagreed before: the schema allowed eight and the eval wanted
# six, so a contract could be valid and still be scored as oversized every single time.
#
# Eight, tracking CODE_AGENT_MODEL_CALLS: the number only means anything as a ratio to the calls
# available to implement it. At 6 items against 20 calls a mechanic got three calls to be written,
# wired and verified; at 8 against 50 it gets six. The count was never the thing that made those
# games unplayable - the order and the size of each item were, and both are still enforced below.
CONTRACT_MAX_ITEMS = 8

# And how long one item may be. A measured Mario-like contract came back with eight mechanics
# averaging 190 characters, the longest 240 - frame-by-frame tuning tables:
#
#   【달리기 & 가속】... 수평 속도가 0→최대 4px/프레임까지 0.4px/프레임²로 선형 가속된다. 키를 떼면
#   0.3px/프레임²로 감속(마찰). 최대 속도 도달 시 '달리기 상태' 플래그가 ON ...
#
# That is a specification, not a contract item, and it is what made the game unplayable: the agent
# spent its budget on ? blocks and flagpole scoring tiers and never produced a build that started.
# The cap is a proxy for the real rule - say what the mechanic IS, not how it is tuned - and a
# proxy the schema can actually hold the model to.
CONTRACT_ITEM_CHARS = 160
# Only the ceiling is meaningful. A floor would reject nothing a real plan produces and does
# reject the short placeholders the routing tests are built from - min_length=1 keeps an
# empty string out and stays out of the way otherwise.
ContractItem = Annotated[str, StringConstraints(min_length=1, max_length=CONTRACT_ITEM_CHARS)]


class ImplementationPlan(BaseModel):
    genre: str
    # Ordered: this is the build order, and the first two items have to be the playable core. A
    # build that runs out of budget half way down the list must still be a game you can play.
    mechanics: list[ContractItem] = Field(min_length=3, max_length=CONTRACT_MAX_ITEMS)
    win_condition: str
    loss_condition: str
    # Not capped: these are the states the game moves between, not work the agent has to do. Four
    # or seven of them costs the build nothing.
    state_transitions: list[str] = Field(min_length=3)
    acceptance_tests: list[ContractItem] = Field(min_length=3, max_length=CONTRACT_MAX_ITEMS)


class RequirementCheck(BaseModel):
    requirement: str
    passed: bool
    # Only a rejection has to be justified. Demanding ten characters of prose for every requirement
    # made the audit's output grow with the contract for no gain - a passing check's evidence is
    # read by nobody and was the bulk of what pushed this answer past its token budget and into a
    # truncated, unparseable array. A failure still has to say what is actually missing.
    evidence: str = ""


class DesignReview(BaseModel):
    checks: list[RequirementCheck]
    # "Other bugs I noticed" - an empty answer is the common case, and a reviewer model with nothing
    # to add simply omits the key. QAReport.findings already defaulted; this one did not, so a clean
    # review raised ValidationError and killed the QA node. These are advisory: they are notes about
    # a game that may well be perfectly playable, not unmet terms of the approved contract, so the
    # QA node records them without letting them hold a release (see ADVISORY_FINDING_LIMIT).
    findings: list[str] = Field(default_factory=list)


class QAReport(BaseModel):
    status: Literal["pass", "repair"]
    findings: list[str] = Field(default_factory=list)
    repair_instructions: str = ""


class SupervisorDecision(BaseModel):
    """What the supervisor decided to do about a build that failed verification.

    repair is one text-only rewrite, code hands the code agent a fresh tool loop, art re-plans the
    asset list before coding again, and abandon publishes the draft with its findings open.
    """

    action: Literal["repair", "code", "art", "abandon"]
    reason: str
    instructions: str


class StudioState(TypedDict, total=False):
    # The stage that most recently reported back to the supervisor, and where the supervisor decided
    # to send the run next. Every edge out of the supervisor reads next_step, so these two fields
    # are the whole routing contract: a worker says what it finished, the supervisor says what
    # happens now. An absent stage means nothing has run yet.
    stage: str
    next_step: str
    brief: str
    # Which engine this run builds for: "html5" for a standalone Canvas page, "godot" for a Godot
    # project. Chosen once when the run starts and read by the code, QA and packaging stages - the
    # two paths share every planning stage and diverge only where the artifact itself differs.
    engine: str
    output_dir: str
    workspace_dir: str
    use_llm: bool
    model_id: str
    code_model_id: str
    implementation_plan: dict[str, Any]
    design_review: dict[str, Any]
    generate_images: bool
    concept: dict[str, Any]
    design_document: dict[str, Any]
    approval: dict[str, Any]
    art: dict[str, Any]
    game_html: str
    game_path: str
    # Where the finished Godot project lives. Set instead of (or alongside) game_path on a Godot
    # run: the project is always the deliverable, and game_path only appears when export templates
    # were available to also produce a web build the dashboard can embed.
    godot_project_path: str
    # The generated run.bat. Written for a Godot run so the output folder plays on its own, and
    # wired to the dashboard's run button.
    launch_script_path: str
    qa: dict[str, Any]
    # Written when a build never passes verification: the run still ends cleanly, with the draft
    # and this report kept for inspection instead of publishing an unverified game.
    qa_report_path: str
    # Hash of the source the last design review judged, so identical HTML is not re-audited.
    design_review_hash: str
    # The supervisor's fix instructions after a failed QA, fed into whichever worker it delegated
    # the repair to.
    qa_guidance: str
    # What the player asked for after playing the finished game. Set only on a revision run, which
    # re-enters at the code agent with the concept, contract and workspace of a run that already
    # shipped - so this is the one instruction that outranks everything already built.
    revision_request: str
    rethink_cycles: int
    # Whether the supervisor asked for the art direction to be re-planned, and whether that
    # has already happened once (it is not repeated every cycle).
    art_revision_needed: bool
    art_revised: bool
    # Sprites a re-planned art direction made mandatory. Empty on a first pass, where asset_plan is
    # a menu the code agent may spend its image budget on as it sees fit; populated only after
    # verification asked for the re-plan, which is when the generation stops being optional.
    required_assets: list[str]
    repair_attempts: int
    trace_notes: list[str]
    # The supervisor's production brief, fed into the idea and design-document prompts so that pass
    # actually shapes the game instead of only being archived in the manifest.
    production_brief: str
    # One conversation drives production: the code agent's tool loop, which includes image
    # generation. There is no separate art-agent message channel.
    messages: Annotated[list[BaseMessage], add_messages]
    tool_iterations: int


def game_output_dir(root: Path, title: str) -> Path:
    safe = "".join(char.lower() if char.isalnum() else "-" for char in title).strip("-")
    return root / (safe or "generated-game")
