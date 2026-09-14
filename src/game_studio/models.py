"""Schemas shared by the specialist agents and LangGraph state."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


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


class ImplementationPlan(BaseModel):
    genre: str
    mechanics: list[str] = Field(min_length=3, max_length=8)
    win_condition: str
    loss_condition: str
    state_transitions: list[str] = Field(min_length=3)
    acceptance_tests: list[str] = Field(min_length=3, max_length=8)


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
    qa: dict[str, Any]
    # Written when a build never passes verification: the run still ends cleanly, with the draft
    # and this report kept for inspection instead of publishing an unverified game.
    qa_report_path: str
    # Hash of the source the last design review judged, so identical HTML is not re-audited.
    design_review_hash: str
    # The supervisor's fix instructions after a failed QA, fed into whichever worker it delegated
    # the repair to.
    qa_guidance: str
    rethink_cycles: int
    # Whether the supervisor asked for the art direction to be re-planned, and whether that
    # has already happened once (it is not repeated every cycle).
    art_revision_needed: bool
    art_revised: bool
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
