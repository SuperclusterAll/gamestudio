"""Schemas shared by the specialist agents and LangGraph state."""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field, StringConstraints, field_validator

# What a reference image is allowed to carry into a run.
#
# A person can show what they want far faster than they can write it, and the brief field is the
# narrowest part of this whole pipeline - one paragraph that then steers planning, art and code. A
# screenshot of the arcade game they have in mind settles the camera, the palette, the object list
# and the screen layout in one upload.
#
# It is read ONCE, at the start of the run, and what travels onward is TEXT. That is deliberate and
# it is the same decision as the art memory (see DESIGN-DECISIONS §18): the planning and code agents
# are text models, so an image kept as an image would have to be re-read by a vision call at every
# stage that wanted it. Describing it once costs one call and every later stage reads words.
REFERENCE_NOTE_CHARS = 200
ReferenceNote = Annotated[str, StringConstraints(min_length=1, max_length=REFERENCE_NOTE_CHARS)]


class ReferenceSketch(BaseModel):
    """What an uploaded reference image says about the game the person has in mind."""

    view: Literal["side-scrolling", "top-down", "isometric", "fixed-screen", "first-person"] = (
        Field(description="the camera this screen is drawn from"))
    genre_guess: ReferenceNote = Field(description="the genre this screen belongs to")
    style_token: ReferenceNote = Field(
        description="the art style as ONE prompt clause, in English - this is stamped on every "
                    "generated asset, so it describes how things are drawn, not what they are")
    palette: list[ReferenceNote] = Field(
        min_length=2, max_length=8,
        description="colour NAMES in English, never hex - a diffusion model follows 'warm orange' "
                    "and ignores '#E8912D'")
    objects: list[ReferenceNote] = Field(
        min_length=2, max_length=8,
        description='one "name: what it is and how it looks" per distinct game object, in English. '
                    "Describe each ALONE - no background, no floor, no shadow - because these "
                    "become sprite prompts and anything behind the object is cut out with it")
    # What the picture is actually being asked for. The art fields above decide how things LOOK;
    # these three decide how the game is BUILT - and they are the reason a screenshot beats a
    # paragraph. A level's shape, the path a player takes through it and where the enemies sit are
    # all obvious at a glance and laborious to write down, which is exactly the gap an upload fills.
    level_structure: list[ReferenceNote] = Field(
        min_length=1, max_length=6,
        description="the traversable layout, one line per element: platform rows and their gaps, "
                    "walls and bounds, ladders, pits, doors. Say WHERE on the screen each sits")
    player_flow: ReferenceNote = Field(
        description="where the player starts, how they move through the level, and what makes it "
                    "progress or end")
    enemy_placement: ReferenceNote = Field(
        description="how many enemies, where they sit or spawn, and how they move through the "
                    "level")

    def as_brief(self) -> str:
        """The sketch as the paragraph the planning and code agents actually read.

        Structure first. The look matters to one stage; the layout, the flow and the placement
        matter to the plan, to the code, and to every check that asks whether the game was built
        the way it was designed.
        """
        return (
            "[참조 이미지 분석]\n"
            f"- 시점: {self.view} · 장르: {self.genre_guess}\n"
            "- 맵 구조:\n" + "\n".join(f"    · {entry}" for entry in self.level_structure) + "\n"
            f"- 플레이 흐름: {self.player_flow}\n"
            f"- 적 배치: {self.enemy_placement}\n"
            f"- 화풍: {self.style_token}\n"
            f"- 색: {', '.join(self.palette)}\n"
            "- 등장 객체:\n" + "\n".join(f"    · {entry}" for entry in self.objects)
        )


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
    # The one art style every asset in this game shares, written once and then appended verbatim to
    # every image prompt. Measured across 29 real prompts from five runs, the same run produced
    # "retro 8-bit pixel art style", "cartoon game boss style" and "cartoon platformer game sprite
    # style" - one run's walking frames came back in a different style from the character they
    # animate. The agent rewrote the style from scratch for every sprite, so it is taken away from
    # the sprite and fixed here.
    style_token: str = Field(default="", max_length=120)
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

# Phrases that assert somebody watched the game. The design review reads source and is told never to
# claim it ran anything, so a criterion written this way can be neither confirmed nor refuted - and
# an unfalsifiable criterion is always passed. Measured: one plan's sixteen requirements all passed
# on a game whose walk cycle turned the character around halfway through.
UNVERIFIABLE_BY_READING = ("확인된다", "확인할 수 있다", "확인됩니다", "육안", "보인다", "보입니다",
                           "느껴진다", "느껴집니다", "체감", "플레이해 보면")


# Wording that says how this run was STARTED rather than what it is. The dropdown's own labels, in
# the forms a model writes them back: "자동 기획 러너", "자동 기획 - 미로 도주 (Maze Escape)",
# "자동 기획 / 단일 화면 플랫폼 아케이드 (버블 보블 류)".
_PROVENANCE_PREFIX = re.compile(r"^\s*(?:자동\s*기획|커스텀|auto[- ]?plan(?:ned)?|custom)\s*[-–—/:·]?\s*")
# A qualifier this run added to the genre name: "횡스크롤 플랫포머 (Super Mario Bros 스타일)". True of
# that game and of nothing else, which is the opposite of what a key is for.
_GENRE_QUALIFIER = re.compile(r"\s*[(（][^)）]*[)）]\s*$")
# Long enough for every label in GENRE_REFERENCES, short enough that a sentence cannot become a key.
GENRE_KEY_LIMIT = 40


def canonical_genre(text: str) -> str:
    """The genre alone, with this run's provenance and its qualifiers taken off.

    The genre is a LOOKUP KEY: art_memory filters on `{"genre": genre}` by exact string equality,
    so two runs of the same genre only learn from each other if they spell it identically. They did
    not. Measured on the live store - 212 sprites under 21 different keys, seven of the top twelve
    carrying either the dropdown's "자동 기획" or a parenthetical true of one run only. "횡스크롤
    플랫포머" and "횡스크롤 플랫포머 (Super Mario Bros 스타일)" were 54 sprites that could not see
    each other, and every auto-planned run was a key of its own with nothing to match.

    Idempotent, and it never returns something it was not given: if stripping leaves nothing, the
    original stands rather than inventing a label.
    """
    cleaned = " ".join(str(text or "").split())
    stripped = _GENRE_QUALIFIER.sub("", _PROVENANCE_PREFIX.sub("", cleaned)).strip()
    # A bare "자동 기획 (Auto-Runner)" leaves the qualifier as the only content there was.
    if not stripped:
        stripped = _PROVENANCE_PREFIX.sub("", cleaned).strip(" ()（）")
    return (stripped or cleaned)[:GENRE_KEY_LIMIT].strip()


class ImplementationPlan(BaseModel):
    # Described, because a field with no description gets whatever format the model feels like -
    # the same way acceptance_tests did below. Asked for as a key, and normalised as one too:
    # this is a lookup value, and "강제는 스키마와 코드로, 부탁은 프롬프트로".
    genre: str = Field(
        description=("the genre alone, as a short reusable label of two to four words "
                     "(\"횡스크롤 플랫포머\", \"퍼즐\", \"로그라이크\"). Not how this run was "
                     "started, not this game's title, and no parenthetical about this particular "
                     "game - it is a key other runs have to match exactly."))

    @field_validator("genre")
    @classmethod
    def strip_provenance(cls, value: str) -> str:
        return canonical_genre(value)

    # Ordered: this is the build order, and the first two items have to be the playable core. A
    # build that runs out of budget half way down the list must still be a game you can play.
    mechanics: list[ContractItem] = Field(min_length=3, max_length=CONTRACT_MAX_ITEMS)
    win_condition: str
    loss_condition: str
    # Not capped: these are the states the game moves between, not work the agent has to do. Four
    # or seven of them costs the build nothing.
    state_transitions: list[str] = Field(min_length=3)
    # Written to be checkable by READING, because reading is all the reviewer can do.
    #
    # This field had no description at all, so the planning model chose its own format - and chose
    # the natural one: what a player would observe. A real plan asked for "60~90초 구간에서 가만히
    # 서 있으면 30초 이내에 몬스터 6마리에 둘러싸여 목숨을 모두 잃고 게임 오버가 되는 것이
    # 확인된다", and "육안으로 확인된다".
    #
    # Nobody confirms those. The design review reads source - its own prompt tells it never to claim
    # it ran the game, which is honest because it cannot - so an observational criterion is one it
    # can neither verify nor refute, and it passes. Sixteen requirements, sixteen passes, on a game
    # whose walk cycle turned the character around halfway through.
    #
    # Making them source-checkable does not find broken games; only running one does that. What it
    # buys is a verdict that MEANS something - a criterion the reviewer can falsify is one it can
    # reject with evidence - and a plan that has to name its numbers is a better specification for
    # the code agent besides.
    acceptance_tests: list[ContractItem] = Field(
        min_length=3, max_length=CONTRACT_MAX_ITEMS,
        description=(
            "Pass/fail criteria that can be checked by READING THE SOURCE, because that is what the "
            "reviewer does - it never runs the game. Name the numbers, the state transitions and "
            "the functions: 'spawn count is 2/4/6 at the 30s and 60s boundaries', 'the collision "
            "handler decrements lives and a life count of 0 enters the game-over state', 'the "
            "combo multiplier is 1/2/4/8 and a 1s idle timer resets it'. Never write what a player "
            "would SEE - no 확인된다, 육안으로, 보인다, 느껴진다. Those can be neither verified nor "
            "refuted from the code, so they are always passed and test nothing."
        ),
    )

    @field_validator("acceptance_tests")
    @classmethod
    def must_be_checkable_by_reading(cls, tests: list[str]) -> list[str]:
        """Enforced rather than requested, the way the contract's size limits are.

        The description above is advice, and advice is what the previous version of this field
        relied on - it had none at all, and the model wrote observational criteria every time. A
        schema is not advice: a violation comes back as a validation error, _structured re-asks with
        the complaint attached, and the plan that reaches the build is one the reviewer can actually
        judge.

        Narrow on purpose. "화면에 표시된다" passes - a draw call is in the source, and refusing
        every sentence about the screen would rule out most of what a game's contract is. What is
        refused is the claim that somebody LOOKED, which is the one thing nobody did.
        """
        for test in tests:
            if found := next((mark for mark in UNVERIFIABLE_BY_READING if mark in test), ""):
                raise ValueError(
                    f"'{found}'은(는) 소스를 읽어서 판정할 수 없습니다. 검수자는 게임을 실행하지 "
                    f"못하므로, 플레이어가 보는 것이 아니라 코드에서 확인할 수 있는 값·상태 전이·"
                    f"함수로 다시 쓰세요: {test[:60]}"
                )
        return tests


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
    # The uploaded reference, already turned into words. See ReferenceSketch.
    reference: dict[str, Any]
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
    # How many frames one character is drawn in. 3 animates; 1 means every character is a single
    # still, the same as a wall or a coin. Chosen on the launch form because it is a taste and a
    # budget decision, not something the pipeline can work out - see sprites.DEFAULT_ANIMATION_FRAMES.
    animation_frames: int
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


# Where the studio keeps its own working state - the checkpoint database and the art memory - as
# opposed to GAME_OUTPUT_DIR, which holds the games it delivers.
#
# The two were mixed together in the output folder, which put a 249MB SQLite file and a vector store
# in the directory a person browses to find their games. They are not deliverables: they are this
# installation's private state, they belong with the code that reads them, and nothing outside this
# process ever opens them. Created on first use, so a fresh clone needs no setup step, and ignored
# by git so it never travels.
def project_data_dir() -> Path:
    """The directory this installation keeps its own state in."""
    if override := os.getenv("STUDIO_DATA_DIR", "").strip():
        return Path(override)
    if getattr(sys, "frozen", False):
        # Beside the executable, because a bundled app has no source tree to sit in.
        return Path(sys.executable).resolve().parent / "data"
    return Path(__file__).resolve().parents[2] / "data"


def game_output_dir(root: Path, title: str) -> Path:
    safe = "".join(char.lower() if char.isalnum() else "-" for char in title).strip("-")
    return root / (safe or "generated-game")


# How much of the title goes in the folder name. Long enough to recognise the game, short enough
# that the path stays workable on Windows, where the whole thing still has to fit in 260 characters
# alongside res://assets/... and a build directory.
WORKSPACE_TITLE_CHARS = 40


def workspace_name(title: str, engine: str, run_id: str) -> str:
    """The output folder's name: "<제목>_<엔진>_<런 id>".

    Folders used to be named by run id alone, which meant a directory of games was a directory of
    hex strings - you could not tell a Godot project from a Canvas page, or one game from another,
    without opening each manifest.

    The run id stays, and stays last. It is the part every lookup resolves by and the only part a
    request ever supplies, so keeping it a fixed-width token at a known position means a folder can
    be *found* by its id rather than built from a title - and a title that reached a path from a URL
    would be a path-traversal surface. Korean survives the slug: str.isalnum() is true for Hangul.
    """
    slug = "".join(char.lower() if char.isalnum() else "-" for char in title).strip("-")
    slug = slug[:WORKSPACE_TITLE_CHARS].strip("-") or "game"
    return f"{slug}_{engine or 'html5'}_{run_id}"
