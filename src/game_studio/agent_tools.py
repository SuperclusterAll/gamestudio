"""State-aware tools used by the autonomous Code Agent through LangGraph ToolNode."""

from __future__ import annotations

import contextlib
import json
import os
import re
import threading
import time
import uuid
from io import BytesIO
from pathlib import Path
from typing import Annotated
from urllib.parse import urlencode

import requests
from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState
from PIL import Image

from . import art_memory
from .agents import _note, normalize_html, static_qa
from .comfyui import load_z_image_turbo_prompt
from .models import GameConcept, game_output_dir
from .required_art import missing_required, missing_required_finding
from .sprites import (
    DEFAULT_FACING,
    FACINGS,
    SHEET_MAX_FRAMES,
    SHEET_MIN_DISTINCT_POSES,
    SHEET_MIN_FRAMES,
    compose_prompt,
    compose_sheet_prompt,
    cut_background,
    distinct_poses,
    house_style,
    keyed_out,
    sheet_size,
    slice_sheet,
)


def _draft_path(state: dict) -> Path:
    concept = GameConcept.model_validate(state["concept"])
    root = Path(state["output_dir"]).resolve()
    target = Path(state.get("workspace_dir") or game_output_dir(root, concept.title)).resolve()
    if not target.is_relative_to(root):
        raise ValueError("Invalid game workspace path")
    target.mkdir(parents=True, exist_ok=True)
    return target / "draft.html"


def _reject_incomplete(normalized: str) -> str | None:
    """Catch obviously incomplete HTML before it burns a write+QA tool round trip."""
    lowered = normalized.lower()
    if "<canvas" not in lowered:
        return "no Canvas element"
    if "</html>" not in lowered:
        return "no closing </html> tag (the response was likely truncated)"
    return None


def _unused_sprites(normalized: str, workspace: Path) -> list[str]:
    """Generated PNGs the HTML never references.

    Runs were generating six sprites and then drawing none of them: the game had no assets/ string
    anywhere, so the art was paid for and thrown away. Reporting it in the write result puts the
    problem in front of the agent immediately instead of a QA cycle later.
    """
    assets = workspace / "assets"
    if not assets.exists():
        return []
    lowered = normalized.lower()
    return sorted(p.name for p in assets.glob("*.png") if p.name.lower() not in lowered)


@tool
def write_game_file(html: str, state: Annotated[dict, InjectedState]) -> str:
    """Write a complete standalone HTML5 Canvas game to the isolated game workspace."""
    normalized = normalize_html(html)
    problem = _reject_incomplete(normalized)
    if problem:
        return f"Rejected: the supplied HTML has {problem}. Create a complete game first."
    path = _draft_path(state)
    path.write_text(normalized, encoding="utf-8")
    return f"Wrote draft game to {path}.\n{_verdict_for(normalized, state, path)}"


def _verdict_for(normalized: str, state: dict, path: Path) -> str:
    """The deterministic verdict on what was just written, returned with the write itself.

    static_qa is a keyword scan plus a JavaScript parse - it costs no model call, and it is cached
    per content so asking twice is free. Making the agent spend a turn to ask for it doubled the
    round trips of the whole build loop: every write was followed by a run_static_qa call whose
    answer was already knowable at write time. It is handed back here instead, and run_static_qa
    remains for re-checking a draft the agent did not just write.
    """
    if missing := required_gap(state):
        return (f"{missing_required_finding(missing)} "
                "이미지를 만든 뒤 repair_html로 게임에 반영하세요.")
    assets = path.parent / "assets"
    sprites = sorted(p.name for p in assets.glob("*.png")) if assets.exists() else []
    report = static_qa(normalized, sprites)
    if report.status == "pass":
        # Advisories come back on a pass too. This is the one moment they are cheap to act on -
        # the agent is mid-loop with calls left - and dropping them here is what let a run ship art
        # it had paid for and never drawn.
        if report.findings:
            return ("정적 QA 통과. 다만 아래는 확인하세요:\n"
                    + "\n".join(f"- {finding}" for finding in report.findings))
        return "정적 QA 통과. 더 고칠 것이 없으면 여기서 끝내세요."
    return ("정적 QA 실패 — repair_html로 아래를 고친 뒤 다시 쓰세요:\n"
            + "\n".join(f"- {finding}" for finding in report.findings))


@tool
def read_game_file(
    state: Annotated[dict, InjectedState],
    start_line: int = 0,
    end_line: int = 0,
    outline: bool = False,
) -> str:
    """Read the current draft game, whole or in part, so it can be inspected or repaired.

    The draft is a full game and can run past 20,000 characters, and whatever comes back stays in
    this conversation and is re-sent on every later turn, so read narrowly when you can. Pass
    outline=True for a numbered map of the functions and blocks, then start_line/end_line (1-based,
    inclusive) to pull only the section you intend to change. With no arguments it returns the whole
    file.
    """
    path = _draft_path(state)
    if not path.exists():
        return "No draft exists yet. Call write_game_file with a complete HTML game."
    lines = path.read_text(encoding="utf-8").splitlines()
    if outline:
        marks = [
            f"{number:>5}: {line.strip()[:110]}"
            for number, line in enumerate(lines, 1)
            if re.search(r"(function\s+\w+|^\s*(const|let|var)\s+\w+\s*=|<canvas|<script|addEventListener|class\s+\w+)", line)
        ]
        return (f"{len(lines)} lines total. Structure:\n" + "\n".join(marks[:120])) if marks \
            else f"{len(lines)} lines total; no recognizable structure markers."
    if start_line or end_line:
        first = max(1, start_line or 1)
        last = min(len(lines), end_line or len(lines))
        if first > last:
            return f"Invalid range: the draft has {len(lines)} lines."
        body = "\n".join(f"{n:>5}: {lines[n - 1]}" for n in range(first, last + 1))
        return f"Lines {first}-{last} of {len(lines)}:\n{body}"
    return "\n".join(lines)


def required_gap(state: dict) -> list[str]:
    """Required sprites this build has not produced yet.

    Empty on a first pass - the asset plan is a menu there. Non-empty only after a re-plan that
    verification asked for, which is exactly when the generation stops being optional.
    """
    return missing_required(list(state.get("required_assets") or []),
                            _draft_path(state).parent / "assets")


@tool
def run_static_qa(state: Annotated[dict, InjectedState]) -> str:
    """Run deterministic safety and gameplay QA on the saved game draft."""
    path = _draft_path(state)
    if not path.exists():
        return json.dumps({"status": "repair", "findings": ["No draft HTML exists."]})
    assets = path.parent / "assets"
    sprites = sorted(p.name for p in assets.glob("*.png")) if assets.exists() else []
    report = static_qa(path.read_text(encoding="utf-8"), sprites)
    # A re-plan that the agent is free to ignore is how the same finding came back twice.
    if missing := required_gap(state):
        report.status = "repair"
        report.findings = [missing_required_finding(missing), *report.findings]
    return report.model_dump_json()


@tool
def repair_html(html: str, state: Annotated[dict, InjectedState]) -> str:
    """Replace the draft with a complete corrected standalone HTML game after QA reports an issue."""
    normalized = normalize_html(html)
    problem = _reject_incomplete(normalized)
    if problem:
        return f"Rejected: the repaired HTML has {problem}."
    path = _draft_path(state)
    path.write_text(normalized, encoding="utf-8")
    return f"Repaired draft at {path}.\n{_verdict_for(normalized, state, path)}"


@tool
def generate_asset(state: Annotated[dict, InjectedState]) -> str:
    """Record the Art Agent's existing canvas-first plan as this workspace's asset reference.

    This does not generate a new image; it persists the already-produced art direction (palette,
    canvas effects, asset plan) to disk so it is reviewable alongside the game. To generate an actual
    raster backdrop, use generate_comfyui_image instead.
    """
    path = _draft_path(state).parent / "assets" / "canvas-art-plan.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state.get("art", {}), indent=2), encoding="utf-8")
    return f"Saved art plan to {path}. Use list_game_assets to inspect available images."


_IMAGE_EXTENSIONS = ("*.png", "*.jpg", "*.jpeg", "*.webp")


@tool
def list_game_assets(state: Annotated[dict, InjectedState]) -> str:
    """List generated game assets with the exact size and rotation each one has to be drawn with.

    A bare list of filenames was not enough to draw them correctly: nothing said whether a PNG had
    its background cut away or was still an opaque rectangle, and nothing said which way the art
    faces - so a sprite generated facing right was rotated by the entity's heading a second time and
    pointed the wrong way. Every entry now carries its own drawing contract.
    """
    assets = _draft_path(state).parent / "assets"
    if not assets.exists():
        return "No generated assets are available. Use Canvas-only art."
    images = sorted(
        path.name for pattern in _IMAGE_EXTENSIONS for path in assets.glob(pattern)
    )
    if not images:
        return "No image assets are available."
    manifest = _read_sprite_manifest(assets)
    lines = [_draw_contract(name, manifest[name]) if name in manifest
             else f"assets/{name} (생성 기록 없음 · 방향/투명도 불명, 축에 맞춰 그리세요)"
             for name in images]
    return ("Available image assets:\n" + "\n".join(lines)
            + _written_so_far(manifest, images))


# How many of this run's own prompts are shown back. Enough to establish the register, few enough
# that the list does not push the drawing contracts out of the model's attention.
_PROMPT_ECHO_LIMIT = 4


def _written_so_far(manifest: dict, images: list[str]) -> str:
    """The descriptions this run already used, so the next one matches them.

    The house style clause fixes the art style; this fixes everything the style clause cannot
    cover - how much anatomy gets described, whether eyes are "big sparkling round" or "two dots",
    how a body is broken down. Measured, a run's sprites drifted in exactly those: the enemy got
    two clauses of description where the player got six, and they read as two different artists
    even where the style token agreed.

    This run's own prompts, not the art memory's. The memory answers "what works on this model"
    across runs; this answers "what have we already said" inside one.
    """
    written = [(name, str(manifest.get(name, {}).get("prompt", "")).strip()) for name in images]
    shown = [f"- {name}: {prompt[:160]}" for name, prompt in written if prompt][-_PROMPT_ECHO_LIMIT:]
    if not shown:
        return ""
    return ("\n\n이 게임에서 이미 쓴 설명입니다. 다음 이미지도 같은 결로 쓰세요 — 묘사의 자세함과 "
            "표현 방식을 맞추면 한 사람이 그린 것처럼 보입니다:\n" + "\n".join(shown))


# What each generated sprite promises about itself: which way it faces, and therefore the rotation
# that makes the art agree with the movement. It is written to disk rather than only returned,
# because the turn that generates a sprite is often not the turn that draws it - a repair cycle, or
# a fresh code-agent loop after a rethink, starts with an empty conversation and would otherwise be
# left guessing at an orientation nobody recorded.
SPRITE_MANIFEST = "sprites.json"


def _read_sprite_manifest(assets: Path) -> dict[str, dict]:
    path = assets / SPRITE_MANIFEST
    if not path.is_file():
        return {}
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
        return loaded if isinstance(loaded, dict) else {}
    except (OSError, ValueError):
        return {}


def _variant_prompt(assets: Path, base_name: str, pose: str) -> str:
    """The base sprite's own description, with this frame's pose appended.

    Falls back to the pose alone when the base is unknown - a wrong name should cost a slightly
    less consistent frame, not a failed generation.
    """
    base = _read_sprite_manifest(assets).get(_safe_asset_stem(base_name) + ".png", {})
    described = str(base.get("prompt", "")).strip()
    if not described:
        return pose
    # The base prompt already carries the subject; the pose replaces whatever pose it described.
    return f"{described}, {pose.strip()}"[:900]


def _record_sprite(assets: Path, name: str, entry: dict) -> None:
    manifest = _read_sprite_manifest(assets)
    manifest[name] = entry
    try:
        (assets / SPRITE_MANIFEST).write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    except OSError:
        return


def _draw_contract(name: str, entry: dict) -> str:
    """One line telling the code exactly how to draw this file."""
    facing = FACINGS.get(str(entry.get("facing", "")), None)
    size = f"{entry.get('width', '?')}x{entry.get('height', '?')}"
    if entry.get("kind") == "backdrop":
        return f"assets/{name} ({size}, 배경 이미지 · 불투명)"
    alpha = "배경 제거됨" if entry.get("transparent") else "배경 제거 실패 · 불투명 사각형"
    return f"assets/{name} ({size}, {alpha}) — {facing.instruction if facing else ''}".strip()


def _remember_prompt(state: dict, name: str, prompt: str, role: str, kind: str,
                     entry: dict) -> None:
    """File the prompt that made this image, so a later run can learn from how it turned out.

    The prompt used to be discarded the moment the PNG landed: the image was kept, its geometry was
    recorded, and the one piece of text that produced it was not - so every run started from
    nothing and the studio never got better at asking. Best effort; a build is not failed over its
    own notes.
    """
    try:
        art_memory.remember(
            name=name, prompt=prompt,
            role=(role or art_memory.guess_role(name)) if kind != "backdrop" else "backdrop",
            genre=str((state.get("implementation_plan") or {}).get("genre", "")),
            run_id=Path(state.get("workspace_dir") or "").name or "unknown",
            entry=entry,
        )
    except Exception:
        return


def _finish_asset(target: Path, content: bytes, kind: str, facing: str) -> dict:
    """Write the generated PNG and record what the game needs to know about it.

    A backdrop is kept exactly as generated. A sprite has its flat background cut away and is
    trimmed to the art, because an opaque 1024px rectangle with a character somewhere in the middle
    is not a sprite - drawn over the game it hides everything behind it, and drawn at the entity's
    size it shrinks the character to a speck inside its own empty margin.
    """
    cutout = cut_background(content) if kind != "backdrop" else None
    target.write_bytes(cutout.png if cutout else content)
    entry = {"kind": kind, "facing": facing if kind != "backdrop" else "none",
             "transparent": cutout is not None}
    # The canvas it was generated on, recorded whether or not the cut worked. A refused cut has no
    # geometry of its own to report, and a successful one can only be judged against the frame it
    # came out of: a cut-out that still spans the whole canvas never got trimmed, which means opaque
    # pixels reach every edge. See sprites.keyed_out.
    with contextlib.suppress(Exception), Image.open(BytesIO(content)) as source:
        entry |= {"source_width": source.width, "source_height": source.height}
    if cutout:
        entry |= {"width": cutout.width, "height": cutout.height,
                  "removed_share": round(cutout.removed_share, 3)}
    else:
        entry |= {"width": entry.get("source_width"), "height": entry.get("source_height")}
    return entry


def _comfy_asset_dir(state: dict) -> Path:
    path = _draft_path(state).parent / "assets"
    path.mkdir(parents=True, exist_ok=True)
    return path


# Extensions the model appends to an asset name it is only supposed to name, not to spell as a
# file. Stripped because the sanitiser below turns every non-alphanumeric character into a hyphen,
# so "enemy-goomba.png" became the stem enemy-goomba-png and the file landed as
# enemy-goomba-png.png. One run shipped four sprites written twice under both spellings - paid for
# twice - and a fifth referenced as res://assets/enemy-goomba-png.png with only enemy-goomba.png on
# disk, which is a resource error the moment that code path runs.
_ASSET_EXTENSIONS = ("png", "jpg", "jpeg", "webp", "bmp", "gif", "svg")


def _safe_asset_stem(name: str) -> str:
    cleaned = "".join(char.lower() if char.isalnum() else "-" for char in name).strip("-")
    # Checked after sanitising, so ".png", "-png" and " png" are all the same trailing token by the
    # time it is looked at - an already-mangled name coming back in normalises too.
    for extension in _ASSET_EXTENSIONS:
        if cleaned.endswith(f"-{extension}"):
            cleaned = cleaned[: -len(extension) - 1].strip("-")
            break
    return cleaned[:60] or "asset"


# A ceiling on PNG COUNT, kept only as a runaway guard and deliberately far above real work.
#
# It used to be the governing budget and it was measuring the wrong thing. Generation is local
# ComfyUI: nothing is billed, so there is no reason to ration REQUESTS. What a request actually
# costs is time - 31s for a 512x512 sprite, 58s for an animation sheet - and that is now its own
# budget (IMAGE_TIME_BUDGET). Counting files made the agent ration the wrong resource, and the tool
# result told it so on every call: "[9/14 of this run's image budget, 5 left]".
#
# So the count is no longer reported to the agent, and what it is told instead is the clock.
_MAX_GENERATED_IMAGES = int(os.getenv("COMFYUI_MAX_ASSETS", "60"))
# Reservations for images that are being generated right now. Concurrent tool calls in one turn
# would otherwise each see the folder as it was before any of them started.
_IMAGE_BUDGET_LOCK = threading.Lock()
_RESERVED_ASSETS: set[str] = set()


# What one sprite is generated at unless it is a backdrop. The canvas decides the generation time -
# 1024x1024 is four times the pixels of 512x512 and takes about four times as long - and a sprite is
# drawn on screen at something like 64px. Measured: agents were asking for 1024x1024 objects, paying
# two minutes each for detail no player can see, out of a run budget of ten.
#
# A bigger canvas also gives the model more room to invent a scene in, which is the failure the cut
# exists to catch.
SPRITE_CANVAS = int(os.getenv("COMFYUI_SPRITE_PIXELS", "512"))
BACKDROP_CANVAS = int(os.getenv("COMFYUI_BACKDROP_PIXELS", "1024"))


def _generate_comfyui_image(
    prompt: str,
    state: Annotated[dict, InjectedState],
    seed: int = 42,
    width: int = 0,
    height: int = 0,
    asset_name: str = "",
    kind: str = "sprite",
    facing: str = DEFAULT_FACING,
    role: str = "",
    variant_of: str = "",
) -> str:
    """Generate one PNG (a backdrop, or one named game object's sprite/icon) with the local
    ComfyUI API, cut its background away, and save it in this game's assets folder."""
    if not state.get("generate_images", False):
        return "Raster image generation is disabled for this run. Continue with the Canvas art plan."
    assets_dir = _comfy_asset_dir(state)
    stem = _safe_asset_stem(asset_name) if asset_name else f"comfyui-{uuid.uuid4().hex[:8]}"
    target = assets_dir / f"{stem}.png"
    # Reserve the slot before generating. When the model asks for several sprites in one turn the
    # tool calls run concurrently, so counting files alone let every one of them see an empty
    # folder: the budget was not enforced and each reported "1/8" no matter how many had been made.
    with _IMAGE_BUDGET_LOCK:
        taken = {path.name for path in assets_dir.glob("*.png")} | _RESERVED_ASSETS
        already = len(taken - {target.name})
        if already >= _MAX_GENERATED_IMAGES:
            return (
                f"Image budget reached ({_MAX_GENERATED_IMAGES} PNGs this run: "
                f"{', '.join(sorted(taken))}). Reuse an existing asset (see list_game_assets) or "
                "fall back to Canvas rendering for this object."
            )
        _RESERVED_ASSETS.add(target.name)
    art = state.get("art") or {}
    workspace = str(state.get("workspace_dir") or "")
    if refusal := _out_of_image_time(workspace):
        return refusal
    # Capped rather than obeyed. The agent has no way to know what a canvas costs, and asking for a
    # bigger one buys detail that is thrown away when the sprite is drawn at 64px.
    ceiling = BACKDROP_CANVAS if kind == "backdrop" else SPRITE_CANVAS
    width = min(int(width) or ceiling, ceiling)
    height = min(int(height) or ceiling, ceiling)
    try:
        # The staging and the facing are not suggestions bolted onto the caller's prompt - they are
        # what makes the image usable afterwards. A subject on a busy background cannot be cut out,
        # and a subject drawn in whatever three-quarter view the model felt like cannot be rotated
        # to match where the entity is going.
        # An animation frame is the same character in a different pose, so it inherits the prompt
        # that drew the character and changes only the pose. Written from scratch it drifts: one
        # run's walk1/walk2/jump came back as "cartoon platformer game sprite style, clean outline"
        # while the player they animate was "retro 8-bit pixel art style, thick dark outline" - the
        # same cat changing art style as it walked. generate_animation_frames removes the problem
        # rather than mitigating it; this path remains for a single frame asked for on its own.
        if variant_of:
            prompt = _variant_prompt(assets_dir, variant_of, prompt)
        # Every asset in one game gets the same style clause and the same palette, byte-identical.
        # Taken from the approved art direction rather than from this call, because the drift came
        # from the agent rewriting the style for each sprite - see sprites.house_style.
        style = house_style(str(art.get("style_token", "")), art.get("palette"))
        prompt = fit_subject(compose_prompt, prompt, kind, facing, style)
        positive, negative = compose_prompt(prompt, kind, facing, style)
        # A surviving background is re-rolled here rather than reported. Measured across 73 real
        # sprite prompts, NOT ONE described a scene - "뿔 두 개 달린 붉은 대형 슬라임, 크고 둥근
        # 몸통" is exactly what the tool asks for, and its cut was refused anyway. The agent's
        # wording is not the problem, so telling the agent to reword it spends a model call and a
        # round trip to arrive back at the same request. Another seed costs GPU time and nothing
        # else, and the run has a clock for that.
        entry_meta = None
        for attempt, attempt_seed in enumerate((seed, seed + 7919)):
            rendered = _render_png(positive, negative, attempt_seed, width, height, workspace)
            if isinstance(rendered, str):
                # A failed re-roll is not a failed call: the first image is already on disk.
                if entry_meta is not None:
                    break
                return rendered
            entry_meta = _finish_asset(target, rendered, kind, facing)
            warning = keyed_out(entry_meta)
            if not warning or attempt or _out_of_image_time(workspace):
                break
            _note("코드 Agent", f"{target.name}: {warning} 다른 시드로 다시 뽑습니다.")
        entry_meta["prompt"] = prompt
        _record_sprite(assets_dir, target.name, entry_meta)
        _remember_prompt(state, target.name, prompt, role, kind, entry_meta)
        if warning := keyed_out(entry_meta):
            return (f"{warning} 두 번 시도했지만 배경이 남았습니다 — 이 객체는 Canvas 도형으로 "
                    f"그리거나, 더 단순한 형태로 다시 요청하세요. {_image_clock(workspace)}")
        return f"Generated {_draw_contract(target.name, entry_meta)} {_image_clock(workspace)}"
    except Exception as error:
        return f"ComfyUI image generation failed: {type(error).__name__}: {error}"
    finally:
        # Release the slot either way: on success the file on disk is counted from here on, and a
        # failed generation must not burn a slot forever.
        with _IMAGE_BUDGET_LOCK:
            _RESERVED_ASSETS.discard(target.name)


@tool
def generate_comfyui_image(
    prompt: str,
    state: Annotated[dict, InjectedState],
    seed: int = 42,
    width: int = 0,
    height: int = 0,
    asset_name: str = "",
    kind: str = "sprite",
    facing: str = "right",
    role: str = "",
    variant_of: str = "",
) -> str:
    """Generate one PNG with the local ComfyUI API and save it in this game's assets folder: one
    game object cut out of its background, or one full-frame backdrop kept whole.

    Two different pictures, so two different requests. A SPRITE is one thing the game positions and
    moves, alone, with nothing behind it. A BACKDROP is a scene with nobody in it. Which one you
    are asking for is `kind`, and it changes what belongs in `prompt`.

    Call this once per distinct object you actually decided needs a raster image instead of a
    Canvas-drawn shape - a player ship, one enemy type, a collectible icon, a backdrop, and so on.
    Check list_game_assets first so you don't regenerate something that already exists. There is a
    small per-run budget; once it is hit, fall back to Canvas rendering for whatever is left.

    prompt: what to draw, and nothing about how it is framed - the framing is added for you.
        Write it in ENGLISH, whatever language the rest of the run is in. Measured over 46 paired
        generations of the same five objects: every image whose subject was described in English
        keyed cleanly (26/26); three described in Korean did not (23/26). The difference is not
        large enough to prove on its own (p=0.24) and the cut quality was identical when both
        worked - but English was never worse, and writing it costs nothing.
        For kind="sprite": describe ONLY the object itself - what it is, its shape and colours. No
            background, no scene, no floor, no shadow: anything you put behind the object survives
            the cut and ships as an opaque box. No OTHER game object either - a ball, a coin or a
            weapon the game moves separately gets its own call, because a ball drawn into the
            player's sprite can never leave the player's hand. A green object is fine: the backdrop
            switches to magenta when the subject sounds green, so a slime, a zombie or a frog no
            longer has to be recoloured to survive the cut.
        For kind="backdrop": describe ONLY the scene - the place, the time of day, the weather, the
            distant shapes. No characters, no creatures, no people: a figure painted into the
            backdrop cannot move, cannot be removed, and is still standing there while the real
            sprites walk past it. No HUD, no score, no text.
    asset_name: a short slug for the object ("player", "enemy-drone", "coin"), so the file is easy
        to reference (assets/<asset_name>.png) and re-generating the same object replaces it.
    kind: "sprite" for an object drawn into the game - staged on a flat screen, background removed,
        trimmed to the art, capped at 512px. "backdrop" for a full-frame background image - kept
        exactly as generated, never cut, up to 1024px.
    facing: which way a sprite is drawn, so your rotation can agree with it. The image model has no
        idea which way "forward" is, so this is fixed here rather than guessed at afterwards.
        Ignored for kind="backdrop", which has no direction and is never rotated.
        "right" - drawn facing +X. Rotate with ctx.rotate(Math.atan2(vy, vx)), no offset. Use this
            for anything that moves in a direction: ships, cars, creatures, projectiles.
        "up" - drawn facing -Y, for top-down art that reads better nose-up. Rotate with
            ctx.rotate(Math.atan2(vy, vx) + Math.PI / 2).
        "none" - no direction at all. Do not rotate it. Use for items, coins, blocks, obstacles.
        Pick one and draw it with exactly that rotation. The return value repeats the rule, and
        list_game_assets repeats it later for every asset in the folder.
    role: what the object IS, so a later game can learn from this prompt: "player", "enemy",
        "projectile", "pickup", "obstacle", "terrain", "effect", "ui", "backdrop". Different from
        kind, which is only about how the image is cut.
    variant_of: for an animation frame, the asset_name of the character it animates ("player").
        The base sprite's own description is reused and your prompt is treated as the pose alone,
        so write only the pose ("mid-stride, left leg forward"). Written from scratch a walk cycle
        drifts into a different art style from the character it belongs to.

    Do not ask for sparkles, stars, glows, trails or motion lines. They get cut out with the sprite
    and then follow the entity around the screen as a star welded above its head. Effects are drawn
    in code, where they can move and stop. The art style is fixed for the whole game and added for
    you - describe the object, not the style.
    """
    return _generate_comfyui_image(
        prompt=prompt, state=state, seed=seed, width=width, height=height,
        asset_name=asset_name, kind=kind, facing=facing, role=role, variant_of=variant_of,
    )


# Wall-clock seconds one run may spend waiting on ComfyUI, across every image it asks for.
#
# The PNG count was the only image budget, and it does not measure the thing that actually makes a
# run feel stuck. Measured across 64 real generations: a 512x512 sprite takes about 31s and a
# 1536x512 animation sheet about 58s, with a worst case of 166s - so the same "14 images" is four
# minutes of sprites or a quarter of an hour of sheets, and an animation that re-rolls costs twice
# again. Meanwhile the agent is blocked on each one, writing nothing.
#
# Ten minutes is comfortably more than a normal run spends and far less than a pathological one.
# Running out is not a failure: the art plan has always been Canvas-first with raster as an
# improvement, so the tools say so and the build carries on drawing shapes.
IMAGE_TIME_BUDGET = int(os.getenv("COMFYUI_TIME_BUDGET_SECONDS", "600"))
# Seconds spent per workspace. Keyed by workspace rather than by process because one dashboard
# serves many runs, and a long afternoon must not make the next run start already over budget.
_IMAGE_SECONDS: dict[str, float] = {}


def _image_time_left(workspace: str) -> float:
    with _IMAGE_BUDGET_LOCK:
        return IMAGE_TIME_BUDGET - _IMAGE_SECONDS.get(workspace, 0.0)


def _spend_image_time(workspace: str, seconds: float) -> None:
    with _IMAGE_BUDGET_LOCK:
        _IMAGE_SECONDS[workspace] = _IMAGE_SECONDS.get(workspace, 0.0) + seconds


def reset_image_time(workspace: str) -> None:
    """Give this workspace its clock back, because a revision is a new run.

    The budget is keyed by workspace so one dashboard can serve many runs at once - and a revision
    works in the SAME folder as the game it is reworking, which meant it inherited that game's spend
    and could start already out of time. A person asking for a change to a game that used nine of
    its ten minutes would be told to draw the rest in Canvas shapes before generating anything.

    The ceiling is per run, not per folder. Called where a revision begins, which is the only place
    a workspace is legitimately reused.
    """
    with _IMAGE_BUDGET_LOCK:
        _IMAGE_SECONDS.pop(workspace, None)


def _image_clock(workspace: str) -> str:
    """What the agent is actually spending. Reported instead of a file count, because the file count
    was never the constraint and saying it made the agent ration the wrong thing."""
    with _IMAGE_BUDGET_LOCK:
        spent = _IMAGE_SECONDS.get(workspace, 0.0)
    return f"[이미지 생성 {spent:.0f}초 / {IMAGE_TIME_BUDGET}초]"


def out_of_image_time(workspace: str) -> bool:
    """Whether this run has spent its image clock. Read by QA, which must not demand art the build
    was unable to make - see qa_node."""
    return _image_time_left(workspace) <= 0


def _out_of_image_time(workspace: str) -> str:
    """Empty while this run can still afford an image, or a sentence saying it cannot."""
    left = _image_time_left(workspace)
    if left > 0:
        return ""
    return (
        f"이 런의 이미지 생성 시간({IMAGE_TIME_BUDGET}초)을 모두 썼습니다. "
        "남은 객체는 Canvas 도형으로 그리세요 — 게임을 완성하는 것이 이미지를 더 만드는 것보다 "
        "우선입니다."
    )


# How much of a composed prompt is actually sent. It exists because a prompt is assembled from
# parts and one of them could run away - a model asking for a 5,000-character object description
# should not turn into a 5,000-character request.
#
# Raised from 1200 when animation sheets arrived. A six-frame sheet composes to about 1,460
# characters, and at 1200 the tail was silently cut: the staging clause sat at the end, so
# "on a flat solid #00FF00 green screen background" never reached the model, the model drew a white
# backdrop, and the chroma key correctly refused an image it could not key. The staging now comes
# first (see compose_sheet_prompt) AND the limit clears the longest prompt the studio composes, so
# neither half of that fix depends on the other.
# Raised again when the worst case was actually measured rather than guessed. The fixed clauses of
# a six-frame sheet come to 1,905 characters on their own - the pose list is one line per frame -
# and the subject needs room after that. 2,000 was a round number; this one clears the largest
# prompt the studio composes with a few hundred to spare.
#
# The ceiling is a runaway guard, not an encoder limit: Z-Image reads its prompt through a Qwen text
# encoder that handles far more than this. What it stops is a caller whose "object description"
# turns out to be five thousand characters.
PROMPT_SEND_LIMIT = int(os.getenv("COMFYUI_PROMPT_CHARS", "2400"))


# Floor on the caller's own description. If the fixed clauses ever grew past the send limit the
# subject would vanish entirely and the model would be asked to draw "a game sprite, centered, on a
# green screen" - a request with no subject in it. Better a clipped description than none.
MIN_SUBJECT_CHARS = 200


def fit_subject(compose, subject: str, *args, **kwargs) -> str:
    """The caller's description, shortened to whatever room the fixed clauses leave.

    Truncation used to happen at the other end and silently: the composed prompt was cut to
    PROMPT_SEND_LIMIT on its way out, which removes the TAIL. That cost a run its green screen once
    - the staging clause sat last, a long pose list pushed the prompt past the limit, and the model
    drew a white backdrop that the chroma key then correctly refused.

    The clauses were reordered so the essentials come first, and that remains the second line of
    defence. This is the first: the only elastic part of the prompt is the caller's own text, so it
    is the part that gives way, and it gives way by exactly as much as is needed.

    Measured worst cases against a 2000-character limit: a sprite composes to 1,476 and a backdrop
    to 1,113, both safe - but a six-frame sheet with a 600-character subject reaches 2,530. The
    overflow is arithmetic, not an accident, and arithmetic is what fixes it.
    """
    fixed = len(compose("", *args, **kwargs)[0])
    room = max(MIN_SUBJECT_CHARS, PROMPT_SEND_LIMIT - fixed)
    return subject[:room]


def _render_png(positive: str, negative: str, seed: int, width: int, height: int,
                workspace: str = "") -> bytes | str:
    """One ComfyUI generation. Returns the PNG bytes, or a sentence saying why there are none.

    Pulled out of _generate_comfyui_image so the sheet path submits work exactly the same way a
    single sprite does - same workflow file, same server, same timeout, same polling.
    """
    workflow_path = Path(os.getenv("COMFYUI_WORKFLOW_PATH",
                                   r"C:\dev\ComfyUI\text_to_image_z_image_turbo_nodes.json"))
    if not workflow_path.is_file():
        return f"ComfyUI workflow is unavailable: {workflow_path}"
    server = os.getenv("COMFYUI_SERVER", "http://127.0.0.1:8188").rstrip("/")
    # The budget covers queue wait as well as generation, and it is scaled by canvas area because
    # what it was tuned for - one 512x512 sprite - is not what an animation sheet asks for. A
    # 1536x512 sheet is six times the pixels; measured, it generates in 59s against 31s for the
    # sprite, and a flat 180s ran out while the job was still queued behind another one. The
    # environment variable stays the per-sprite figure so the setting still means what it did.
    timeout = int(os.getenv("COMFYUI_TIMEOUT_SECONDS", "180"))
    timeout = max(timeout, round(timeout * (width * height) / (512 * 512) / 2))
    session = requests.Session()
    started = time.monotonic()
    try:
        workflow = load_z_image_turbo_prompt(
            workflow_path,
            positive_prompt=positive[:PROMPT_SEND_LIMIT], negative_prompt=negative,
            seed=max(0, min(int(seed), 2**32 - 1)),
            width=max(256, min(int(width), 4096)), height=max(256, min(int(height), 2048)),
            filename_prefix=f"game-studio/{uuid.uuid4().hex[:8]}",
        )
        queued = session.post(f"{server}/prompt",
                              json={"prompt": workflow, "client_id": str(uuid.uuid4())}, timeout=30)
        if not queued.ok:
            return f"ComfyUI rejected the workflow ({queued.status_code}): {queued.text[:800]}"
        prompt_id = queued.json()["prompt_id"]
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            history = session.get(f"{server}/history/{prompt_id}", timeout=15)
            history.raise_for_status()
            entry = history.json().get(prompt_id, {})
            status = entry.get("status", {})
            if status.get("status_str") == "error":
                errors = [msg[1] for msg in status.get("messages", [])
                          if msg and msg[0] == "execution_error"]
                return f"ComfyUI execution failed for prompt {prompt_id}: {errors or status}"
            for output in entry.get("outputs", {}).values():
                if images := output.get("images", []):
                    query = urlencode({key: images[0].get(key, "")
                                       for key in ("filename", "subfolder", "type")})
                    return session.get(f"{server}/view?{query}", timeout=30).content
            time.sleep(1)
        return f"ComfyUI timed out after {timeout} seconds (prompt {prompt_id})."
    except Exception as error:
        return f"ComfyUI image generation failed: {type(error).__name__}: {error}"
    finally:
        session.close()
        # Charged whatever happened. A generation that timed out cost the run its wall clock just
        # as surely as one that returned a picture - more, in fact.
        _spend_image_time(workspace, time.monotonic() - started)


# Roles that are never a character with an animation. A wall, a floor tile, a HUD icon and a
# background do not walk, and a sheet asked for one comes back as three drawings of the same wall -
# which then get written out as "frames" and blitted in sequence, so the wall flickers between three
# near-identical images for no reason and three quarters of the image budget is gone.
#
# Taken from art_memory.ROLES, where "obstacle" is defined as 벽·블록·장애물 and "terrain" as
# 바닥·타일·플랫폼 - the two a wall is actually named as.
STATIC_ROLES = frozenset({"obstacle", "terrain", "ui", "backdrop"})


def _animatable(role: str, asset_name: str) -> str:
    """Empty when this object may have an animation sheet, or a sentence saying it may not."""
    settled = (role or "").strip().lower() or art_memory.guess_role(asset_name)
    if settled not in STATIC_ROLES:
        return ""
    return (
        f"\"{asset_name or settled}\"은(는) {settled} 입니다 — 벽·바닥·타일·배경·UI는 "
        "애니메이션 대상이 아닙니다. 같은 그림이 여러 장 나올 뿐이고 이미지 예산만 씁니다. "
        f"generate_comfyui_image(asset_name=\"{asset_name or settled}\", role=\"{settled}\", ...) "
        "로 한 장만 만드세요. 움직이는 연출이 필요하면 코드에서 위치나 회전을 바꾸세요."
    )


def _generate_animation_frames(
    prompt: str,
    state: Annotated[dict, InjectedState],
    motion: str = "a walk cycle",
    asset_name: str = "",
    facing: str = DEFAULT_FACING,
    role: str = "",
    frames: int = 3,
    seed: int = 42,
) -> str:
    """Generate every frame of one animation in a single image, then cut it into aligned frames."""
    if not state.get("generate_images", False):
        return "Raster image generation is disabled for this run. Continue with the Canvas art plan."
    # Refused before anything is generated or reserved: a wall sheet costs a minute of ComfyUI and
    # most of the run's image budget before anyone could notice it was three copies of a wall.
    if refusal := _animatable(role, asset_name):
        return refusal
    workspace = str(state.get("workspace_dir") or "")
    if refusal := _out_of_image_time(workspace):
        return refusal
    assets_dir = _comfy_asset_dir(state)
    stem = _safe_asset_stem(asset_name) if asset_name else f"anim-{uuid.uuid4().hex[:8]}"
    wanted = max(SHEET_MIN_FRAMES, min(int(frames), SHEET_MAX_FRAMES))
    # Reserve the whole set up front. The frames are one generation but several files, and the
    # budget counts files - without reserving them all, two concurrent calls each see room for a
    # full set and the run lands over budget.
    placeholders = [f"{stem}-{index}.png" for index in range(1, wanted + 1)]
    with _IMAGE_BUDGET_LOCK:
        taken = {path.name for path in assets_dir.glob("*.png")} | _RESERVED_ASSETS
        room = _MAX_GENERATED_IMAGES - len(taken - set(placeholders))
        if room < SHEET_MIN_FRAMES:
            return (f"Image budget reached ({_MAX_GENERATED_IMAGES} PNGs this run). "
                    "Animate this object in code instead, or reuse an existing sprite.")
        wanted = min(wanted, room)
        placeholders = placeholders[:wanted]
        _RESERVED_ASSETS.update(placeholders)
    try:
        art = state.get("art") or {}
        style = house_style(str(art.get("style_token", "")), art.get("palette"))
        prompt = fit_subject(compose_sheet_prompt, prompt, motion[:120], facing, wanted, style)
        positive, negative = compose_sheet_prompt(prompt, motion[:120], facing, wanted,
                                                  style)
        width, height = sheet_size(wanted)
        # Two attempts at most, and the second one only buys a better animation - never a better
        # character, which the sheet already guarantees. A frozen sheet is a property of the
        # generation rather than of the request, so a different seed is the only lever; it costs
        # ComfyUI time and no model calls, which is the cheap side of this pipeline's budget.
        # Enough distinct poses to animate. Asking for four and getting three real ones is fine;
        # getting one pose drawn three times is not, and only counting them apart tells them apart.
        target_poses = min(SHEET_MIN_DISTINCT_POSES, wanted)
        best: list = []
        for attempt, attempt_seed in enumerate((seed, seed + 7919)):
            rendered = _render_png(positive, negative, attempt_seed, width, height, workspace)
            if isinstance(rendered, str):
                # A failed re-roll is not a failed call: the first sheet is still on hand.
                if best:
                    break
                return rendered
            # Repeats are dropped rather than counted: a four-frame sheet holding three real poses
            # is a three-frame animation, not a four-frame one with a stutter in it.
            poses = distinct_poses(slice_sheet(rendered))
            if len(poses) > len(best):
                best = poses
            if len(best) >= target_poses:
                break
            # A re-roll is a whole second generation - about a minute of GPU with the agent
            # blocked on it. Worth it when the run has time to spare, never worth finishing the
            # run's image budget on.
            if _image_time_left(workspace) <= 0:
                break
            if attempt == 0:
                _note("코드 Agent",
                      f"서로 다른 포즈가 {len(poses)}개뿐입니다(요청 {wanted}). "
                      "다른 시드로 다시 뽑습니다.")
        cuts = best
        if not cuts:
            # Worth saying precisely: the sheet is one call, so the fallback is the old path rather
            # than a retry of the same thing.
            return (
                f"애니메이션 시트에서 프레임을 분리하지 못했습니다 ({width}x{height}). "
                "그린스크린이 잡히지 않았거나 프레임이 서로 붙어 있습니다. "
                f"generate_comfyui_image(asset_name=\"{stem}-1\", variant_of=\"...\") 로 "
                "프레임을 한 장씩 만드세요."
            )
        # The model decides how many frames it draws; asking for three and getting four is normal.
        # Taking what arrived beats re-rolling the generation for a count nobody can enforce.
        cuts = cuts[:wanted]
        written: list[str] = []
        for index, cut in enumerate(cuts, start=1):
            name = f"{stem}-{index}.png"
            (assets_dir / name).write_bytes(cut.png)
            entry = {"kind": "sprite", "facing": facing, "transparent": True,
                     "width": cut.width, "height": cut.height,
                     "removed_share": round(cut.removed_share, 3),
                     "prompt": f"{prompt} — {motion} ({index}/{len(cuts)})",
                     "frame_of": stem, "frame_index": index}
            _record_sprite(assets_dir, name, entry)
            _remember_prompt(state, name, entry["prompt"], role, "sprite", entry)
            written.append(name)
        # Said plainly when it is true. A caller told it has a walk cycle blits three identical
        # drawings and the character slides along without moving its legs - which reads as a bug in
        # the game, at the far end of the pipeline from what caused it.
        # Proportionate, because "fewer than asked" and "not an animation" are different problems.
        # Two distinct poses IS a walk cycle - contact and passing is the minimal one - so telling
        # the agent to abandon animation there would throw away something that works. One pose is
        # a still image whatever it is called.
        if len(written) >= target_poses:
            frozen = ""
        elif len(written) >= 2:
            frozen = (f" 참고: 요청한 {wanted}개 중 서로 다른 포즈는 {len(written)}개입니다. "
                      "그대로 순환시키면 됩니다 — 최소한의 걷기 주기는 2프레임입니다.")
        else:
            frozen = (" 경고: 서로 다른 포즈가 1개뿐입니다. 애니메이션이 아니라 정지 이미지이므로, "
                      "한 장만 쓰고 이동 효과는 코드로 주세요.")
        # The first frame's real entry, so the contract line carries its size rather than "?x?".
        contract = _draw_contract(written[0], _read_sprite_manifest(assets_dir)[written[0]])
        return (
            f"Generated a {len(written)} frame animation of \"{stem}\" in one image: "
            f"{', '.join(written)}. Every frame is {cuts[0].width}x{cuts[0].height} and aligned on "
            f"the same centre, so draw them at one fixed size and position and swap only the "
            f"image - do not re-measure or re-centre per frame. {contract}{frozen} "
            f"{_image_clock(workspace)}"
        )
    except Exception as error:
        return f"Animation sheet generation failed: {type(error).__name__}: {error}"
    finally:
        with _IMAGE_BUDGET_LOCK:
            _RESERVED_ASSETS.difference_update(placeholders)


@tool
def generate_animation_frames(
    prompt: str,
    state: Annotated[dict, InjectedState],
    motion: str = "a walk cycle",
    asset_name: str = "",
    facing: str = "right",
    role: str = "",
    frames: int = 3,
    seed: int = 42,
) -> str:
    """Generate EVERY frame of ONE ANIMATED CHARACTER at once, as aligned PNGs sharing one canvas.

    Only for something that actually moves under its own power - a player, an enemy, a creature.
    Walls, floors, tiles, blocks, platforms, pickups that just sit there, HUD icons and backgrounds
    do NOT use this: they have no animation, so what comes back is the same wall drawn three times,
    written out as three "frames" and blitted in sequence. Those go to generate_comfyui_image as one
    image, and anything that should appear to move is moved in code. This call refuses the roles
    that are never characters (obstacle, terrain, ui, backdrop) rather than wasting the budget.

    Use this instead of calling generate_comfyui_image once per frame. Asked separately, each frame
    comes back as a different character - a different art style, different markings, sometimes
    facing the other way - because nothing connects one generation to the next. Here all the frames
    are drawn in a single image, so they cannot disagree, and the image is cut into frames for you.

    It also costs one model call instead of one per frame, out of the same budget you use to write
    the game.

    prompt: describe ONLY the character, exactly as you would for generate_comfyui_image - what it
        is, its shape and colours. No background, no floor, no shadow, and no pose: the pose is
        what changes between frames, so it belongs in `motion`.
        In ENGLISH - see generate_comfyui_image for the measurement behind that.
        And nothing the GAME moves on its own. A ball, a bullet, a coin, a platform, a pickup is a
        separate sprite with its own position in your code; written into this prompt it is drawn at
        the character's feet, cut out attached to them, and then follows the character around the
        screen while the real one moves independently. Ask for "a football player", never "a
        football player kicking a ball".
    motion: what the character does across the frames - "a walk cycle", "a jump", "an attack swing",
        "an idle bob". One short phrase.
    asset_name: the base slug. Frames are written as <asset_name>-1.png, <asset_name>-2.png and so
        on, in order, and the return value lists them.
    frames: how many to ask for, 2 to 6. The image model decides the real count - asking for 3 and
        getting 4 is normal - so the return value tells you what actually landed. Use that.
    facing: which way the character is drawn, same meaning as in generate_comfyui_image. Every
        frame gets the same facing, so one rotation rule covers the whole animation.
    role: what the object IS ("player", "enemy", ...), so a later game can learn from this prompt.

    Every frame comes back the same size, centred on the character's centre of mass. Draw them at
    one fixed size and position and swap only which image you blit - measuring or re-centring each
    frame yourself puts the jitter back that this removes.

    If the frames cannot be separated the call says so and you fall back to generate_comfyui_image
    with variant_of, one frame at a time.
    """
    return _generate_animation_frames(
        prompt=prompt, state=state, motion=motion, asset_name=asset_name, facing=facing,
        role=role, frames=frames, seed=seed,
    )


# The code agent owns image generation end to end. generate_comfyui_image sits in its own toolset
# alongside write_game_file/run_static_qa, so a character or object image gets made at the moment
# the agent decides it needs one - there is no separate image-agent phase that has to run to
# completion before any code is written.
GAME_TOOLS = [
    write_game_file, read_game_file, run_static_qa, repair_html, list_game_assets,
    generate_comfyui_image, generate_animation_frames,
]
