"""State-aware tools used by the autonomous Code Agent through LangGraph ToolNode."""

from __future__ import annotations

import json
import os
import re
import threading
import time
import uuid
from pathlib import Path
from typing import Annotated
from urllib.parse import urlencode

import requests
from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState

from . import art_memory
from .agents import _note, normalize_html, static_qa
from .comfyui import load_z_image_turbo_prompt
from .models import GameConcept, game_output_dir
from .required_art import missing_required, missing_required_finding
from .sprites import (
    DEFAULT_FACING,
    FACINGS,
    SHEET_MAX_FRAMES,
    SHEET_MIN_FRAMES,
    SHEET_MIN_POSE_SPREAD,
    compose_prompt,
    compose_sheet_prompt,
    cut_background,
    house_style,
    keyed_out,
    pose_spread,
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
    if cutout:
        entry |= {"width": cutout.width, "height": cutout.height,
                  "removed_share": round(cutout.removed_share, 3)}
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


# Images are generated by local ComfyUI, so this cap is not about dollars - it is about the model
# calls spent requesting and wiring each one, and about a game not turning into a slideshow. Raised
# with the code agent's call budget.
_MAX_GENERATED_IMAGES = int(os.getenv("COMFYUI_MAX_ASSETS", "14"))
# Reservations for images that are being generated right now. Concurrent tool calls in one turn
# would otherwise each see the folder as it was before any of them started.
_IMAGE_BUDGET_LOCK = threading.Lock()
_RESERVED_ASSETS: set[str] = set()


def _generate_comfyui_image(
    prompt: str,
    state: Annotated[dict, InjectedState],
    seed: int = 42,
    width: int = 1024,
    height: int = 1024,
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
        slot = already + 1
    art = state.get("art") or {}
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
        positive, negative = compose_prompt(prompt[:900], kind, facing, style)
        rendered = _render_png(positive, negative, seed, width, height)
        if isinstance(rendered, str):
            return rendered
        entry_meta = _finish_asset(target, rendered, kind, facing)
        entry_meta["prompt"] = prompt
        _record_sprite(assets_dir, target.name, entry_meta)
        _remember_prompt(state, target.name, prompt, role, kind, entry_meta)
        remaining = _MAX_GENERATED_IMAGES - slot
        # A cut that removed almost nothing means the model painted a scene instead of a green
        # screen, so what landed is a rectangle with the background baked in. Measured: one run's
        # three walking frames cut to 48%, 61% and 97% opaque - the third shipped its whole
        # background and nothing said so.
        if warning := keyed_out(entry_meta):
            return (f"{warning} 같은 asset_name으로 다시 생성하세요 — 프롬프트에서 "
                    f"장면·배경 묘사를 빼고 물체 하나만 적으면 키가 잡힙니다. "
                    f"[{slot}/{_MAX_GENERATED_IMAGES}, {remaining} left]")
        return (
            f"Generated {_draw_contract(target.name, entry_meta)} "
            f"[{slot}/{_MAX_GENERATED_IMAGES} of this run's image budget, {remaining} left]"
        )
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
    width: int = 1024,
    height: int = 1024,
    asset_name: str = "",
    kind: str = "sprite",
    facing: str = "right",
    role: str = "",
    variant_of: str = "",
) -> str:
    """Generate one PNG for a single game object with the local ComfyUI API, with its background
    cut away and a known facing direction, and save it in this game's assets folder.

    Call this once per distinct object you actually decided needs a raster sprite instead of a
    Canvas-drawn shape - a player ship, one enemy type, a collectible icon, a backdrop, and so on.
    Check list_game_assets first so you don't regenerate something that already exists. There is a
    small per-run budget; once it is hit, fall back to Canvas rendering for whatever is left.

    prompt: describe ONLY the object itself - what it is, its shape, colours and style. Do not
        describe a background, a scene, a floor or a shadow: the framing is added for you, and
        anything you put behind the object survives the cut and ships as an opaque box. The object
        is generated on a green screen, so do not ask for a bright green object - green that runs
        straight into the backdrop with no outline between them is cut away with it. Pick another
        colour for that object, or give it a dark outline.
    asset_name: a short slug for the object ("player", "enemy-drone", "coin"), so the file is easy
        to reference (assets/<asset_name>.png) and re-generating the same object replaces it.
    kind: "sprite" for an object drawn into the game (background removed, trimmed to the art), or
        "backdrop" for a full-frame background image (kept exactly as generated).
    facing: which way a sprite is drawn, so your rotation can agree with it. The image model has no
        idea which way "forward" is, so this is fixed here rather than guessed at afterwards.
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


def _render_png(positive: str, negative: str, seed: int, width: int, height: int) -> bytes | str:
    """One ComfyUI generation. Returns the PNG bytes, or a sentence saying why there are none.

    Pulled out of _generate_comfyui_image so the sheet path submits work exactly the same way a
    single sprite does - same workflow file, same server, same timeout, same polling.
    """
    workflow_path = Path(os.getenv("COMFYUI_WORKFLOW_PATH",
                                   r"C:\dev\ComfyUI\text_to_image_z_image_turbo_nodes.json"))
    if not workflow_path.is_file():
        return f"ComfyUI workflow is unavailable: {workflow_path}"
    server = os.getenv("COMFYUI_SERVER", "http://127.0.0.1:8188").rstrip("/")
    timeout = int(os.getenv("COMFYUI_TIMEOUT_SECONDS", "180"))
    session = requests.Session()
    try:
        workflow = load_z_image_turbo_prompt(
            workflow_path,
            positive_prompt=positive[:1200], negative_prompt=negative,
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
        positive, negative = compose_sheet_prompt(prompt[:600], motion[:120], facing, wanted, style)
        width, height = sheet_size(wanted)
        # Two attempts at most, and the second one only buys a better animation - never a better
        # character, which the sheet already guarantees. A frozen sheet is a property of the
        # generation rather than of the request, so a different seed is the only lever; it costs
        # ComfyUI time and no model calls, which is the cheap side of this pipeline's budget.
        best: list = []
        best_spread = -1.0
        for attempt, attempt_seed in enumerate((seed, seed + 7919)):
            rendered = _render_png(positive, negative, attempt_seed, width, height)
            if isinstance(rendered, str):
                # A failed re-roll is not a failed call: the first sheet is still on hand.
                if best:
                    break
                return rendered
            cuts = slice_sheet(rendered)
            spread = pose_spread(cuts)
            if spread > best_spread:
                best, best_spread = cuts, spread
            if cuts and spread >= SHEET_MIN_POSE_SPREAD:
                break
            if attempt == 0:
                _note("코드 Agent",
                      f"애니메이션 프레임이 거의 같습니다(변화 {spread:.0%}). "
                      "다른 시드로 다시 뽑습니다.")
        cuts, spread = best, best_spread
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
        used = len({path.name for path in assets_dir.glob("*.png")})
        # Said plainly when it is true. A caller told it has a walk cycle blits three identical
        # drawings and the character slides along without moving its legs - which reads as a bug in
        # the game, at the far end of the pipeline from what caused it.
        frozen = ("" if spread >= SHEET_MIN_POSE_SPREAD else
                  f" 경고: 프레임끼리 거의 차이가 없습니다(변화 {spread:.0%}). "
                  "걷는 모습이 아니라 같은 그림 여러 장일 수 있으니, 애니메이션 대신 한 장만 쓰거나 "
                  "이동 효과는 코드로 주세요.")
        # The first frame's real entry, so the contract line carries its size rather than "?x?".
        contract = _draw_contract(written[0], _read_sprite_manifest(assets_dir)[written[0]])
        return (
            f"Generated a {len(written)} frame animation of \"{stem}\" in one image: "
            f"{', '.join(written)}. Every frame is {cuts[0].width}x{cuts[0].height} and aligned on "
            f"the same centre, so draw them at one fixed size and position and swap only the "
            f"image - do not re-measure or re-centre per frame. {contract}{frozen} "
            f"[{used}/{_MAX_GENERATED_IMAGES} of this run's image budget]"
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
