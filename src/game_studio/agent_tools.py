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

from .agents import normalize_html, static_qa
from .comfyui import load_z_image_turbo_prompt
from .models import GameConcept, game_output_dir
from .sprites import DEFAULT_FACING, FACINGS, compose_prompt, cut_background


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
    unused = _unused_sprites(normalized, path.parent)
    if unused:
        return (
            f"Wrote draft game to {path}, but it does not reference the sprites you generated: "
            f"{', '.join(unused)}. Load each one (new Image(); img.src='assets/<name>'), draw it "
            "with ctx.drawImage for that object, keep a Canvas fallback for when it fails to load, "
            "then call repair_html with the updated game. Next, call run_static_qa."
        )
    return f"Wrote draft game to {path}. Next, call run_static_qa."


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


@tool
def run_static_qa(state: Annotated[dict, InjectedState]) -> str:
    """Run deterministic safety and gameplay QA on the saved game draft."""
    path = _draft_path(state)
    if not path.exists():
        return json.dumps({"status": "repair", "findings": ["No draft HTML exists."]})
    assets = path.parent / "assets"
    sprites = sorted(p.name for p in assets.glob("*.png")) if assets.exists() else []
    return static_qa(path.read_text(encoding="utf-8"), sprites).model_dump_json()


@tool
def repair_html(html: str, state: Annotated[dict, InjectedState]) -> str:
    """Replace the draft with a complete corrected standalone HTML game after QA reports an issue."""
    normalized = normalize_html(html)
    problem = _reject_incomplete(normalized)
    if problem:
        return f"Rejected: the repaired HTML has {problem}."
    path = _draft_path(state)
    path.write_text(normalized, encoding="utf-8")
    return f"Repaired draft at {path}. Call run_static_qa again."


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
    return "Available image assets:\n" + "\n".join(lines)


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


def _safe_asset_stem(name: str) -> str:
    cleaned = "".join(char.lower() if char.isalnum() else "-" for char in name).strip("-")
    return cleaned[:60] or "asset"


_MAX_GENERATED_IMAGES = int(os.getenv("COMFYUI_MAX_ASSETS", "10"))
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
    workflow_path = Path(os.getenv("COMFYUI_WORKFLOW_PATH", r"C:\dev\ComfyUI\text_to_image_z_image_turbo_nodes.json"))
    if not workflow_path.is_file():
        return f"ComfyUI workflow is unavailable: {workflow_path}"
    server = os.getenv("COMFYUI_SERVER", "http://127.0.0.1:8188").rstrip("/")
    timeout = int(os.getenv("COMFYUI_TIMEOUT_SECONDS", "180"))
    session = requests.Session()
    try:
        prompt_id_hint = uuid.uuid4().hex[:8]
        # The staging and the facing are not suggestions bolted onto the caller's prompt - they are
        # what makes the image usable afterwards. A subject on a busy background cannot be cut out,
        # and a subject drawn in whatever three-quarter view the model felt like cannot be rotated
        # to match where the entity is going.
        positive, negative = compose_prompt(prompt[:900], kind, facing)
        workflow = load_z_image_turbo_prompt(
            workflow_path,
            positive_prompt=positive[:1200],
            negative_prompt=negative,
            seed=max(0, min(int(seed), 2**32 - 1)),
            width=max(256, min(int(width), 2048)),
            height=max(256, min(int(height), 2048)),
            filename_prefix=f"game-studio/{prompt_id_hint}",
        )
        queued = session.post(
            f"{server}/prompt",
            json={"prompt": workflow, "client_id": str(uuid.uuid4())},
            timeout=30,
        )
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
                errors = [msg[1] for msg in status.get("messages", []) if msg and msg[0] == "execution_error"]
                return f"ComfyUI execution failed for prompt {prompt_id}: {errors or status}"
            for output in entry.get("outputs", {}).values():
                images = output.get("images", [])
                if images:
                    image = images[0]
                    query = urlencode({key: image.get(key, "") for key in ("filename", "subfolder", "type")})
                    content = session.get(f"{server}/view?{query}", timeout=30).content
                    entry_meta = _finish_asset(target, content, kind, facing)
                    _record_sprite(assets_dir, target.name, entry_meta)
                    remaining = _MAX_GENERATED_IMAGES - slot
                    return (
                        f"Generated {_draw_contract(target.name, entry_meta)} "
                        f"[{slot}/{_MAX_GENERATED_IMAGES} of this run's image budget, "
                        f"{remaining} left]"
                    )
            time.sleep(1)
        return f"ComfyUI timed out after {timeout} seconds (prompt {prompt_id})."
    except Exception as error:
        return f"ComfyUI image generation failed: {type(error).__name__}: {error}"
    finally:
        session.close()
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
) -> str:
    """Generate one PNG for a single game object with the local ComfyUI API, with its background
    cut away and a known facing direction, and save it in this game's assets folder.

    Call this once per distinct object you actually decided needs a raster sprite instead of a
    Canvas-drawn shape - a player ship, one enemy type, a collectible icon, a backdrop, and so on.
    Check list_game_assets first so you don't regenerate something that already exists. There is a
    small per-run budget; once it is hit, fall back to Canvas rendering for whatever is left.

    prompt: describe ONLY the object itself - what it is, its shape, colours and style. Do not
        describe a background, a scene, a floor or a shadow: the framing is added for you, and
        anything you put behind the object survives the cut and ships as an opaque box.
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
    """
    return _generate_comfyui_image(
        prompt=prompt, state=state, seed=seed, width=width, height=height,
        asset_name=asset_name, kind=kind, facing=facing,
    )


# The code agent owns image generation end to end. generate_comfyui_image sits in its own toolset
# alongside write_game_file/run_static_qa, so a character or object image gets made at the moment
# the agent decides it needs one - there is no separate image-agent phase that has to run to
# completion before any code is written.
GAME_TOOLS = [
    write_game_file, read_game_file, run_static_qa, repair_html, generate_asset, list_game_assets,
    generate_comfyui_image,
]
