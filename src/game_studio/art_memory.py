"""What this studio learned about writing image prompts, kept across runs.

Every sprite this pipeline generates is made by a prompt the art director wrote, and until now that
prompt was thrown away the moment the PNG landed. The image was kept, its size and cut ratio were
recorded, and the one piece of text that actually produced it was not - so the next run started
from nothing and the studio never got better at asking.

This is that text, stored with what it produced and how it turned out, and read back when the art
director plans the next game. The image itself never travels: what transfers is the wording, so a
remembered prompt yields a new sprite rather than the old one again.

Deliberately our own corpus. Famous games' screenshots would be the obvious thing to retrieve over
and are the one thing this pipeline refuses to borrow - IDEA_SYSTEM says the mechanics may be
reproduced and "제목·캐릭터·아트만 직접 가져오지 않으면 됩니다", and an image store of other
people's art is exactly that line. Prompts we wrote for images we generated have no such problem,
and they answer a question nobody else can: what worked *here*, on this image model, at this size.

Chroma, persisted next to the games. Optional in the same way ComfyUI and Godot are optional: if
the package is missing or the store cannot be opened, art planning proceeds with no examples rather
than failing. A memory that can break a build is worse than no memory.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from .models import canonical_genre, project_data_dir

# What an image is *for*, as opposed to `kind`, which is how it gets cut and drawn (a backdrop
# keeps its background, a sprite does not). Fixed rather than free text because it is a lookup key:
# "good enemy prompts in 슈팅" only works if every enemy agreed to call itself one.
ROLES = (
    "player",      # 플레이어가 조작하는 것
    "enemy",       # 적·추격자
    "projectile",  # 총알·미사일·마법 등 날아가는 것
    "pickup",      # 코인·아이템·강화
    "obstacle",    # 벽·블록·장애물
    "terrain",     # 바닥·타일·플랫폼
    "effect",      # 폭발·파티클·잔상
    "ui",          # 아이콘·HUD 조각
    "backdrop",    # 전체 화면 배경
)
DEFAULT_ROLE = "player"

# Where the store lives. In the project's own data directory, not in the output folder: this is
# not a deliverable, nobody browses it looking for a game, and it belongs with the code that is the
# only thing that ever opens it. Created on first use, so a fresh clone needs no setup.
ART_MEMORY_DIR = os.getenv("ART_MEMORY_DIR", "").strip()
COLLECTION = "sprite-prompts"

# How many remembered prompts reach the art director. Few-shot, not a catalogue - past three the
# model starts averaging them instead of learning from them.
RECALL_LIMIT = int(os.getenv("ART_MEMORY_RECALL", "3"))


def memory_dir(store_root: str | Path | None = None) -> Path:
    """The store's directory. `store_root` overrides it, which is how tests stay isolated."""
    if store_root:
        return Path(store_root)
    if ART_MEMORY_DIR:
        return Path(ART_MEMORY_DIR)
    return project_data_dir() / "art-memory"


def _client(store_root: str | Path | None = None):
    """An open Chroma collection, or None if it cannot be had.

    Imported here rather than at module scope: chromadb pulls in onnxruntime and a tokenizer stack,
    which is a second or two of import time that a run without image generation should not pay.
    """
    directory = memory_dir(store_root)
    try:
        import chromadb

        directory.mkdir(parents=True, exist_ok=True)
        return chromadb.PersistentClient(path=str(directory)).get_or_create_collection(
            COLLECTION, metadata={"hnsw:space": "cosine"})
    except Exception:
        # Missing package, a locked file, a corrupt store - none of them is a reason to stop a
        # build that was only going to get a few example prompts out of this.
        return None


# Below this the model drew a small subject inside a large empty margin, and the cut left a sprite
# too small to draw at the entity's size without blurring. Measured: 21% of one corpus of 80 came
# back under 120px on a side from a 1024px generation.
MIN_USABLE_PIXELS = int(os.getenv("ART_MEMORY_MIN_PIXELS", "120"))


def auto_verdict(entry: dict[str, Any]) -> tuple[str, str]:
    """A label the pipeline can assign without asking anyone, plus why.

    Only the failures it can be sure of. "The subject came out tiny" is visible in the geometry the
    background cut already records; "this looks good" is not, and is left for a person.
    """
    if entry.get("kind") == "backdrop":
        return "", ""
    width, height = entry.get("width") or 0, entry.get("height") or 0
    if width and height and min(width, height) < MIN_USABLE_PIXELS:
        return "bad", f"잘라내고 {width}x{height}만 남아 확대하면 뭉갭니다."
    # There used to be a second rule here: more than 90% of the frame removed was called a waste.
    # It measured the canvas rather than the sprite, and a sprite is not square. A tall knight and a
    # wide drone leave most of the frame empty BECAUSE they are long in one axis, so the rule fired
    # on them while they were perfectly usable.
    #
    # Measured over 24 generations: it mislabelled 8, including a 180x180 coin and a 155x279 knight.
    # And it separated nothing - the genuinely unusable ones scored 0.91-0.97 removed and the good
    # ones 0.90-0.94, overlapping ranges, so no threshold could have saved it. What it was reaching
    # for is "did this come back too small to draw", which is the rule above, measured directly.
    return "", ""


def _document(prompt: str, role: str, genre: str) -> str:
    """What gets embedded. The prompt carries the wording; role and genre carry the context that
    makes two similar-sounding prompts different requests."""
    return f"[{canonical_genre(genre) or '미분류'} · {role}] {prompt}".strip()


def remember(
    store_root: str | Path | None = None,
    *,
    name: str,
    prompt: str,
    role: str,
    genre: str,
    run_id: str,
    entry: dict[str, Any],
) -> bool:
    """Record one generated sprite's prompt. Returns whether it was stored.

    A revision runs in the same folder under the same run id, so regenerating `player.png` lands on
    the same row - and it should: that row is the *current* image, the verdict on the old one does
    not describe it, and the reviewer has to be asked again. What must not vanish with it is the
    lesson. A prompt a person called good is the thing this whole store exists to collect, so a
    superseded one is archived before the row is taken over.
    """
    collection = _client(store_root)
    if collection is None or not prompt.strip():
        return False
    sprite_id = f"{run_id}:{name}"
    _archive_superseded(collection, sprite_id)
    label, reason = auto_verdict(entry)
    try:
        collection.upsert(
            ids=[sprite_id],
            documents=[_document(prompt, role, genre)],
            metadatas=[{
                # Normalised here as well as in the schema: this is the key the WHERE clause
                # matches on, and it has to be the same string no matter who called.
                "name": name, "role": role or DEFAULT_ROLE, "genre": canonical_genre(genre),
                "run_id": run_id, "prompt": prompt, "kind": entry.get("kind", "sprite"),
                "width": entry.get("width", 0), "height": entry.get("height", 0),
                "removed_share": float(entry.get("removed_share") or 0.0),
                # "" until somebody judges it. Auto only ever writes "bad", and only for a failure
                # it can see in the geometry - see auto_verdict.
                "verdict": label, "verdict_by": "auto" if label else "",
                "verdict_note": reason,
            }],
        )
        return True
    except Exception:
        return False


# Marks an archived copy of a prompt whose sprite was regenerated. Kept out of the review panel -
# the image it describes no longer exists - and kept in the corpus, because the prompt still worked.
ARCHIVE_MARK = "@was"


def _archive_superseded(collection, sprite_id: str) -> None:
    """Copy a human verdict out of the way before its row is overwritten."""
    try:
        existing = collection.get(ids=[sprite_id], include=["documents", "metadatas"])
        if not existing["ids"]:
            return
        meta = dict(existing["metadatas"][0])
        # Only a person's opinion is worth keeping. An automatic "bad" describes geometry that the
        # new image has its own version of, and an unjudged row is not a lesson at all.
        if meta.get("verdict_by") != "human":
            return
        collection.upsert(
            ids=[f"{sprite_id}{ARCHIVE_MARK}{meta.get('recorded_at', '') or len(sprite_id)}"],
            documents=[existing["documents"][0]],
            metadatas=[meta | {"superseded": True}],
        )
    except Exception:
        return


def judge(store_root: str | Path | None, sprite_id: str, label: str, note: str = "") -> bool:
    """Record a person's opinion of one sprite, which is the label that actually matters."""
    collection = _client(store_root)
    if collection is None or label not in {"good", "bad", ""}:
        return False
    try:
        existing = collection.get(ids=[sprite_id], include=["metadatas"])
        if not existing["ids"]:
            return False
        meta = dict(existing["metadatas"][0])
        meta |= {"verdict": label, "verdict_by": "human" if label else "", "verdict_note": note}
        collection.update(ids=[sprite_id], metadatas=[meta])
        return True
    except Exception:
        return False


def recall(
    store_root: str | Path | None = None,
    *,
    visual_direction: str,
    genre: str = "",
    role: str = "",
    limit: int = RECALL_LIMIT,
) -> list[dict[str, Any]]:
    """Prompts that produced sprites worth repeating, nearest this game's visual direction first.

    Rejected sprites are never returned - the point is to show the model what worked, not to
    caption a gallery of failures. Nothing judged yet is still returned: an unlabelled prompt from
    the same genre is better than no example, and the auto verdict has already removed the ones
    that came back unusably small.
    """
    collection = _client(store_root)
    if collection is None:
        return []
    genre = canonical_genre(genre)
    where: dict[str, Any] = {"verdict": {"$ne": "bad"}}
    if genre and role:
        where = {"$and": [where, {"genre": genre}, {"role": role}]}
    elif genre:
        where = {"$and": [where, {"genre": genre}]}
    elif role:
        where = {"$and": [where, {"role": role}]}
    try:
        found = collection.query(
            query_texts=[_document(visual_direction or "게임 스프라이트", role or "", genre)],
            n_results=max(1, limit), where=where, include=["metadatas"])
    except Exception:
        return []
    return [dict(meta) for meta in (found.get("metadatas") or [[]])[0]]


def as_examples(entries: list[dict[str, Any]]) -> str:
    """The recalled prompts as the art director should read them: wording first, then why it is
    here. Empty when there is nothing to show, so the caller can skip the whole block."""
    lines = []
    for entry in entries:
        praise = "플레이어가 좋다고 평가" if entry.get("verdict") == "good" else "이전 실행에서 사용"
        size = (f" · {entry['width']}x{entry['height']}px"
                if entry.get("width") and entry.get("height") else "")
        lines.append(f"- [{entry.get('role', '?')}] {entry.get('prompt', '')}\n"
                     f"  ({praise}{size})")
    return "\n".join(lines)


_ROLE_HINTS = (
    ("backdrop", ("backdrop", "background")),
    ("projectile", ("bullet", "missile", "shot", "arrow", "laser", "magic", "spell")),
    ("enemy", ("enemy", "ghost", "monster", "boss", "zombie", "slime", "goomba", "koopa", "bat")),
    ("pickup", ("coin", "item", "gem", "pickup", "powerup", "power-up", "heart", "star", "pellet")),
    ("obstacle", ("block", "wall", "obstacle", "brick", "spike", "rock")),
    ("terrain", ("tile", "ground", "floor", "platform", "terrain")),
    ("effect", ("effect", "explosion", "particle", "spark", "trail")),
    ("ui", ("icon", "hud", "button", "cursor")),
    ("player", ("player", "hero", "ship", "car", "character")),
)


def guess_role(name: str) -> str:
    """The role a sprite's own name implies.

    A fallback for a caller that did not say. The agent is asked for a role directly - a guess from
    a filename is right often enough to be useful and wrong often enough not to be trusted
    ("ghost-red" is an enemy and says nothing about it).
    """
    lowered = re.sub(r"[^a-z0-9]+", "-", (name or "").lower())
    # "bg" is the one hint too short to look for as a substring - it is inside "bgone" and
    # "debug" - so it is matched as a whole word instead. Worth the special case because it is
    # what agents actually name backgrounds: a delivered run called its backdrop "board-bg", and
    # a revision regenerating that name would have staged a scene on a green screen and cut it.
    if "bg" in lowered.split("-"):
        return "backdrop"
    for role, hints in _ROLE_HINTS:
        if any(hint in lowered for hint in hints):
            return role
    return DEFAULT_ROLE

def sprites_of(store_root: str | Path | None, run_id: str,
               present: set[str] | None = None) -> list[dict[str, Any]]:
    """The images this run is currently shipping, for a person to look at and judge.

    Read from the store rather than from the folder, because the folder has the PNGs and the store
    has what is being judged: the prompt behind each one, and whatever verdict it already carries.
    `present` is the folder's side of that - the filenames actually on disk now - so a revision that
    replaced or dropped a sprite leaves an evaluation set describing the game as it is, not as it
    was two builds ago.

    Ordered so the ones still waiting on an opinion come first, which after a revision means the
    regenerated ones: their row was taken over by a new image and its verdict reset with it.
    """
    collection = _client(store_root)
    if collection is None or not run_id:
        return []
    try:
        found = collection.get(where={"run_id": run_id}, include=["metadatas"])
    except Exception:
        return []
    rows = [dict(meta) | {"id": sprite_id}
            for sprite_id, meta in zip(found.get("ids") or [], found.get("metadatas") or [])
            if ARCHIVE_MARK not in sprite_id
            and (present is None or meta.get("name") in present)]
    rows.sort(key=lambda row: (row.get("verdict") != "", row.get("name", "")))
    return rows


def rejected_sprites(store_root: str | Path | None, run_id: str,
                     present: set[str] | None = None) -> list[str]:
    """The file names of this run's sprites that were judged bad, and only those.

    A revision used to reuse every image on disk. That is right for "fix this one mechanic" and
    wrong the moment the art itself is what needed fixing: a prompt improved between runs changes
    nothing for a game that already exists, because nothing asks for the old pictures again.

    The rule is the narrowest one that still acts. A sprite somebody rejected is known to be wrong,
    so it is remade. A sprite nobody looked at is not known to be anything, and remaking it would
    spend a minute of GPU to replace a picture that may well be better than its replacement.
    Silence is not a complaint.

    Both kinds of verdict count. A person saying "별로" is the one that matters, but the automatic
    label only ever fires on failures the geometry proves - a subject that came out 60px wide, a
    background the cut could not remove - and those are wrong whether or not anyone has looked.
    """
    return [str(row.get("name") or "") for row in sprites_of(store_root, run_id, present)
            if row.get("verdict") == "bad" and row.get("name")]
