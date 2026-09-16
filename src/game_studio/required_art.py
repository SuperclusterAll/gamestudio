"""Which sprites a build is obliged to produce, and whether it produced them.

The art direction's asset_plan is a menu on the first pass: the code agent owns the image budget
and decides which entries are worth spending it on. That is deliberate, and it stays that way.

A re-planned asset list is not a menu. It only exists because verification said art was missing, so
the objects it names are the answer to that finding - and an answer nobody is obliged to act on is
how the same finding came back on the next cycle: art re-planned, the code agent skipped the
generation for its own reasons, QA reported the identical gap, and the run spent a whole rethink
learning nothing. From a revision onward the named objects are required, and both the agent's own
verification tool and the graph's QA node check the same list through this module.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

# How many objects one revision may make mandatory. The image budget is shared with everything the
# agent decides it needs on its own, so a re-plan that demanded a dozen sprites would spend the
# whole run answering one finding.
#
# Six rather than four now that the code agent has 50 calls instead of 20. Generation itself is
# local ComfyUI and costs no API tokens; what it costs is the calls to request each sprite and wire
# it in, and those are what got more room.
MAX_REQUIRED_ASSETS = 6
# asset_plan entries are written as "<name>: <description>" (see ART_SYSTEM), so the object's name
# is what precedes the first colon. Anything else is taken whole and slugged.
_NAME_SPLIT = re.compile(r"[:：]")


def asset_slug(entry: str) -> str:
    """The file stem an asset_plan entry maps to, matching _safe_asset_stem in agent_tools."""
    name = _NAME_SPLIT.split(str(entry), 1)[0].strip()
    cleaned = "".join(char.lower() if char.isalnum() else "-" for char in name).strip("-")
    return cleaned[:60]


def required_assets(asset_plan: list[str], existing: list[str]) -> list[str]:
    """The planned objects that still have no generated image, in plan order.

    Anything already on disk is dropped: a revision that re-lists the player's own sprite is not
    asking for it to be generated twice, and re-generating costs budget that the object the finding
    was actually about then cannot have.
    """
    have = {Path(name).stem.lower() for name in existing}
    wanted: list[str] = []
    for entry in asset_plan or []:
        slug = asset_slug(entry)
        if slug and slug not in have and slug not in wanted:
            wanted.append(slug)
    return wanted[:MAX_REQUIRED_ASSETS]


def missing_required(required: list[str], assets_dir: Path) -> list[str]:
    """Which required assets have still not been generated."""
    if not required:
        return []
    have = ({path.stem.lower() for path in assets_dir.glob("*.png")}
            if assets_dir.is_dir() else set())
    return [name for name in required if name.lower() not in have]


def missing_required_finding(missing: list[str]) -> str:
    """One finding that names the files and says exactly how to produce them."""
    listed = ", ".join(missing)
    return (
        f"아트 재수립이 요구한 스프라이트가 생성되지 않았습니다: {listed}. "
        f"각각 generate_comfyui_image(asset_name=\"<이름>\", ...)로 만들고 게임에 그려 넣으세요."
    )


# Asset paths a game builds at runtime rather than writing out. A falling-block puzzle names its
# seven pieces in a loop - load("res://assets/block-" + kind + ".png") - so the literal
# "block-i.png" appears nowhere, and a check that only looks for literals calls a game that uses
# its art perfectly "art it never referenced". That is what this pattern exists to notice.
_DYNAMIC_ASSET_PATH = re.compile(
    r"res://[^\"']*?(?:\"\s*[+%]|\{|%\s*[\[(a-zA-Z_])"
    r"|(?:str|load|preload)\s*\(\s*[\"']res://[^\"']*[\"']\s*[+%]"
)


def unused_sprites(body: str, sprites: list[str]) -> list[str]:
    """Generated images the project appears never to use.

    Conservative on purpose, in both directions it can be wrong. A literal match is proof of use.
    A shared prefix that appears in the source - "block-" for block-i/block-l/block-o - is proof
    enough, because that is exactly what a runtime-assembled path looks like. And if the project
    assembles any res:// path at all, nothing is reported: the answer is genuinely unknowable by
    reading, and a false "you wasted this art" costs a rethink cycle to disprove.
    """
    if not sprites or _DYNAMIC_ASSET_PATH.search(body):
        return []
    lowered = body.lower()
    stems = [Path(name).stem.lower() for name in sprites]
    prefix = os.path.commonprefix(stems) if len(stems) > 1 else ""
    if len(prefix) >= 3 and prefix in lowered:
        return []
    return [name for name, stem in zip(sprites, stems, strict=True)
            if name.lower() not in lowered and stem not in lowered]
