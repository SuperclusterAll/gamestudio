"""Turn a generated PNG into a game-ready sprite.

Two problems live here, and they are the same problem seen from both ends.

A text-to-image model cannot draw an alpha channel. It always returns a filled rectangle, so a
"sprite" pasted into a game arrives as a photo of a character sitting on a backdrop - an opaque
square that covers whatever it is drawn over. The fix is to ask for a flat, empty background we can
identify afterwards, and then cut it away ourselves.

And a model has no idea which way "forward" is. Ask twice for the same ship and it faces right,
then up, then three-quarter view; the game rotates the sprite by the entity's heading and the art
points somewhere the movement does not. Detecting the facing from the pixels is slow and wrong
often enough to be worse than useless, so nothing here tries. The orientation is fixed as a
contract at generation time instead - the prompt demands one specific facing, and the caller is
told exactly which rotation makes the art agree with the movement.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from io import BytesIO

# Everything below is one decision: the backdrop is a chroma key the prompt demands, and the cut is
# measured against the backdrop the model actually painted rather than the one it was asked for.
# Letting the model pick "some flat colour" is what broke the first version - asked for a neon car
# on a plain background it produced a dark car on a dark backdrop, the colour distance between the
# two fell inside the tolerance, and the fill walked straight through the bodywork.
# How far a pixel may sit from the backdrop's OWN measured colour and still count as backdrop,
# per channel. Measured backdrops came back as (6,224,10), (6,210,15), (16,236,22) - green, but
# never the pure key, and never uniform: corners drifted up to 42 per channel from the middle. So
# the reference is the backdrop the model actually painted, not the one the prompt asked for.
# Tight enough to exclude a green subject, wide enough to cover that drift.
CHROMA_TOLERANCE = 60
# What makes a backdrop count as the chroma key at all. Measured green screens came back as
# (6,224,10), (22,254,91) and (117,212,113) - all unmistakably green, none of them close to pure
# #00FF00, so an earlier check on "distance to the nominal key" threw away two perfectly good cuts.
# What actually has to be true is that the backdrop is nowhere near the colours a subject is painted
# in, and green dominance measures exactly that. This is the guard that catches the case it was
# written for: a model that ignores the instruction and paints a dark backdrop behind a dark subject
# fails it, instead of silently producing a sprite with holes cut through the bodywork.
KEY_MIN_GREEN = 120
KEY_MIN_DOMINANCE = 60
# Sanity bounds on what the cut removed. A subject that fills the frame leaves almost nothing to
# remove; a fill that escaped eats everything. Either way the original opaque image is kept.
MIN_BACKGROUND_SHARE = 0.06
MAX_BACKGROUND_SHARE = 0.985
# Chroma spill: the rim of the subject is blended with the key colour by anti-aliasing, so a cut on
# colour alone leaves a green fringe. Eroding the kept region by a pixel removes it.
SPILL_EROSION = 1
# A row or column is only content if this share of it is opaque. Cropping on "any opaque pixel at
# all" let a handful of stray specks at the frame edge hold the full height of a 768px image.
CONTENT_PROFILE_SHARE = 0.004
# Transparent margin kept around the trimmed art, so a rotated sprite does not clip its own corners.
CROP_PADDING = 2
# Improbable colour used to mark the filled region while flooding.
_FILL_KEY = (1, 254, 3)


@dataclass(frozen=True)
class Facing:
    """One canonical orientation a sprite may be generated in.

    `offset` is the angle added to an entity's heading so that the art points where the entity is
    going. Canvas rotation maps a vector v to (cos O*vx - sin O*vy, sin O*vx + cos O*vy), and a
    heading from Math.atan2(vy, vx) is 0 when moving right - so art drawn pointing right needs no
    offset, and art drawn pointing up (towards -Y, the top of the image) needs +PI/2.
    """

    key: str
    prompt: str
    offset: float
    instruction: str


FACINGS: dict[str, Facing] = {
    "right": Facing(
        key="right",
        prompt=(
            "strict side view, in profile, the subject faces and moves toward the RIGHT edge of "
            "the frame, nose and front pointing right, tail and back on the left"
        ),
        offset=0.0,
        instruction=(
            "이 스프라이트는 오른쪽(+X)을 향해 그려져 있습니다. 진행 방향으로 회전시키려면 "
            "ctx.rotate(Math.atan2(vy, vx))만 적용하세요(보정각 0). 좌우 이동만 하는 게임이면 "
            "회전 대신 ctx.scale(-1, 1)로 뒤집어 왼쪽을 보게 하세요."
        ),
    ),
    "up": Facing(
        key="up",
        prompt=(
            "strict top-down view seen from directly overhead, the subject faces and moves toward "
            "the TOP edge of the frame, nose and front pointing up, tail and back at the bottom"
        ),
        offset=math.pi / 2,
        instruction=(
            "이 스프라이트는 위쪽(-Y)을 향해 그려져 있습니다. 진행 방향으로 회전시키려면 "
            "ctx.rotate(Math.atan2(vy, vx) + Math.PI / 2)를 적용하세요."
        ),
    ),
    "none": Facing(
        key="none",
        prompt=(
            "seen straight on, symmetrical, with no implied direction of travel"
        ),
        offset=0.0,
        instruction=(
            "이 스프라이트는 방향이 없습니다. 회전시키지 말고 축에 맞춰 그대로 그리세요."
        ),
    ),
}
DEFAULT_FACING = "right"

# What the subject is staged on, kept deliberately short. A long staging clause does not just
# describe the frame - it competes with the subject for the model's attention. An earlier, wordier
# version turned "dark neon cyan racing car" into a black silhouette with green wheel rims: the
# background instruction had bled into the paintwork and the subject description had been diluted
# to almost nothing. Shadows are still called out, because a soft drop shadow is neither subject
# nor flat key, so the cut stops at it and the sprite ships with a grey smear welded to its feet.
_SPRITE_STAGING = (
    "single game sprite, centered, fully in frame, on a flat solid #00FF00 green screen background, "
    "no shadow, no ground, no scenery"
)
# The first four entries are the bleed guard: they push back on the green key colouring the subject
# and on the subject collapsing into a flat silhouette, which is how that bleed actually showed up.
_SPRITE_NEGATIVE = (
    "green tint on the subject, green glow, green rim light, silhouette, solid black shape, "
    "background scenery, environment, landscape, room, gradient background, textured background, "
    "shadow, drop shadow, reflection, ground plane, floor, grass, foliage, multiple objects, "
    "collage, duplicate, cropped, cut off, border, frame, watermark, text, logo, signature, "
    "blurry, low quality, distorted"
)
_BACKDROP_NEGATIVE = (
    "characters, people, creatures, text, letters, watermark, logo, signature, user interface, "
    "blurry, low quality, distorted"
)


def compose_prompt(subject: str, kind: str, facing: str) -> tuple[str, str]:
    """Build the (positive, negative) pair for one asset.

    A backdrop is a full-frame image and must keep its background; a sprite is a cut-out object and
    has to be staged so the background can be removed afterwards.
    """
    if kind == "backdrop":
        return subject, _BACKDROP_NEGATIVE
    orientation = FACINGS.get(facing, FACINGS[DEFAULT_FACING])
    return f"{subject}, {orientation.prompt}, {_SPRITE_STAGING}", _SPRITE_NEGATIVE


@dataclass(frozen=True)
class Cutout:
    """A finished sprite and what the caller needs to know to draw it."""

    png: bytes
    width: int
    height: int
    removed_share: float


def _backdrop_colour(pixels):
    """The colour the model actually painted behind the subject, or None if it is not the key.

    Taken as the median of a thin ring around the frame, which is backdrop unless the subject runs
    off every side at once. The green-ness check on it is the guard that makes the cut safe to
    trust: the first version let the model choose any flat colour, it answered a dark car with a
    dark backdrop, and the fill walked straight through the bodywork. Nothing downstream could tell
    that apart from a clean cut.
    """
    import numpy as np

    ring = np.concatenate([
        pixels[:4].reshape(-1, 3), pixels[-4:].reshape(-1, 3),
        pixels[:, :4].reshape(-1, 3), pixels[:, -4:].reshape(-1, 3),
    ]).astype(np.int16)
    median = np.median(ring, axis=0)
    green, other = float(median[1]), float(max(median[0], median[2]))
    if green < KEY_MIN_GREEN or green - other < KEY_MIN_DOMINANCE:
        return None
    return median


def _background_mask(pixels, backdrop, tolerance: int):
    """Pixels that are both the backdrop colour and reachable from the frame's edge.

    Both halves are load-bearing. Colour alone punches holes through any part of the subject that
    happens to be the backdrop's colour; reachability alone walks through a subject painted a
    similar colour. The colour test is done here rather than by the flood fill because Pillow
    compares a candidate against the SEED pixel using the sum of its channel differences - on a
    backdrop that drifts across the frame the fill starved and covered almost none of it, which
    read downstream as "the cut failed" on a perfectly good green screen.
    """
    import numpy as np
    from PIL import Image, ImageDraw

    candidate = (np.abs(pixels.astype(np.int16) - backdrop).max(axis=-1) <= tolerance)
    height, width = candidate.shape
    # .copy() is not tidiness. Image.fromarray hands back a readonly image sharing the numpy
    # buffer, and every flood fill into it is silently discarded - the fill reports success, the
    # pixels never change, and the cut comes back empty on a perfectly good green screen.
    canvas = Image.fromarray((candidate * 255).astype(np.uint8), mode="L").copy()
    marked = 128
    for x, y in ((0, 0), (width - 1, 0), (0, height - 1), (width - 1, height - 1),
                 (width // 2, 0), (width // 2, height - 1),
                 (0, height // 2), (width - 1, height // 2)):
        if canvas.getpixel((x, y)) == 255:
            ImageDraw.floodfill(canvas, (x, y), marked, thresh=0)
    return np.asarray(canvas) == marked


def _erode(mask, rounds: int):
    """Shrink the kept region by `rounds` pixels, taking the key-coloured anti-aliased rim with it."""
    for _ in range(rounds):
        shrunk = mask.copy()
        shrunk[1:] &= mask[:-1]
        shrunk[:-1] &= mask[1:]
        shrunk[:, 1:] &= mask[:, :-1]
        shrunk[:, :-1] &= mask[:, 1:]
        mask = shrunk
    return mask


def _content_bounds(kept, axis: int) -> tuple[int, int] | None:
    """First and last index along `axis` that carries real content rather than a stray speck."""
    import numpy as np

    counts = kept.sum(axis=axis)
    threshold = max(2, round(CONTENT_PROFILE_SHARE * kept.shape[axis]))
    found = np.flatnonzero(counts >= threshold)
    return (int(found[0]), int(found[-1])) if found.size else None


def cut_background(png_bytes: bytes, tolerance: int = CHROMA_TOLERANCE) -> Cutout | None:
    """Cut the chroma-key background away and trim to the art that remains.

    Background is what is both close to the key colour and reachable from the frame's edge. Both
    halves matter: matching on colour alone punches holes through any part of the subject that
    happens to be green, and connectivity alone - which is what the first version did - walks
    straight through a subject painted a similar colour to whatever backdrop the model invented.

    Trimming afterwards matters as much as the cut. A 768px frame holding a 300px car is mostly
    empty, and a game drawing it untrimmed either shrinks the car to a speck inside its own margin
    or guesses at an inset.

    Returns None whenever the result should not be trusted - the model ignored the chroma key, the
    fill escaped, or Pillow is not installed. The caller keeps the original opaque image and says
    so, rather than shipping a sprite with holes in it.
    """
    try:
        import numpy as np
        from PIL import Image
    except ImportError:
        return None
    try:
        with Image.open(BytesIO(png_bytes)) as opened:
            image = opened.convert("RGB")
        width, height = image.size
        pixels = np.asarray(image)
        backdrop = _backdrop_colour(pixels)
        if backdrop is None:
            return None
        background = _background_mask(pixels, backdrop, tolerance)
        share = float(background.mean())
        if not MIN_BACKGROUND_SHARE <= share <= MAX_BACKGROUND_SHARE:
            return None
        kept = _erode(~background, SPILL_EROSION)
        rows = _content_bounds(kept, axis=1)
        columns = _content_bounds(kept, axis=0)
        if rows is None or columns is None:
            return None
        with Image.open(BytesIO(png_bytes)) as opened:
            cut = opened.convert("RGBA")
        alpha = np.asarray(cut)[..., 3].copy()
        alpha[~kept] = 0
        cut.putalpha(Image.fromarray(alpha, mode="L"))
        box = (max(0, columns[0] - CROP_PADDING), max(0, rows[0] - CROP_PADDING),
               min(width, columns[1] + 1 + CROP_PADDING),
               min(height, rows[1] + 1 + CROP_PADDING))
        trimmed = cut.crop(box)
        buffer = BytesIO()
        trimmed.save(buffer, format="PNG", optimize=True)
        return Cutout(png=buffer.getvalue(), width=trimmed.width, height=trimmed.height,
                      removed_share=share)
    except Exception:
        # A cut-out is an improvement on an opaque rectangle, not a requirement for shipping. Any
        # failure here leaves the original image in place.
        return None
