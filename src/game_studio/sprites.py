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
#
# The sparkle group is next. A generated sprite kept arriving with stars and glitter floating above
# the character's head - sometimes because the agent asked for "sparkle effect", often because the
# model adds them to anything described as cute or magical. Baked into the cut-out they follow the
# entity around the screen, and a star welded above the player is not a visual effect, it is a
# defect. Effects belong in code, where they can move, fade and stop.
_SPRITE_NEGATIVE = (
    "green tint on the subject, green glow, green rim light, silhouette, solid black shape, "
    "sparkles, stars, glitter, twinkles, floating particles, motion lines, speed lines, "
    "aura, halo, glow trail, magic effect around the subject, "
    "background scenery, environment, landscape, room, gradient background, textured background, "
    "shadow, drop shadow, reflection, ground plane, floor, grass, foliage, multiple objects, "
    "collage, duplicate, cropped, cut off, border, frame, watermark, text, logo, signature, "
    "photorealistic, 3d render, blurry, low quality, distorted"
)
_BACKDROP_NEGATIVE = (
    "characters, people, creatures, text, letters, watermark, logo, signature, user interface, "
    "blurry, low quality, distorted"
)


# The floor when the art director leaves style_token empty. Deliberately plain: it does not decide
# what the game looks like, it only stops the assets from each deciding separately.
DEFAULT_STYLE_TOKEN = "flat 2D game art, clean readable shapes, solid dark outline"


def house_style(style_token: str, palette: dict[str, str] | None) -> str:
    """The clause every asset in one game shares, so they look like one game.

    Measured across 29 real prompts from five runs: the same run produced "retro 8-bit pixel art
    style", "cartoon game boss style", "cartoon pixel-vector style" and "cartoon platformer game
    sprite style". One run's walking frames came back in a different style from the character they
    animate - the same cat changing art style as it moved.

    The cause is that the agent rewrites the style from scratch for every sprite. So it is taken
    away from the sprite and attached here, byte-identical on every call, the same way a contract
    ceiling is enforced by the schema instead of asked for in the prompt.

    The palette rides along for the same reason and a worse one: it was computed by the art
    director, stored in ArtDirection, and then never reached the image model at all. Colour names,
    not hex - a diffusion model follows "warm orange" and ignores "#E8912D".

    An empty style_token falls back rather than producing no clause at all. A model that skips the
    field would otherwise switch the whole consistency mechanism off for that run, silently and
    exactly where it is needed - and the default is not a guess about the game, only a floor that
    keeps every asset in one register.
    """
    parts = [(style_token or "").strip() or DEFAULT_STYLE_TOKEN]
    if names := [name.replace("_", " ") for name in (palette or {}) if name][:5]:
        parts.append("colour palette: " + ", ".join(names))
    return ", ".join(parts)


# What share of a finished sprite may still be opaque before the cut has to be called a failure.
# A cut-out character leaves 40-70% opaque after trimming; measured normal frames came back at 48%
# and 61%. One frame of the same character came back 97% opaque at 468x464 - the model had painted
# a scene rather than a green screen, the key found nothing to remove, and the "sprite" shipped as
# a rectangle with its background welded on.
MAX_OPAQUE_SHARE = 0.90


def keyed_out(entry: dict) -> str:
    """Empty when the background really was removed, or a sentence saying it was not."""
    if entry.get("kind") == "backdrop" or not entry.get("transparent"):
        return ""
    width, height = entry.get("width") or 0, entry.get("height") or 0
    removed = entry.get("removed_share")
    if not (width and height) or removed is None:
        return ""
    # Opaque share of what survived the trim, which is what the game actually draws.
    if (1.0 - removed) <= MAX_OPAQUE_SHARE:
        return ""
    return (f"배경 제거에 실패했습니다: {width}x{height} 중 "
            f"{(1.0 - removed):.0%}가 불투명하게 남았습니다. 그린스크린이 잡히지 않았습니다.")


def compose_prompt(subject: str, kind: str, facing: str, style: str = "") -> tuple[str, str]:
    """Build the (positive, negative) pair for one asset.

    A backdrop is a full-frame image and must keep its background; a sprite is a cut-out object and
    has to be staged so the background can be removed afterwards. `style` is the house style from
    house_style() and goes on both, because a backdrop in a different style from the sprites drawn
    over it is the same defect seen from the other side.
    """
    suffix = f", {style}" if style else ""
    if kind == "backdrop":
        return f"{subject}{suffix}", _BACKDROP_NEGATIVE
    orientation = FACINGS.get(facing, FACINGS[DEFAULT_FACING])
    return f"{subject}, {orientation.prompt}{suffix}, {_SPRITE_STAGING}", _SPRITE_NEGATIVE


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


# --- Animation frames ---------------------------------------------------------------------------
#
# A walk cycle asked for one frame at a time comes back as a different character each time. Measured
# against every mechanism the image models expose: re-prompting drifts the art style, Nova Canvas's
# IMAGE_VARIATION returned a differently-marked cat facing the other way even at similarityStrength
# 1.0, its INPAINTING re-drew the parts outside the mask, and edge/segmentation conditioning held
# the character but also held the pose - a duplicate, not a frame.
#
# The one thing that works is asking for every frame in a SINGLE image. Style cannot drift inside
# one generation, so the frames are consistent by construction rather than by instruction. Measured
# on the real workflow: four frames came back at identical height (369px), identical top edge (y=2)
# and identical baseline (y=370) - zero vertical jitter, for free.
#
# It is also cheaper twice over: one 1536x512 sheet took 66.7s against 31.5s x 4 = 126s for the same
# frames separately, and it spends ONE of the code agent's model calls instead of four. Image
# generation shares that budget with writing the game, so the second saving is the larger one.
SHEET_MIN_FRAMES = 2
SHEET_MAX_FRAMES = 6
# Room per frame on the canvas. The model lays the frames out itself and ignores an exact count, so
# this only has to leave each one space; where they actually landed is worked out afterwards by
# looking at the image.
SHEET_FRAME_WIDTH = 384
SHEET_FRAME_HEIGHT = 512
# A gap this many columns wide is what separates two frames. Below it, a tail or a lifted paw
# reaching towards the neighbour would split one character into two.
SHEET_MIN_GAP = 12
# And the same for rows. Asked for "ONE single horizontal row" the model sometimes draws two anyway
# - measured on a 1152x512 sheet, which came back as 3 columns x 2 rows. Slicing that by columns
# alone puts two stacked characters in every "frame", and because every frame then holds the same
# pair they agree perfectly: 98% identical silhouettes, an animation that does not move. It looked
# like the best result yet until the frames were laid out and looked at.
SHEET_MIN_ROW_GAP = 16
# Anything narrower than this share of the widest frame is a speck, not a frame.
SHEET_MIN_FRAME_SHARE = 0.15

_SHEET_STAGING = (
    "sprite sheet, all frames in ONE single horizontal row, evenly spaced, identical character in "
    "every frame, every frame the same size and the same eye level, "
    "on a flat solid #00FF00 green screen background, no shadow, no ground, no scenery"
)
# The sprite negative minus the clauses that fight this request: several near-identical characters
# in one image is the whole point here, so "multiple objects, collage, duplicate" has to go.
_SHEET_NEGATIVE = (
    _SPRITE_NEGATIVE.replace("multiple objects, collage, duplicate, ", "")
    + ", different characters, changing colours, grid lines, panel borders, frame numbers, "
      "second row, stacked rows"
)


def compose_sheet_prompt(subject: str, motion: str, facing: str, frames: int,
                         style: str = "") -> tuple[str, str]:
    """Build the (positive, negative) pair for one animation sheet.

    `subject` describes the character and `motion` what it does across the frames. They are separate
    arguments because the point of the whole mechanism is that the character is described once for
    every frame - the drift this replaces came from re-describing it per frame.
    """
    count = max(SHEET_MIN_FRAMES, min(int(frames), SHEET_MAX_FRAMES))
    orientation = FACINGS.get(facing, FACINGS[DEFAULT_FACING])
    suffix = f", {style}" if style else ""
    positive = (
        f"a {count} frame animation sheet of ONE {subject}, "
        f"each frame a different moment of {motion}, "
        f"{orientation.prompt}{suffix}, {_SHEET_STAGING}"
    )
    return positive, _SHEET_NEGATIVE


def sheet_size(frames: int) -> tuple[int, int]:
    """The canvas to ask ComfyUI for, wide enough that the frames are not squeezed together."""
    count = max(SHEET_MIN_FRAMES, min(int(frames), SHEET_MAX_FRAMES))
    return SHEET_FRAME_WIDTH * count, SHEET_FRAME_HEIGHT


def _frame_spans(opaque, min_gap: int = SHEET_MIN_GAP) -> list[tuple[int, int]]:
    """Column ranges holding one character each.

    Frames are found by looking rather than by dividing the canvas into equal cells, because the
    model does not lay them out on an even pitch. Measured on a real 1536px sheet: the four frames
    sat at 99-376, 496-772, 876-1081 and 1189-1433, while equal 384px cells would have cut the
    second one in half.
    """
    import numpy as np

    columns = opaque.any(axis=0)
    if not columns.any():
        return []
    edges = np.diff(columns.astype(np.int8))
    starts = list(np.flatnonzero(edges == 1) + 1)
    ends = list(np.flatnonzero(edges == -1) + 1)
    if columns[0]:
        starts.insert(0, 0)
    if columns[-1]:
        ends.append(len(columns))
    merged: list[list[int]] = []
    for start, end in zip(starts, ends, strict=True):
        if merged and start - merged[-1][1] < min_gap:
            merged[-1][1] = end
        else:
            merged.append([start, end])
    widest = max(end - start for start, end in merged)
    floor = max(8, int(widest * SHEET_MIN_FRAME_SHARE))
    # int(), not numpy's int64. These travel into the sprite manifest as image geometry, and a
    # numpy integer is not JSON serialisable - measured: the first frame landed on disk and the
    # manifest write then threw, leaving a PNG nothing knew about.
    return [(int(start), int(end)) for start, end in merged if end - start >= floor]


def _centroid(opaque, start: int, end: int) -> float:
    """Horizontal centre of mass of one frame.

    The anchor every frame is aligned on, and the one choice here settled by looking rather than by
    a number. Frame-to-frame spacing variance preferred the bounding-box centre (27.8px against
    37.4px), but overlaying the aligned frames showed the opposite: aligned on the bounding box the
    heads scattered into a smear, while aligned on the centre of mass the bodies registered as one
    silhouette with only the legs and tail moving. A bounding box is decided entirely by whichever
    limb reaches furthest, which in a walk cycle is the part that is supposed to move.
    """
    import numpy as np

    weights = opaque[:, start:end].sum(axis=0)
    total = weights.sum()
    if not total:
        return (start + end) / 2
    return start + float((weights * np.arange(len(weights))).sum() / total)


# How different the frames have to be before this counts as an animation at all.
#
# The sheet makes the frames CONSISTENT reliably. Whether they actually MOVE is not reliable: the
# same prompt at two seeds produced a real walk cycle once and three near-identical drawings the
# other time, and no wording tried changed that - measured across three phrasings at two seeds
# each, five of the six came back under 2%. It is a property of the generation, not of the request.
#
# So it is measured instead of assumed. Calibrated on real sheets: a working walk cycle scored
# 13.9% and 9.9%, the frozen ones 0.8%, 1.1%, 1.1%, 1.7%. Nothing lands near 5%, which is what makes
# it a safe line to draw.
SHEET_MIN_POSE_SPREAD = 0.05


def pose_spread(frames: list[Cutout]) -> float:
    """How much the frames differ from each other, as a share of one frame's silhouette.

    Near zero means the model drew the same pose several times: consistent, and not an animation.
    A caller blitting those in sequence gets a character that slides along without moving its legs.
    """
    try:
        import numpy as np
        from PIL import Image

        if len(frames) < 2:
            return 0.0
        masks = [np.asarray(Image.open(BytesIO(frame.png)).convert("RGBA"))[..., 3] > 8
                 for frame in frames]
        first = masks[0]
        return float(np.mean([(first ^ mask).sum() / max(1, mask.sum()) for mask in masks[1:]]))
    except Exception:
        return 0.0


def slice_sheet(png_bytes: bytes, tolerance: int = CHROMA_TOLERANCE) -> list[Cutout]:
    """Cut one animation sheet into aligned, background-free frames.

    The background is cut from the whole sheet in one pass rather than per frame, and that is what
    keeps the frames vertically aligned for nothing: they are trimmed as a single image, so a
    baseline they share in the sheet they still share afterwards.

    Every frame comes back on the same canvas with its centre of mass on the canvas centre. Without
    that each frame is trimmed to its own bounds, and a game drawing them in sequence gets a
    character that changes size and jumps sideways on every frame.

    Returns [] when the sheet cannot be trusted - every condition cut_background refuses on, plus a
    sheet that turned out to hold fewer than two characters.
    """
    try:
        import numpy as np
        from PIL import Image

        cut = cut_background(png_bytes, tolerance)
        if cut is None:
            return []
        with Image.open(BytesIO(cut.png)) as opened:
            sheet = np.asarray(opened.convert("RGBA"))
        opaque = sheet[..., 3] > 8
        # Which rows the model actually used, and the one row to take the frames from. Frames from
        # different rows are not interchangeable - each row has its own scale and baseline, which is
        # the property that makes this whole approach work - so one row is chosen and the rest are
        # left. The best row is the one holding the most frames.
        bands = _frame_spans(opaque.T, SHEET_MIN_ROW_GAP) or [(0, sheet.shape[0])]
        top, bottom = max(bands, key=lambda band: (len(_frame_spans(opaque[band[0]:band[1]])),
                                                   band[1] - band[0]))
        sheet, opaque = sheet[top:bottom], opaque[top:bottom]
        spans = _frame_spans(opaque)
        if len(spans) < SHEET_MIN_FRAMES:
            return []
        height, sheet_width = sheet.shape[0], sheet.shape[1]
        centres = [_centroid(opaque, start, end) for start, end in spans]
        # Wide enough for the worst frame once it is CENTRED ON ITS CENTRE OF MASS, which is not the
        # same as wide enough to hold it. A frame whose mass sits left of its bounding box - a body
        # with one limb reaching out - moves right when it is aligned, and a canvas sized to the
        # bounding box clips whatever now hangs over the edge. Measured: a 230px frame in a 234px
        # canvas lost the end of the limb that made it 230px wide in the first place.
        reach = max(max(centre - start, end - centre)
                    for (start, end), centre in zip(spans, centres, strict=True))
        width = int(np.ceil(reach)) * 2 + CROP_PADDING * 2
        frames: list[Cutout] = []
        for (start, end), centre in zip(spans, centres, strict=True):
            canvas = np.zeros((height, width, 4), dtype=np.uint8)
            # Where this frame has to move for its centre of mass to land on the canvas centre.
            offset = round(width / 2 - centre)
            left = max(0, start + offset, offset)
            right = min(width, end + offset, sheet_width + offset)
            if right > left:
                canvas[:, left:right] = sheet[:, left - offset:right - offset]
                buffer = BytesIO()
                Image.fromarray(canvas, "RGBA").save(buffer, format="PNG")
                frames.append(Cutout(png=buffer.getvalue(), width=width, height=height,
                                     removed_share=1.0 - float((canvas[..., 3] > 8).mean())))
        return frames if len(frames) >= SHEET_MIN_FRAMES else []
    except Exception:
        # A sheet that cannot be sliced is not a failed run: the caller falls back to asking for the
        # frames one at a time, which is what it did before this existed.
        return []
