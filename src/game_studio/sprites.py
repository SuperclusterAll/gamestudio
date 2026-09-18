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
            "the frame, its front toward the right edge and its back toward the left"
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
            "the TOP edge of the frame, its front toward the top edge and its back at the bottom"
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
# Said in the POSITIVE prompt, because that is the only prompt this model reads.
#
# Z-Image Turbo is distilled and runs at cfg 1, where classifier-free guidance is off and the
# negative branch has nothing to steer away from - the negative prompt is simply inert. Everything
# the pipeline depends on therefore has to be stated as something to draw rather than something to
# avoid. The negative list is kept for anyone who raises cfg, but nothing relies on it.
#
# Two of these were learned the hard way and are phrased as instructions rather than prohibitions:
# "arms relaxed at its sides" replaces a ban on pointing (every character came back gesturing), and
# "nothing floating around it" replaces a ban on sparkles (stars kept appearing over their heads,
# welded to the sprite by the cut and following the character around the screen afterwards).
_CLEAN_SUBJECT = ("the subject alone with nothing floating around it, arms relaxed at its sides, "
                  "no shadow, no ground, no scenery")


def _sprite_staging(key: str) -> str:
    return (f"single game sprite, centered, fully in frame, on a flat solid {key} background, "
            f"{_CLEAN_SUBJECT}")
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
    # Measured: every character came back with one arm extended, on single sprites as well as on
    # animation frames. The cause was one word - the facing clause said "nose and front POINTING
    # right", which beside a humanoid reads as a pointing gesture rather than as an orientation.
    # The clause is reworded, and the gesture is refused here too, so a prompt that says "forward"
    # about an arm cannot bring it back.
    "pointing gesture, pointing finger, index finger extended, arm extended straight out, "
    "gesturing at something, presenting pose, "
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
    """Empty when the background really was removed, or a sentence saying it was not.

    Three ways to still have a background, and this used to report only the middle one.

    The loudest case was the silent one: when cut_background REFUSES - the model painted something
    that is not a green screen - the original opaque image is kept, and this returned "" because
    there was no transparency to judge. Measured on a delivered game, boss-slime.png shipped as a
    1024x1024 rectangle with its whole scene baked in, and nothing in the run said so.

    The third is subtler. A cut-out sprite is always SMALLER than the canvas it was generated in,
    because the trim shrinks it to the art. One that still spans the full canvas has opaque pixels
    touching every edge - background fragments the fill could not reach, which is what a busy scene
    leaves behind. Measured: mushroom.png came back 1024x1024 at 55% opaque, under the opacity limit
    and still wrong.
    """
    if entry.get("kind") == "backdrop":
        return ""
    width, height = entry.get("width") or 0, entry.get("height") or 0
    if not entry.get("transparent"):
        size = f"{width}x{height} " if width and height else ""
        return (f"배경을 전혀 제거하지 못했습니다: {size}이미지가 불투명한 사각형 그대로입니다. "
                "모델이 그린스크린 대신 장면을 그렸습니다.")
    removed = entry.get("removed_share")
    if not (width and height) or removed is None:
        return ""
    # Opaque share of what survived the trim, which is what the game actually draws.
    if (1.0 - removed) > MAX_OPAQUE_SHARE:
        return (f"배경 제거에 실패했습니다: {width}x{height} 중 "
                f"{(1.0 - removed):.0%}가 불투명하게 남았습니다. 그린스크린이 잡히지 않았습니다.")
    source = entry.get("source_width") or 0, entry.get("source_height") or 0
    if all(source) and width >= source[0] and height >= source[1]:
        return (f"배경 일부가 남았습니다: 잘라낸 결과가 원본 캔버스와 같은 {width}x{height}입니다. "
                "가장자리까지 불투명하다는 뜻이고, 장면이나 바닥이 함께 그려진 경우입니다.")
    return ""


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
    return (f"{subject}, {orientation.prompt}{suffix}, {_sprite_staging(key_for(subject))}",
            _SPRITE_NEGATIVE)


@dataclass(frozen=True)
class Cutout:
    """A finished sprite and what the caller needs to know to draw it."""

    png: bytes
    width: int
    height: int
    removed_share: float


def _backdrop_colour(pixels):
    """The colour the model actually painted behind the subject, or None if it is not a key.

    Taken as the median of a thin ring around the frame, which is backdrop unless the subject runs
    off every side at once. The dominance check on it is the guard that makes the cut safe to trust:
    the first version let the model choose any flat colour, it answered a dark car with a dark
    backdrop, and the fill walked straight through the bodywork. Nothing downstream could tell that
    apart from a clean cut.

    Either key is accepted, and which one the image used is not asked in advance - the model may
    ignore the colour it was told to paint, and what matters is the colour it actually painted.
    """
    import numpy as np

    ring = np.concatenate([
        pixels[:4].reshape(-1, 3), pixels[-4:].reshape(-1, 3),
        pixels[:, :4].reshape(-1, 3), pixels[:, -4:].reshape(-1, 3),
    ]).astype(np.int16)
    median = np.median(ring, axis=0)
    return median if _is_key(median) else None


def _is_key(colour) -> bool:
    """Whether this colour is one of the chroma keys, by dominance rather than by distance.

    Measured green screens came back as (6,224,10), (22,254,91) and (117,212,113) - all
    unmistakably green, none of them close to pure #00FF00. Magenta behaves the same way, except
    that it is dominant in TWO channels at once and its minimum is the green one.
    """
    red, green, blue = float(colour[0]), float(colour[1]), float(colour[2])
    if green >= KEY_MIN_GREEN and green - max(red, blue) >= KEY_MIN_DOMINANCE:
        return True
    return (min(red, blue) >= KEY_MIN_GREEN
            and min(red, blue) - green >= KEY_MIN_DOMINANCE)


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
    "animation frames in ONE single horizontal row, evenly spaced, the same character in every "
    "frame seen from THE SAME CAMERA ANGLE, every frame the same size and the same eye level, "
    "only the pose changes between frames, the character ALONE in every frame with no other object, "
    "EVERY frame faces the SAME way - none of them mirrored, turned around or looking back, "
    "on a flat solid {key} background, nothing floating around the character, "
    "no shadow, no ground, no scenery"
)
# The sprite negative minus the clauses that fight this request: several near-identical characters
# in one image is the whole point here, so "multiple objects, collage, duplicate" has to go.
# The turnaround group is the one that had to be learned. Asked for a "sprite sheet" of a character
# the model drew what that phrase means to an illustrator - a reference sheet showing the same
# figure from the front, the side and the BACK. Measured on a real four-frame sheet: three identical
# standing poses and a rear view. Blitted in sequence that is a soldier who turns his back for one
# frame of every walk cycle, which reads as a rendering bug rather than as animation.
# Dropping "multiple objects" from the sprite negative is what makes several characters in one
# image possible - and it took the guard against OTHER objects with it. Measured: a request for a
# football player came back with a ball at the character's feet in every frame. Cut out, that ball
# is welded to the player and follows him around the pitch while the real ball moves separately;
# it is the "star above the head" defect wearing a different object. So the ban on a second
# CHARACTER is lifted and the ban on a second THING is put back explicitly.
_SHEET_NEGATIVE = (
    _SPRITE_NEGATIVE.replace("multiple objects, collage, duplicate, ", "")
    + ", ball, football, soccer ball, sports equipment, props, loose objects, "
      "a second separate object, items on the ground, scenery objects, "
    + ", different characters, changing colours, grid lines, panel borders, frame numbers, "
      "second row, stacked rows, "
      "character turnaround, model sheet, reference sheet, rotation sheet, "
      "back view, rear view, view from behind, front view and side view, "
      "different camera angles, changing viewing angle, T-pose, identical repeated pose, "
      # Observed: two frames of one soldier, identical except that the second had drawn a sword.
      # Equipment that appears between frames flickers in and out as the animation loops, which
      # reads as a missing-texture bug rather than as movement.
      "appearing weapon, disappearing weapon, changing equipment, different held items, "
      "different clothing, recoloured costume, changing outfit colour, "
      # The failure the positive prompt above caused, kept guarded from both sides.
      "headless, missing head, no head, cropped head, decapitated, body parts only, "
      "cut off at the neck"
)


# What changes between frames, spelled out limb by limb.
#
# The request used to say "each frame a different moment of a walk cycle" and leave the rest to the
# model, which is asking it to invent both the animation AND what an animation is. It answered the
# way that phrasing deserves: a measured four-frame soldier came back as two poses that differed
# only in that the second had drawn a sword. Technically different. Not walking.
#
# A walk cycle is arms and legs in a different position and NOTHING else moving, so that is what
# each frame now asks for by name. The named positions also give the model something concrete per
# frame instead of one instruction repeated N times, which is what let it repeat the drawing.
# A walk is the two legs taking TURNS, and the arms swinging opposite to them - left leg forward
# with right arm forward, then right leg forward with left arm forward. The first version of this
# list did not say so: every entry read "the near leg ...", which names no side at all, and frames 1
# and 4 both said the near leg was forward. Measured on a real sheet, three of four frames came back
# as the same stance with only the third differing - the model had been asked for the same pose
# three times in slightly different words, and it obliged.
#
# So each entry now names WHICH leg and WHICH arm, and they alternate. The contact poses (1 and 4)
# are mirror images of each other rather than repeats, and the passing poses between them differ by
# which leg carries the weight, which is what stops the de-duplicator from collapsing them.
_WALK = (
    "LEFT leg forward with the heel down, RIGHT arm forward and LEFT arm back",
    "weight on the LEFT leg, RIGHT leg passing under the body, arms at the sides",
    "RIGHT leg swinging to the front, LEFT heel lifting behind",
    "RIGHT leg forward with the heel down, LEFT arm forward and RIGHT arm back",
    "weight on the RIGHT leg, LEFT leg passing under the body, arms at the sides",
    "LEFT leg swinging to the front, RIGHT heel lifting behind",
)
_RUN = (
    "LEFT foot striking forward, RIGHT arm driving forward, elbows bent",
    "body low over the LEFT leg, RIGHT knee driving up",
    "pushing off the LEFT toe, both feet off the ground, RIGHT knee high",
    "RIGHT foot striking forward, LEFT arm driving forward, elbows bent",
    "body low over the RIGHT leg, LEFT knee driving up",
    "pushing off the RIGHT toe, both feet off the ground, LEFT knee high",
)
_JUMP = (
    "crouched low with the knees deeply bent and the arms drawn back",
    "pushing off with the legs extending and the arms thrown up",
    "at the top of the jump with the legs tucked under the body",
    "falling with the legs reaching down and the arms out for balance",
    "landing with the knees bending to absorb the impact",
    "rising back to a standing position",
)
_ATTACK = (
    "winding up with the weapon or fist drawn back behind the shoulder",
    "beginning the swing with the body turning into it",
    "the strike fully extended forward at its furthest reach",
    "following through with the arm carried across the body",
    "recovering with the arm returning toward the body",
    "settled back into a ready stance",
)
_IDLE = (
    "standing with the weight on one leg, arms relaxed at the sides",
    "breathing in, the chest and shoulders lifted slightly",
    "weight shifted to the other leg, one arm drifting out a little",
    "breathing out, the shoulders settling back down",
    "head tilted slightly, arms swaying gently",
    "returned to the neutral standing position",
)
# A walk cycle described for the wrong body is noise, and noise is what the model averages away.
#
# Measured: every walking case that failed the ten-case evaluation was told about "the RIGHT arm
# swung forward" and "the heel down" - including a cat, which has neither. Four frames of
# instructions aimed at limbs the subject does not have leaves the model with nothing to vary, and
# it drew the same animal four times. The cases that passed were the ones whose subject happened to
# match the vocabulary.
_WALK_QUADRUPED = (
    "the near FRONT leg reaching forward and the near HIND leg pushing back",
    "the near front and hind legs passing close together under the body",
    "the far FRONT leg reaching forward while the near front leg is tucked back",
    "the far HIND leg extended back with the paw lifting, the far front leg forward",
    "all four legs gathered under the body mid-stride",
    "the near front leg stretched far forward, the near hind leg stretched far back",
)
_RUN_QUADRUPED = (
    "the body stretched out with the front legs reaching forward and the hind legs trailing",
    "all four legs gathered under a curled body",
    "pushing off with both hind legs, the front legs still reaching",
    "airborne with the body fully extended, front legs forward and hind legs back",
    "the front paws landing with the hind legs swinging under",
    "the hind paws planted with the body compressing over them",
)
# Anything with four legs gets the four-legged description, in either language the studio runs in.
# A creature with no legs still animates - it squashes, stretches and leans. Saying nothing at all
# was worse than saying the wrong thing: the slime case went from four distinct poses to ONE the
# moment its (wrong) leg instructions were removed, because the leg instructions had been the only
# thing asking for any change.
_LIMBLESS_MOVE = (
    "squashed wide and low against the ground",
    "stretched tall and narrow with the top leaning forward",
    "mid-hop with the body rounded and tilted forward",
    "landing squashed again with the sides bulging out",
    "leaning back with the body compressed at the rear",
    "settled into a neutral round shape",
)
# Only when the request SAYS four legs. Naming a species is not enough: a game cat is usually drawn
# standing upright, and the measured cat sheet was walking on two legs while being told about its
# front and hind legs. A horse or a deer is a different matter - nobody draws those upright.
_QUADRUPED_HINTS = (
    "quadruped", "four-legged", "on all fours", "horse", "pony", "deer", "cow", "bull",
    "네발", "말", "사슴",
)
# And anything with no legs at all gets no leg instructions. A named pose it cannot perform is
# worse than saying nothing, which is the same rule motion_poses already applies to an unknown
# motion - see there.
_LIMBLESS_HINTS = (
    "slime", "blob", "ghost", "spirit", "orb", "ball", "cloud", "fish", "jellyfish", "squid",
    "snake", "worm", "bird", "bat", "bee", "butterfly", "drone", "ship", "spaceship", "car",
    "슬라임", "유령", "물고기", "뱀", "새", "박쥐", "우주선", "자동차", "공",
)


_MOTION_POSES: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = (
    (("walk", "walking", "걷", "보행"), _WALK),
    (("run", "running", "sprint", "dash", "달리", "질주"), _RUN),
    (("jump", "jumping", "leap", "hop", "점프", "뛰"), _JUMP),
    (("attack", "swing", "strike", "punch", "slash", "공격", "베", "때리"), _ATTACK),
    (("idle", "stand", "breathe", "bob", "대기", "서"), _IDLE),
)


def _mentions(description: str, hints: tuple[str, ...]) -> bool:
    """Whether the description names one of these things - as a WORD, not as a substring.

    "a football player" contains "ball", and matching on substrings classified him as limbless and
    took his walk cycle away. So a single English word has to match a whole word.

    A hint that is not a single plain word - "on all fours", "four-legged", or anything Korean -
    cannot be matched that way and is looked for as written. Those are phrases nobody writes by
    accident, so a substring match on them is safe in the way "ball" is not.
    """
    import re

    words = set(re.findall("[a-z0-9]+", description))
    return any(hint in words if hint.isascii() and hint.isalnum() else hint in description
               for hint in hints)


# Which colour to ask for the backdrop, and the one case where green is the wrong answer.
#
# A green subject drawn on a green screen is cut away with it. The guard against that used to be a
# sentence in the tool's docs telling the agent not to ask for a green object - which pushes a
# solvable problem onto the caller, and quietly loses when the game genuinely needs a slime, a
# zombie, a frog or a forest creature. Those are not rare in a 2D game; they are most of its
# bestiary.
#
# Magenta is the standard second key for exactly this reason: nothing in a green subject is near
# it. The idea is from jay6697117/game-skills, which reaches for "#ff00ff (magenta) if character
# uses green" from the other direction - it targets a different image backend, but the constraint
# it is solving is the same one.
#
# The cut does not need to be told which was used: _backdrop_colour reads the colour the model
# actually painted, and accepts either. That matters, because a model handed "magenta screen" does
# sometimes paint green anyway.
GREEN_KEY = "#00FF00 green screen"
MAGENTA_KEY = "#FF00FF magenta screen"
# Words that mean the subject is likely to be green. Deliberately broad: asking for magenta when
# green would have worked costs nothing, and asking for green when the subject is green costs the
# whole sprite.
_GREEN_SUBJECT = (
    "green", "olive", "lime", "emerald", "jade", "moss", "mint", "teal", "slime", "zombie",
    "frog", "toad", "lizard", "snake", "cactus", "leaf", "leaves", "vine", "grass", "goblin",
    "orc", "turtle", "alien", "초록", "녹색", "연두", "슬라임", "좀비", "개구리", "도마뱀",
    "선인장", "나뭇잎", "덩굴", "고블린", "오크", "거북",
)


def key_for(subject: str) -> str:
    """The chroma backdrop to ask for, given what is being drawn on it.

    Green unless the subject sounds green, because green is what the model paints most reliably
    when asked - magenta is the exception, not a coin flip.
    """
    return MAGENTA_KEY if _mentions((subject or "").lower(), _GREEN_SUBJECT) else GREEN_KEY


def motion_poses(motion: str, frames: int, subject: str = "") -> list[str]:
    """One named limb position per frame, for this motion performed by THIS body.

    Two ways to get no list, and both are deliberate. An unrecognised motion gets none because
    "a different moment of a tail whipping" is vague but not misleading. A subject with no legs gets
    none for a stronger reason: a named pose it cannot perform is worse than silence. Measured, that
    is not a hypothetical - every walking case that failed the ten-case evaluation had been told
    about "the RIGHT arm swung forward" and "the heel down", a cat among them.
    """
    lowered, described = (motion or "").lower(), (subject or "").lower()
    if _mentions(described, _LIMBLESS_HINTS):
        # Deforming, not stepping. Only for the motions that are locomotion; a limbless thing
        # swinging an attack is still better served by the generic attack poses.
        if any(hint in lowered for hint in ("walk", "run", "jump", "idle", "걷", "달리", "점프", "대기")):
            return list(_LIMBLESS_MOVE[:frames])
    quadruped = _mentions(described, _QUADRUPED_HINTS)
    for hints, poses in _MOTION_POSES:
        if any(hint in lowered for hint in hints):
            if quadruped:
                poses = {_WALK: _WALK_QUADRUPED, _RUN: _RUN_QUADRUPED}.get(poses, poses)
            return list(poses[:frames])
    return []


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
    # "sprite sheet" is deliberately not said. To an illustrator that phrase means a reference sheet
    # showing a figure from several angles, and the model draws exactly that - a real request came
    # back as three standing poses and a rear view. "Animation of one character" asks for the thing
    # actually wanted. It also avoids "ONE {subject}" reading as "ONE an armoured soldier".
    # Frame by frame, by name. "A different moment of a walk cycle" leaves the model to decide what
    # varies, and it decided a sword: two frames of a soldier identical except that one had drawn a
    # weapon. Naming the limb positions says what moves - and gives each frame its own instruction
    # instead of one instruction repeated N times, which is what let it repeat the drawing.
    if poses := motion_poses(motion, count, subject):
        described = ", ".join(f"frame {index} {pose}" for index, pose in enumerate(poses, 1))
        # Stated once as a rule as well as four times as poses. A model that half-follows the frame
        # list still has the principle to fall back on, and the principle is the thing that makes a
        # walk read as walking: limbs take turns, and the arm swings opposite its own leg.
        # "LEFT" and "RIGHT" here name the character's own limbs, and they sit in the same prompt
        # as a facing clause that also says RIGHT. Measured on a delivered game: the ghost's frames
        # 2, 3 and 4 all came back mirrored against frame 1, and the dinosaur turned around halfway
        # through its own walk cycle - while the manifest recorded every one of them as facing
        # right. So the two senses of the word are separated by saying so.
        movement = (f"{motion}: {described}; the left and right limbs ALTERNATE between frames and "
                    f"each arm swings opposite its own leg. LEFT and RIGHT above mean the "
                    f"character's OWN left and right limbs, never which way it faces - the whole "
                    f"animation faces one direction and no frame is mirrored")
    else:
        movement = f"each frame a different moment of {motion}"
    # What stays the same is named, not implied. "Only the arms and legs change between frames"
    # plus four frames of "the near leg... the opposite arm..." put every word of the request on two
    # body parts, and the model drew two body parts: a measured football sheet came back with four
    # HEADLESS players. Saying "everything else identical" does not put a head in the image; saying
    # "head" does.
    # Order matters, and it is not a matter of taste: the prompt is truncated before it is sent, so
    # whatever sits at the end is what gets thrown away. The staging used to be last, the pose list
    # pushed the whole thing past the limit, and "on a flat solid #00FF00 green screen background"
    # was cut off - the model drew a white backdrop, and the chroma key correctly refused an image
    # it could not key. Everything the pipeline DEPENDS on now comes first: the background it has to
    # cut, the framing it has to slice, the facing the game rotates by. The pose list is the only
    # part that can be shortened without breaking something downstream, so it goes last.
    positive = (
        f"a {count} frame animation of one character: {subject}, "
        f"{_SHEET_STAGING.format(key=key_for(subject))}, {orientation.prompt}{suffix}, "
        f"the complete character from head to feet in every frame, the head and face always drawn, "
        f"between frames only the arm and leg POSES change - the head, face, body, colours and "
        f"clothing are identical in all {count} frames, the SAME garment in the SAME colour on "
        f"every frame, no recolouring and no costume change, "
        f"{movement}"
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


# Two frames closer than this are the same drawing, not two poses.
#
# The gap in the real data is wide enough to make this easy. On a working four-frame walk cycle the
# pairwise differences were 0.9%, 11.9%, 12.3%, 14.4%, 15.2%, 25.7% - frames 1 and 2 were the same
# pose drawn twice, and every genuinely different pair was more than ten times further apart. So
# anything under a few percent is a repeat and there is nothing near the line to argue about.
DUPLICATE_FRAME_SHARE = 0.04
# How many genuinely different poses a set has to contain to be worth animating. Below this the
# model mostly repeated itself, and blitting the result gives a character that twitches rather than
# walks. Capped by how many frames were asked for, so a 2-frame request is not held to a 3-frame
# standard.
SHEET_MIN_DISTINCT_POSES = 3


def _silhouettes(frames: list[Cutout]):
    import numpy as np
    from PIL import Image

    return [np.asarray(Image.open(BytesIO(frame.png)).convert("RGBA"))[..., 3] > 8
            for frame in frames]


# How much better a mirrored frame has to match before it is treated as mirrored.
#
# Asking for one facing is not the same as getting it. Measured on a delivered game: the ghost's
# frames 2, 3 and 4 all matched frame 1's MIRROR better than frame 1 itself (60/53, 78/67, 86/69),
# and the player dinosaur turned around halfway through its own walk - while the sprite manifest
# recorded every frame as facing right. The game then flips those sprites by that recorded facing,
# so a character walking right plays its cycle facing backwards.
#
# Detecting facing from pixels in general is slow and wrong often enough to be useless, which is why
# this codebase fixes facing as a contract instead. But CONSISTENCY is a different question and a
# much easier one: frame 1 is the reference, and every later frame only has to agree with it. No
# absolute judgement is needed, and a frame that disagrees is repaired by mirroring it back rather
# than thrown away - the pose it holds is still a real pose.
#
# The margin keeps near-symmetric subjects alone. A round slime matches its own mirror almost
# exactly either way, and flipping it on a one-point difference would be noise pretending to be a
# fix.
MIRROR_MARGIN = 0.05


def _agreement(first, other) -> float:
    return float((first & other).sum()) / max(1, int((first | other).sum()))


def unmirror(frames: list[Cutout]) -> list[Cutout]:
    """Frames that came back facing the other way, flipped to agree with the first one.

    Only consistency is judged, never which way is "correct": the facing contract already decides
    that, and frame 1 is taken to honour it.
    """
    try:
        import numpy as np
        from PIL import Image

        if len(frames) < 2:
            return list(frames)
        images = [np.asarray(Image.open(BytesIO(frame.png)).convert("RGBA")) for frame in frames]
        first = images[0][..., 3] > 8
        fixed = [frames[0]]
        for frame, image in zip(frames[1:], images[1:], strict=True):
            opaque = image[..., 3] > 8
            direct, mirrored = _agreement(first, opaque), _agreement(first, opaque[:, ::-1])
            if mirrored <= direct + MIRROR_MARGIN:
                fixed.append(frame)
                continue
            buffer = BytesIO()
            Image.fromarray(image[:, ::-1], "RGBA").save(buffer, format="PNG")
            fixed.append(Cutout(png=buffer.getvalue(), width=frame.width, height=frame.height,
                                removed_share=frame.removed_share))
        return fixed
    except Exception:
        # A frame facing the wrong way is a defect; failing the whole sheet over the repair is
        # worse. Whatever was sliced is still usable.
        return list(frames)


# How far one frame's colours may sit from its nearest sibling before it is a different drawing.
#
# The prompt already says the clothing and colours are identical in every frame. The model agrees
# and then does it anyway: a measured four-frame soldier came back with a cream skirt in frame 1 and
# a red one in the other three, which flickers on every loop of the walk.
#
# Surveyed across 140 real frames on disk, the nearest-sibling colour distance has a median of 0.02
# and a 95th percentile of 0.05. Two frames in that whole set sat outside: the cream skirt at 0.27,
# and a slime whose frame 1 kept a cyan corner of its background at 0.64. Nothing at all lands
# between 0.07 and 0.27, so the line is drawn in open space.
#
# One check, two defects - clothing that changes and background that survived are the same signal
# from the frame's point of view: this one does not belong with the others.
COLOUR_DRIFT = 0.15
# Coarse buckets per channel. Fine enough to tell cream from red, coarse enough that shading and
# anti-aliasing do not register as a different costume.
_COLOUR_BINS = 4


def _colour_profile(image):
    """Share of each coarse colour bucket among the opaque pixels."""
    import numpy as np

    opaque = image[..., 3] > 8
    rgb = image[..., :3][opaque] // (256 // _COLOUR_BINS)
    index = (rgb[:, 0] * _COLOUR_BINS + rgb[:, 1]) * _COLOUR_BINS + rgb[:, 2]
    counts = np.bincount(index, minlength=_COLOUR_BINS ** 3).astype(float)
    return counts / max(1.0, counts.sum())


def consistent_colours(frames: list[Cutout]) -> list[Cutout]:
    """The frames drawn in the same colours as the rest, with the odd one out removed.

    Judged against the NEAREST sibling rather than the average, because a four-frame set with one
    bad frame has its average dragged by that frame: measured against the mean, the cream skirt and
    the red ones accused each other. Against the nearest neighbour the good frames have company and
    the odd one does not.

    Never cuts below two. A set where everything disagrees with everything is not a set with one bad
    frame in it - it is a failed sheet, and the caller's own checks say so better than silently
    handing back one picture.
    """
    try:
        import numpy as np
        from PIL import Image

        if len(frames) < 3:
            return list(frames)
        profiles = [_colour_profile(np.asarray(Image.open(BytesIO(f.png)).convert("RGBA")))
                    for f in frames]
        nearest = [min(1.0 - float(np.minimum(a, b).sum())
                       for j, b in enumerate(profiles) if j != i)
                   for i, a in enumerate(profiles)]
        kept = [frame for frame, distance in zip(frames, nearest, strict=True)
                if distance <= COLOUR_DRIFT]
        return kept if len(kept) >= SHEET_MIN_FRAMES else list(frames)
    except Exception:
        return list(frames)


def distinct_poses(frames: list[Cutout]) -> list[Cutout]:
    """The frames with repeats removed, in order, keeping the first of each group.

    This is measured rather than assumed because the model repeats itself constantly, and an
    average hides it. A real four-frame sheet came back as three identical standing poses and a
    fourth drawn from BEHIND - the model had produced a character turnaround instead of a walk
    cycle - and averaging every frame against the first let the back view alone carry the set over
    the threshold. Three quarters of that animation was one still image and the check passed it.

    Dropping the repeats is better than failing the set: a four-frame sheet holding three real poses
    becomes a three-frame animation instead of a four-frame one with a stutter in it.
    """
    try:
        if len(frames) < 2:
            return list(frames)
        masks = _silhouettes(frames)
        kept: list[int] = []
        for index, mask in enumerate(masks):
            if all((masks[other] ^ mask).sum() / max(1, mask.sum()) > DUPLICATE_FRAME_SHARE
                   for other in kept):
                kept.append(index)
        return [frames[index] for index in kept]
    except Exception:
        return list(frames)


def pose_spread(frames: list[Cutout]) -> float:
    """How far the two most similar frames are apart, as a share of a silhouette.

    A duplicate detector, not a quality score: near zero means some pair is the same drawing twice.
    Reported alongside the distinct-pose count because it says *how* close the repeat was.
    """
    try:
        if len(frames) < 2:
            return 0.0
        masks = _silhouettes(frames)
        return min(float((a ^ b).sum() / max(1, b.sum()))
                   for i, a in enumerate(masks) for b in masks[i + 1:])
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
        # Facing first, then costume. A mirrored frame is repaired; a frame drawn in different
        # colours cannot be, so it is dropped - and dropping it after the mirror repair means a
        # frame is never discarded for a fault that was fixable.
        frames = consistent_colours(unmirror(frames))
        return frames if len(frames) >= SHEET_MIN_FRAMES else []
    except Exception:
        # A sheet that cannot be sliced is not a failed run: the caller falls back to asking for the
        # frames one at a time, which is what it did before this existed.
        return []
