"""What a generated image has to become before a game can draw it.

Two failures these pin down were both silent. A sprite that keeps its background is an opaque
rectangle that covers whatever it is drawn over, and nothing in the pipeline could tell a clean cut
from one that had eaten holes through the subject. And a sprite whose facing nobody recorded gets
rotated by the entity's heading into pointing the wrong way - the most obvious defect a finished
game can ship with, and invisible to every check that only looks at the code.

Nothing here calls ComfyUI. The images are synthesised, so the cut is tested against the exact
conditions that broke it in practice rather than against whatever the model happens to draw today.
"""

from io import BytesIO

import pytest

from game_studio.sprites import (
    DEFAULT_FACING,
    FACINGS,
    Cutout,
    compose_prompt,
    cut_background,
)

np = pytest.importorskip("numpy")
Image = pytest.importorskip("PIL.Image")


def render(backdrop, subject, size=200, box=(60, 70, 140, 130)) -> bytes:
    """One flat backdrop with a solid rectangle sitting in the middle of it."""
    image = Image.new("RGB", (size, size), backdrop)
    for x in range(box[0], box[2]):
        for y in range(box[1], box[3]):
            image.putpixel((x, y), subject)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def alpha_of(cutout: Cutout):
    with Image.open(BytesIO(cutout.png)) as opened:
        return np.asarray(opened.convert("RGBA"))[..., 3]


def test_a_green_screen_is_cut_away_and_the_art_is_trimmed_to_its_own_bounds():
    """A 200px frame holding an 80x60 subject is mostly empty. A game drawing it untrimmed either
    shrinks the subject to a speck inside its own margin or guesses at an inset."""
    cutout = cut_background(render((10, 220, 15), (200, 40, 40)))
    assert cutout is not None
    # The 80x60 subject, plus the padding kept so a rotated sprite cannot clip its own corners,
    # minus the one-pixel erosion that takes the anti-aliased rim with it.
    assert 78 <= cutout.width <= 86 and 58 <= cutout.height <= 66
    assert cutout.removed_share > 0.8
    alpha = alpha_of(cutout)
    assert alpha[alpha.shape[0] // 2, alpha.shape[1] // 2] == 255, "the subject has to survive"
    assert (alpha == 0).any(), "and the backdrop has to be gone"


def test_a_dark_backdrop_is_refused_instead_of_cutting_holes_in_the_subject():
    """The failure this guard exists for. Asked for a neon car on a plain background the model
    produced a dark car on a dark backdrop; the colour distance between the two fell inside the
    tolerance and the fill walked straight through the bodywork, and nothing downstream could tell
    the result from a clean cut."""
    assert cut_background(render((38, 40, 46), (20, 24, 30))) is None
    assert cut_background(render((245, 245, 245), (250, 250, 250))) is None


def test_a_green_subject_behind_its_own_outline_keeps_its_interior():
    """Matching on colour alone punches holes through anything the subject paints green. Background
    is what is BOTH the backdrop colour and reachable from the frame's edge, so a green subject that
    is separated from the backdrop - by the dark outline cartoon game art nearly always has - keeps
    its interior. This is what saved a green slime generated on a green screen.

    The limit is real and worth stating: a subject whose green touches the key directly, with no
    outline between them, is contiguous with the background and gets cut. Nothing downstream can
    recover from that, which is why the tool tells the agent not to ask for a pure-green subject.
    """
    with Image.open(BytesIO(render((10, 220, 15), (20, 20, 25)))) as opened:
        image = opened.convert("RGB")
    for x in range(64, 136):
        for y in range(74, 126):
            image.putpixel((x, y), (60, 200, 70))
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    cutout = cut_background(buffer.getvalue())
    assert cutout is not None
    alpha = alpha_of(cutout)
    assert alpha[alpha.shape[0] // 2, alpha.shape[1] // 2] == 255, "the green interior survives"


def test_a_few_stray_specks_cannot_hold_the_whole_frame_open():
    """Cropping on "any opaque pixel at all" let a handful of specks at the frame edge keep the full
    768px height of a sprite whose art was 132px tall."""
    with Image.open(BytesIO(render((10, 220, 15), (200, 40, 40)))) as opened:
        image = opened.convert("RGB")
    for xy in ((2, 2), (197, 3), (4, 196)):
        image.putpixel(xy, (10, 10, 10))
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    cutout = cut_background(buffer.getvalue())
    assert cutout is not None
    assert cutout.height <= 70, "three specks must not hold 200px of empty frame open"


def test_the_cut_actually_writes_through_to_the_returned_image():
    """Image.fromarray hands back a readonly image sharing the numpy buffer, and flood fills into
    one are silently discarded - the fill reports success, the pixels never change, and the cut
    comes back empty on a perfectly good green screen. It cost two rounds of real generations to
    find, because every intermediate number looked right."""
    cutout = cut_background(render((10, 220, 15), (200, 40, 40)))
    assert cutout is not None
    alpha = alpha_of(cutout)
    assert alpha[0, 0] == 0, "the padding around the trimmed art has to be transparent"
    assert alpha[alpha.shape[0] // 2, alpha.shape[1] // 2] == 255, "and the art itself opaque"


def test_every_facing_states_the_rotation_that_makes_the_art_agree_with_the_movement():
    """Canvas rotation by O maps a vector v to (cos O*vx - sin O*vy, sin O*vx + cos O*vy), and a
    heading from Math.atan2(vy, vx) is 0 when moving right. So art drawn pointing right needs no
    offset and art drawn pointing up (-Y) needs +PI/2. Get this wrong and every sprite in the game
    points somewhere the entity is not going."""
    import math

    assert DEFAULT_FACING in FACINGS
    assert FACINGS["right"].offset == pytest.approx(0.0)
    assert FACINGS["up"].offset == pytest.approx(math.pi / 2)
    assert FACINGS["none"].offset == pytest.approx(0.0)

    # Each facing's own prompt has to demand the direction its offset assumes, or the contract is
    # a number with nothing behind it.
    assert "RIGHT" in FACINGS["right"].prompt
    assert "TOP" in FACINGS["up"].prompt
    assert "no implied direction" in FACINGS["none"].prompt

    # And the instruction handed to the code agent has to name that same rotation.
    assert "Math.atan2(vy, vx))" in FACINGS["right"].instruction
    assert "Math.PI / 2" in FACINGS["up"].instruction
    assert "회전시키지 말고" in FACINGS["none"].instruction
    assert "Math.atan2" not in FACINGS["none"].instruction


def test_an_unknown_facing_falls_back_to_the_default_rather_than_failing_the_build():
    """The facing arrives as a free-text argument from a model. A typo must cost a default, not a
    generation."""
    positive, _ = compose_prompt("우주선", "sprite", "sideways-ish")
    assert FACINGS[DEFAULT_FACING].prompt in positive


def test_every_asset_in_one_game_carries_the_same_style_clause():
    """Measured across 29 real prompts from five runs: the same run produced "retro 8-bit pixel art
    style", "cartoon game boss style", "cartoon pixel-vector style" and "cartoon platformer game
    sprite style". One run's walking frames came back in a different style from the character they
    animate - the same cat changing art style as it moved.

    The cause was the agent rewriting the style for every sprite, so it is taken away from the
    sprite and fixed here, byte-identical on every call.
    """
    from game_studio.sprites import compose_prompt, house_style

    style = house_style("retro 8-bit pixel art, thick dark outline",
                        {"warm_orange": "#E8912D", "cream": "#FFF3D6"})
    assert "retro 8-bit pixel art, thick dark outline" in style
    # Colour names, not hex: a diffusion model follows "warm orange" and ignores "#E8912D".
    assert "warm orange" in style and "cream" in style and "#E8912D" not in style

    sprite, _ = compose_prompt("주황 고양이", "sprite", "right", style)
    backdrop, _ = compose_prompt("숲 배경", "backdrop", "none", style)
    # A backdrop in a different style from the sprites drawn over it is the same defect from the
    # other side, so it gets the clause too.
    assert style in sprite and style in backdrop
    assert "green screen" in sprite and "green screen" not in backdrop

    # An art director that skips style_token would otherwise switch the whole mechanism off for
    # that run, silently and exactly where it is needed. The floor is plain on purpose: it does not
    # decide what the game looks like, only that its assets decide together.
    from game_studio.sprites import DEFAULT_STYLE_TOKEN

    assert house_style("", {}) == DEFAULT_STYLE_TOKEN
    assert house_style("", {"sky_blue": "#8FD"}) == f"{DEFAULT_STYLE_TOKEN}, colour palette: sky blue"
    plain, _ = compose_prompt("주황 고양이", "sprite", "right")
    assert ", ," not in plain, "and a missing style leaves no dangling separator"


def test_effects_that_would_be_welded_to_the_sprite_are_refused():
    """A generated sprite kept arriving with stars floating above the character's head. Cut out with
    the sprite they follow the entity around the screen - a star welded above the player is not a
    visual effect, it is a defect. Effects belong in code, where they can move and stop."""
    from game_studio.sprites import compose_prompt

    _, negative = compose_prompt("고양이", "sprite", "right")
    for banned in ("sparkles", "stars", "glitter", "motion lines", "aura", "halo", "glow trail"):
        assert banned in negative, banned
    # And the drift guard that came with it.
    assert "photorealistic" in negative and "3d render" in negative


def test_a_sprite_that_kept_its_background_says_so():
    """Measured: one run's three walking frames cut to 48%, 61% and 97% opaque. The third had its
    whole background baked in and nothing reported it - it shipped as a rectangle."""
    from game_studio.sprites import keyed_out

    assert keyed_out({"kind": "sprite", "transparent": True, "width": 342, "height": 264,
                      "removed_share": 0.61}) == ""
    failed = keyed_out({"kind": "sprite", "transparent": True, "width": 468, "height": 464,
                        "removed_share": 0.03})
    assert "배경 제거에 실패" in failed and "468x464" in failed and "97%" in failed

    # A backdrop is never cut, and an image the cut refused to touch already reports itself.
    assert keyed_out({"kind": "backdrop", "transparent": False}) == ""
    assert keyed_out({"kind": "sprite", "transparent": False, "width": 10, "height": 10,
                      "removed_share": 0.0}) == ""
    assert keyed_out({"kind": "sprite", "transparent": True}) == "", "no geometry, no verdict"
