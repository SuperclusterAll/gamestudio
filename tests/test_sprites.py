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


def sheet(boxes, size=(1152, 512), backdrop=(10, 220, 15), subject=(200, 40, 40)) -> bytes:
    """A green sheet with a solid rectangle per frame. `boxes` are (x0, y0, x1, y1)."""
    image = Image.new("RGB", size, backdrop)
    for x0, y0, x1, y1 in boxes:
        for x in range(x0, x1):
            for y in range(y0, y1):
                image.putpixel((x, y), subject)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def alpha_frames(frames):
    return [np.asarray(Image.open(BytesIO(f.png)).convert("RGBA"))[..., 3] for f in frames]


def test_one_sheet_becomes_several_frames_on_a_shared_canvas():
    """The whole reason the sheet exists. Frames asked for one at a time come back as different
    characters; frames drawn in one image cannot disagree, so they are cut apart afterwards instead.

    Each frame must land on the SAME canvas. Trimmed to their own bounds they are all different
    sizes, and a game drawing them in sequence gets a character that grows and shrinks as it walks.
    """
    from game_studio.sprites import slice_sheet

    frames = slice_sheet(sheet([(100, 150, 300, 400), (500, 150, 680, 400),
                                (900, 150, 1120, 400)]))
    assert len(frames) == 3
    assert len({(f.width, f.height) for f in frames}) == 1, "one canvas for every frame"
    assert frames[0].width >= 220, "the canvas fits the widest frame"


def test_frames_are_aligned_on_the_centre_of_mass_not_the_bounding_box():
    """Settled by looking, against what the numbers first suggested. Frame-to-frame spacing variance
    preferred the bounding-box centre (27.8px against 37.4px), but overlaying real aligned frames
    showed the heads scattered into a smear on the bounding box and registering as one silhouette on
    the centre of mass. A bounding box is decided entirely by whichever limb reaches furthest -
    which in a walk cycle is exactly the part that is supposed to move.

    Here the third frame carries a thin limb reaching far to the right, which drags its bounding box
    without moving its body. Aligned on the box, that body would sit left of the others.
    """
    from game_studio.sprites import slice_sheet

    frames = slice_sheet(sheet([(100, 150, 260, 400), (500, 150, 660, 400),
                                (900, 150, 1060, 400), (1060, 260, 1130, 290)]))
    assert len(frames) == 3
    centres = []
    for alpha in alpha_frames(frames):
        columns = (alpha > 8).sum(axis=0)
        centres.append((columns * np.arange(len(columns))).sum() / columns.sum())
    assert max(centres) - min(centres) <= 2.0, f"bodies must register: {centres}"


def test_a_sheet_the_model_drew_in_two_rows_yields_one_row_of_frames():
    """Asked for "ONE single horizontal row" the model sometimes draws two anyway. Measured on a
    real 1152x512 sheet that came back as 3 columns by 2 rows.

    Cutting that by columns alone puts two stacked characters into every frame - and because every
    frame then holds the same pair, they agree almost perfectly: 99% identical silhouettes, an
    animation that does not move. It measured better than the working version right up until the
    frames were laid out and looked at.
    """
    from game_studio.sprites import slice_sheet

    top = [(100, 40, 300, 200), (500, 40, 700, 200), (900, 40, 1100, 200)]
    bottom = [(100, 300, 300, 460), (500, 300, 700, 460)]
    frames = slice_sheet(sheet(top + bottom))
    assert len(frames) == 3, "the row with more frames wins, and only that row is used"
    # One row's height, not the whole sheet's: a frame holding both rows is the defect above.
    assert frames[0].height < 260, f"two rows were stacked into one frame ({frames[0].height}px)"


def test_a_sheet_holding_one_character_is_refused_rather_than_returned_as_an_animation():
    """A single frame is not an animation, and a caller handed one would blit the same image
    forever believing it had a walk cycle."""
    from game_studio.sprites import slice_sheet

    assert slice_sheet(sheet([(400, 150, 700, 400)])) == []


def test_a_sheet_whose_background_was_never_a_green_screen_is_refused():
    """The same guard the single-sprite cut has, reached through the same code. A dark subject on a
    dark backdrop cannot be keyed, and slicing what comes back would produce frames with holes
    punched through them."""
    from game_studio.sprites import slice_sheet

    assert slice_sheet(sheet([(100, 150, 300, 400), (500, 150, 700, 400)],
                             backdrop=(38, 40, 46), subject=(20, 24, 30))) == []


def test_a_limb_reaching_towards_the_next_frame_does_not_become_its_own_frame():
    """Frames are separated by a gap, and a tail crossing most of one does not end the frame."""
    from game_studio.sprites import slice_sheet

    frames = slice_sheet(sheet([(100, 150, 300, 400), (304, 260, 330, 280),
                                (500, 150, 700, 400)]))
    assert len(frames) == 2, "a 4px break is a tail, not a frame boundary"


def test_the_sheet_prompt_asks_for_several_characters_where_the_sprite_prompt_forbids_them():
    """The sprite negative bans "multiple objects, collage, duplicate" because a sprite is one
    object. A sheet is several near-identical characters in one image, so leaving those in fights
    the request - the same words, opposite meanings, one image model."""
    from game_studio.sprites import (
        SHEET_MAX_FRAMES,
        SHEET_MIN_FRAMES,
        compose_sheet_prompt,
        sheet_size,
    )

    positive, negative = compose_sheet_prompt("orange cat", "a walk cycle", "right", 3, "flat art")
    assert "3 frame" in positive and "a walk cycle" in positive and "flat art" in positive
    assert "collage" not in negative and "duplicate" not in negative
    assert "different characters" in negative and "second row" in negative
    # The facing contract still applies: one facing for the whole animation, so one rotation rule.
    assert "RIGHT" in positive

    # A model asking for 40 frames gets a sheet, not a 15,000px canvas.
    assert f"{SHEET_MAX_FRAMES} frame" in compose_sheet_prompt("x", "y", "right", 99)[0]
    assert f"{SHEET_MIN_FRAMES} frame" in compose_sheet_prompt("x", "y", "right", 1)[0]
    assert sheet_size(99) == sheet_size(SHEET_MAX_FRAMES)
    assert sheet_size(4)[0] > sheet_size(2)[0], "more frames need more canvas"


def test_frame_geometry_is_plain_python_so_it_can_reach_the_sprite_manifest():
    """Measured: the first frame landed on disk, then the manifest write threw
    "Object of type int64 is not JSON serializable" - leaving a PNG nothing knew about. The spans
    come from numpy, and everything derived from them travels into that manifest as geometry."""
    import json

    from game_studio.sprites import slice_sheet

    frames = slice_sheet(sheet([(100, 150, 300, 400), (500, 150, 700, 400)]))
    assert frames
    json.dumps([{"width": f.width, "height": f.height,
                 "removed_share": round(f.removed_share, 3)} for f in frames])


def test_frames_that_never_moved_are_measured_rather_than_assumed_to_be_an_animation():
    """The sheet makes the frames CONSISTENT reliably. Whether they MOVE is not reliable, and the
    two are easy to confuse because the failure looks like success from every angle except this one.

    Measured on real sheets: the same wording at two seeds gave a real walk cycle (13.9% spread) and
    three near-identical drawings (0.8%); across three phrasings at two seeds each, five of six came
    back under 2%. No wording tried changed it - it is a property of the generation, not the
    request. So it is measured, a second seed is tried, and a sheet that still has not moved says so
    instead of being handed over as a walk cycle.
    """
    from game_studio.sprites import SHEET_MIN_POSE_SPREAD, pose_spread, slice_sheet

    walking = slice_sheet(sheet([(100, 150, 300, 400), (500, 150, 700, 340),
                                 (900, 120, 1100, 400)]))
    frozen = slice_sheet(sheet([(100, 150, 300, 400), (500, 150, 700, 400),
                                (900, 150, 1100, 400)]))
    assert len(walking) == 3 and len(frozen) == 3
    assert pose_spread(frozen) < SHEET_MIN_POSE_SPREAD, "identical poses are not an animation"
    assert pose_spread(walking) > SHEET_MIN_POSE_SPREAD

    # The threshold has to sit in open space, or it reports noise as motion and motion as noise.
    # The real measurements cluster at 0.8-1.7% and 9.9-13.9%, with nothing between.
    assert 0.02 < SHEET_MIN_POSE_SPREAD < 0.09
    # One frame is not an animation, and nothing may divide by zero on the way to saying so.
    assert pose_spread(walking[:1]) == 0.0 and pose_spread([]) == 0.0


def test_only_a_character_gets_an_animation_sheet():
    """A wall has no walk cycle. Asked for one it comes back as the same wall drawn three times,
    those get written out as three "frames", and the game blits them in sequence - a wall flickering
    between three near-identical images, with three quarters of the run's image budget spent on it.

    The roles come from art_memory.ROLES, where "obstacle" is defined as 벽·블록·장애물 and "terrain"
    as 바닥·타일·플랫폼 - the two a wall is actually named as.
    """
    from game_studio.agent_tools import STATIC_ROLES, _animatable

    for role in STATIC_ROLES:
        assert _animatable(role, "wall"), role
    for role in ("player", "enemy", "projectile", "effect"):
        assert _animatable(role, "hero") == "", role

    # The refusal has to name the tool that does work, or the agent retries the same call.
    assert "generate_comfyui_image" in _animatable("obstacle", "brick")


def test_a_static_object_is_caught_even_when_the_agent_names_no_role():
    """role is optional on the tool, so a guard that only reads it is a guard that can be skipped by
    leaving an argument out. The name is the fallback, the same one the art memory uses."""
    from game_studio.agent_tools import _animatable

    assert _animatable("", "brick-wall"), "the name says wall even when the role is missing"
    assert _animatable("", "floor-tile")
    assert _animatable("", "player") == ""


def test_the_single_sprite_path_still_asks_for_exactly_one_object():
    """The sheet prompt deliberately drops "multiple objects, collage, duplicate" from the negative,
    because several near-identical characters in one image is the point there. That relaxation must
    not reach the ordinary sprite path, where one object is still the whole contract - a wall tile
    generated as a collage of four wall tiles tiles wrongly and reads as a texture, not a block."""
    from game_studio.sprites import _SPRITE_NEGATIVE, compose_prompt

    for banned in ("multiple objects", "collage", "duplicate"):
        assert banned in _SPRITE_NEGATIVE, banned
    positive, negative = compose_prompt("a brick wall tile", "sprite", "none", "flat 2D")
    assert "single game sprite" in positive
    assert "sprite sheet" not in positive and "animation sheet" not in positive
    assert "collage" in negative
