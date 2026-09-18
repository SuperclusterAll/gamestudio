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

    # The loudest case used to be the silent one. When the cut is REFUSED - the model painted a
    # scene instead of a green screen - the original opaque image is kept, and this returned ""
    # because there was no transparency to judge. Measured on a delivered game: boss-slime.png
    # shipped as a 1024x1024 rectangle with its whole scene baked in and nothing said so.
    refused = keyed_out({"kind": "sprite", "transparent": False, "width": 1024, "height": 1024})
    assert "전혀 제거하지 못했습니다" in refused and "1024x1024" in refused

    # And the subtle one. A cut-out sprite is always SMALLER than the canvas it came from, because
    # the trim shrinks it to the art. One that still spans the whole canvas has opaque pixels
    # touching every edge. Measured: mushroom.png came back 1024x1024 at 55% opaque - under the
    # opacity limit above, and still carrying its scene.
    untrimmed = keyed_out({"kind": "sprite", "transparent": True, "width": 1024, "height": 1024,
                           "removed_share": 0.44, "source_width": 1024, "source_height": 1024})
    assert "배경 일부가 남았습니다" in untrimmed
    # A real cut-out is smaller than its canvas and says nothing.
    assert keyed_out({"kind": "sprite", "transparent": True, "width": 342, "height": 264,
                      "removed_share": 0.61, "source_width": 512, "source_height": 512}) == ""

    # A backdrop is meant to be opaque, whatever its geometry.
    assert keyed_out({"kind": "backdrop", "transparent": False}) == ""
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


def test_repeated_poses_are_dropped_instead_of_being_counted_as_animation():
    """The sheet makes the frames CONSISTENT reliably. Whether they MOVE is not reliable, and the
    two are easy to confuse because the failure looks like success from every angle except this one.

    The measurement that settled it, from a real four-frame sheet of an armoured soldier: three
    identical standing poses and a fourth drawn from BEHIND - the model had produced a character
    turnaround rather than a walk cycle. Averaging every frame against the first let that back view
    alone carry the set over the threshold, so three quarters of the "animation" was one still image
    and the check passed it.

    Counting distinct poses catches it, and dropping the repeats is better than failing the set: a
    four-frame sheet holding three real poses becomes a clean three-frame animation.
    """
    from game_studio.sprites import (
        DUPLICATE_FRAME_SHARE,
        SHEET_MIN_DISTINCT_POSES,
        distinct_poses,
        pose_spread,
        slice_sheet,
    )

    walking = slice_sheet(sheet([(100, 150, 300, 400), (500, 150, 700, 340),
                                 (900, 120, 1100, 400)]))
    frozen = slice_sheet(sheet([(100, 150, 300, 400), (500, 150, 700, 400),
                                (900, 150, 1100, 400)]))
    assert len(walking) == 3 and len(frozen) == 3
    assert len(distinct_poses(walking)) == 3, "three real poses survive"
    assert len(distinct_poses(frozen)) == 1, "one pose drawn three times is one pose"

    # The turnaround case: three repeats plus one very different frame. The set must not pass on the
    # strength of the outlier.
    turnaround = slice_sheet(sheet([(100, 150, 300, 400), (500, 150, 700, 400),
                                    (900, 150, 1100, 400), (1300, 150, 1500, 250)],
                                   size=(1600, 512)))
    assert len(turnaround) == 4
    assert len(distinct_poses(turnaround)) == 2 < SHEET_MIN_DISTINCT_POSES

    # The threshold sits in open space. Measured on a real sheet: the repeated pair differed by
    # 0.9% and every genuinely different pair by more than 11.9% - ten times the gap.
    assert 0.02 <= DUPLICATE_FRAME_SHARE <= 0.09
    assert pose_spread(frozen) < DUPLICATE_FRAME_SHARE < pose_spread(walking)
    # One frame is not an animation, and nothing may divide by zero on the way to saying so.
    assert pose_spread(walking[:1]) == 0.0 and pose_spread([]) == 0.0
    assert distinct_poses([]) == []


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


def test_both_engines_are_told_to_animate_characters_rather_than_offered_the_option():
    """The failure this pins down was in the wording, not the code. The sheet tool existed, worked,
    and went unused: the Godot prompt said "for a character that animates, call
    generate_animation_frames" - a permission - and the HTML5 prompt did not mention it at all. A
    measured run generated six characters as six single stills, each frozen in an action pose, and
    every one of them slid across the screen without moving a limb.

    Both prompts now carry the same paragraph, from one constant, so neither can drift back.
    """
    from game_studio.prompts import ART_TOOLING, CODE_SYSTEM, GODOT_CODE_SYSTEM

    for prompt in (CODE_SYSTEM, GODOT_CODE_SYSTEM):
        assert ART_TOOLING in prompt
    # Which tool for which object has to be stated, or a wall becomes a four-frame animation of a
    # wall and a walking enemy becomes four unrelated enemies.
    # Three, not four. A walk cycle reads from three poses, and the fourth costs a quarter more
    # canvas (1536x512 against 1152x512) for a frame most players never resolve.
    assert "generate_animation_frames" in ART_TOOLING and "frames=3" in ART_TOOLING
    assert "generate_comfyui_image" in ART_TOOLING
    for static in ("walls", "floors", "tiles", "icons", "backdrops"):
        assert static in ART_TOOLING, static
    # Left and right come from one frame set. Generating a second, mirrored set would spend the
    # image budget twice for something a transform does exactly.
    assert "ctx.scale(-1, 1)" in ART_TOOLING and "flip_h" in ART_TOOLING


def test_the_image_budget_survives_two_animated_characters_and_a_scene():
    """Four PNGs per character is the point of the sheet, not a cost of it - they are one model call
    and one generation. But they are still four files against a cap that counts files, and at 8 a
    player plus one enemy left nothing for the backdrop, the pickups or the walls."""
    from game_studio.agent_tools import _MAX_GENERATED_IMAGES

    assert _MAX_GENERATED_IMAGES >= 12, (
        f"{_MAX_GENERATED_IMAGES} leaves no room for scenery once characters animate")


def test_the_comfyui_budget_grows_with_the_canvas_it_is_waiting_on():
    """The timeout covers queue wait as well as generation, and it was tuned for one 512x512
    sprite. An animation sheet is up to six times the pixels - measured, a 1536x512 sheet generates
    in 59s against 31s for the sprite - and a flat 180s ran out while the job was still sitting in
    the queue behind another one, reporting a timeout for work that had not started."""
    def budget(width, height, base=180):
        return max(base, round(base * (width * height) / (512 * 512) / 2))

    assert budget(512, 512) == 180, "the per-sprite setting still means what it always did"
    # Never below the configured floor, however small the canvas.
    assert budget(256, 256) == 180
    # And comfortably above measured generation time for the sheets that broke it.
    assert budget(1536, 512) >= 59 * 3
    assert budget(2304, 512) > budget(1536, 512) > budget(512, 512)


def test_each_frame_is_asked_for_by_the_limbs_that_move_in_it():
    """The request used to say "each frame a different moment of a walk cycle" and leave the rest to
    the model - which is asking it to invent both the animation and what an animation is. It
    answered the way that deserves: a measured four-frame soldier came back as two poses that
    differed only in that the second had drawn a sword. Technically different. Not walking.

    A walk cycle is arms and legs in a different position and nothing else moving, so each frame now
    names its own limb positions. That also gives the model N different instructions instead of one
    instruction repeated N times, which is what let it repeat the drawing.
    """
    from game_studio.sprites import compose_sheet_prompt, motion_poses

    positive, _ = compose_sheet_prompt("a soldier", "a walk cycle", "right", 4, "flat art")
    for index in (1, 2, 3, 4):
        assert f"frame {index} " in positive, index
    assert "leg" in positive and "arm" in positive
    # The other half of the instruction: what must NOT change. Without it the model finds its own
    # way to make frames differ, and a weapon appearing is a cheaper difference than a stride.
    assert "only the arm and leg POSES change" in positive

    # Enough named poses for the largest sheet, and they are asked for in order.
    from game_studio.sprites import SHEET_MAX_FRAMES

    assert len(motion_poses("walk", SHEET_MAX_FRAMES)) == SHEET_MAX_FRAMES
    assert motion_poses("walk", 2) == motion_poses("walk", 4)[:2]

    # Walking is the two legs TAKING TURNS, with each arm swinging opposite its own leg. The first
    # version of this list said "the near leg" every time - naming no side at all - and frames 1 and
    # 4 both put the near leg forward. Measured: three of four frames came back as the same stance.
    # The model was asked for one pose in four wordings and it obliged.
    walk = motion_poses("walk", 4)
    assert walk[0].startswith("LEFT leg forward") and "RIGHT arm forward" in walk[0]
    assert walk[3].startswith("RIGHT leg forward") and "LEFT arm forward" in walk[3]
    assert walk[0] != walk[3], "the two contact poses are mirrors, not repeats"
    # Every frame names a side, so none of them can collapse into another by accident.
    for index, pose in enumerate(motion_poses("walk", SHEET_MAX_FRAMES)):
        assert "LEFT" in pose and "RIGHT" in pose, index
    for index, pose in enumerate(motion_poses("run", SHEET_MAX_FRAMES)):
        assert "LEFT" in pose and "RIGHT" in pose, index

    # And once as a rule, so a model that half-follows the frame list still has the principle.
    assert "ALTERNATE between frames" in positive
    assert "each arm swings opposite its own leg" in positive

    # The motions a 2D game actually animates, named in either language the studio runs in.
    for motion in ("a walk cycle", "걷는 모습", "running", "a jump", "점프", "an attack swing",
                   "idle breathing"):
        assert motion_poses(motion, 3), motion

    # An unrecognised motion gets no list rather than a wrong one: walk-cycle leg positions on a
    # spinning coin would be worse than saying nothing.
    assert motion_poses("꼬리 흔들기", 3) == []
    vague, _ = compose_sheet_prompt("a coin", "spinning", "none", 3)
    assert "each frame a different moment of spinning" in vague and "frame 1 " not in vague


def test_a_sheet_asks_for_the_character_alone_and_not_the_things_the_game_moves():
    """Measured: a request for a football player came back with a ball at his feet in every frame.
    Cut out, that ball is welded to the player and follows him around the pitch while the real ball
    moves separately - the "star above the head" defect wearing a different object.

    The cause was structural. Making several characters in one image possible meant dropping
    "multiple objects, collage, duplicate" from the sprite negative, and that took the guard against
    a second THING with it. The ban on a second character is lifted; the ban on a second object is
    put back by name.
    """
    from game_studio.sprites import _SPRITE_NEGATIVE, compose_sheet_prompt

    positive, negative = compose_sheet_prompt("a football player", "a walk cycle", "right", 3)
    for banned in ("ball", "football", "sports equipment", "props", "a second separate object"):
        assert banned in negative, banned
    assert "character ALONE in every frame" in positive
    # Several characters are still what a sheet is for, so that ban stays lifted.
    assert "multiple objects" in _SPRITE_NEGATIVE and "multiple objects" not in negative


def test_the_agent_is_told_that_one_sprite_is_one_thing_the_game_positions():
    """No negative saves a prompt that asks for the wrong picture. If the agent writes "a player
    kicking a ball", the ball gets drawn - so the instruction has to reach the agent, in both
    engines and in both image tools."""
    from game_studio.agent_tools import generate_animation_frames, generate_comfyui_image
    from game_studio.prompts import CODE_SYSTEM, GODOT_CODE_SYSTEM

    for prompt in (CODE_SYSTEM, GODOT_CODE_SYSTEM):
        assert "one thing your code positions" in prompt
        assert "kicking a ball" in prompt, "the concrete example is what makes the rule land"
    # @tool replaces __doc__ with the wrapper's; what the model is shown is .description.
    assert "kicking a ball" in generate_animation_frames.description
    assert "gets its own call" in generate_comfyui_image.description


def test_the_sheet_asks_for_a_whole_character_and_not_only_the_parts_that_move():
    """Naming the limbs is what made the frames differ. Naming ONLY the limbs is what made the
    character disappear around them.

    Measured: with "only the arms and legs change between frames, everything else identical" plus
    four frames of "the near leg ... the opposite arm ...", every word of the request pointed at two
    body parts - and the model drew two body parts. A football sheet came back with four HEADLESS
    players, and the raw image was headless before any cutting touched it.

    "Everything else identical" does not put a head in the image. "head" does.
    """
    from game_studio.sprites import compose_sheet_prompt

    positive, negative = compose_sheet_prompt("a football player", "a walk cycle", "right", 4)
    assert "the complete character from head to feet in every frame" in positive
    assert "the head and face always drawn" in positive
    # What stays the same is listed by name, not left as a residual category.
    assert "the head, face, body, colours and clothing are identical" in positive
    # And guarded from the other side, because this one is invisible until someone looks.
    for banned in ("headless", "missing head", "cut off at the neck", "body parts only"):
        assert banned in negative, banned


def test_the_subject_gives_way_so_the_staging_never_has_to():
    """Truncation is silent and cuts the TAIL. That cost a run its green screen once: the staging
    clause sat last, a long pose list pushed the prompt past the limit, and the model drew a white
    backdrop the chroma key then correctly refused.

    Two fixes, and this is the first one. The clauses the pipeline depends on are fixed in size;
    the caller's description is the only elastic part, so it is the part that gives way - by exactly
    the amount needed and no more. Measured: the fixed clauses of a six-frame sheet are 1,905
    characters on their own, a sprite's are 551 and a backdrop's 188.
    """
    from game_studio.agent_tools import MIN_SUBJECT_CHARS, PROMPT_SEND_LIMIT, fit_subject
    from game_studio.sprites import (
        SHEET_MAX_FRAMES,
        compose_prompt,
        compose_sheet_prompt,
        house_style,
    )

    style = house_style("x" * 120, {f"colour_{i}": "#000000" for i in range(5)})
    runaway = "a " * 3000
    cases = [
        (compose_prompt, ("sprite", "right", style)),
        (compose_prompt, ("backdrop", "none", style)),
        (compose_sheet_prompt, ("a walk cycle", "right", 2, style)),
        (compose_sheet_prompt, ("a walk cycle", "right", SHEET_MAX_FRAMES, style)),
    ]
    for compose, args in cases:
        subject = fit_subject(compose, runaway, *args)
        assert len(compose(subject, *args)[0]) <= PROMPT_SEND_LIMIT, args
        assert len(subject) >= MIN_SUBJECT_CHARS, "a request with no subject in it is not a request"

    # A description that already fits is untouched - this shortens, it does not reformat.
    short = "a knight in blue steel plate armour"
    assert fit_subject(compose_prompt, short, "sprite", "right", style) == short


def test_no_composed_prompt_is_long_enough_to_be_truncated():
    """Truncation is silent, cuts the TAIL, and the tail used to hold the staging clause. A
    six-frame sheet composed to 1,460 characters against a 1,200 limit, so "on a flat solid #00FF00
    green screen background" never reached the model - it drew a white backdrop and the chroma key
    correctly refused an image it could not key. Nothing reported a truncated prompt, because
    nothing was looking.

    Two independent guards, because either alone is one edit away from failing: the things the
    pipeline depends on come first in the prompt, and the limit clears the longest prompt composed.
    """
    from game_studio.agent_tools import PROMPT_SEND_LIMIT
    from game_studio.sprites import (
        SHEET_MAX_FRAMES,
        compose_prompt,
        compose_sheet_prompt,
        house_style,
    )

    style = house_style("flat 2D game art, clean readable shapes, solid dark outline",
                        {"kit_red": "#D93A3A", "white": "#FFFFFF", "grass": "#4C9A3A"})
    longest = max(
        [compose_prompt("a football player in a red kit", kind, "right", style)[0]
         for kind in ("sprite", "backdrop")]
        + [compose_sheet_prompt("a football player in a red kit", motion, "right", frames, style)[0]
           for motion in ("a walk cycle", "running", "an attack swing")
           for frames in range(2, SHEET_MAX_FRAMES + 1)],
        key=len)
    assert len(longest) <= PROMPT_SEND_LIMIT, f"{len(longest)} characters would be cut"

    # And the clause the cut depends on is near the front, not near the cliff.
    sheet, _ = compose_sheet_prompt("a player", "a walk cycle", "right", SHEET_MAX_FRAMES, style)
    assert sheet.index("green screen") < len(sheet) // 2, "staging must survive any truncation"


def test_the_pose_list_matches_the_body_it_is_describing():
    """A walk cycle described for the wrong body is noise, and noise is what a model averages away.

    Measured across ten real cases: every walking one that failed had been told about "the RIGHT arm
    swung forward" and "the heel down" - a cat among them, which has neither. Four frames of
    instructions aimed at limbs the subject does not have leaves nothing to vary, and the model drew
    the same animal four times. The cases that passed were the ones whose subject happened to match
    the vocabulary.
    """
    from game_studio.sprites import motion_poses

    biped = motion_poses("walk", 4, "a football player in a red kit")
    assert biped and "arm" in biped[0] and "FRONT leg" not in biped[0]

    quadruped = motion_poses("walk", 4, "a horse")
    assert quadruped and "FRONT leg" in quadruped[0] and "HIND leg" in quadruped[0]
    assert "arm" not in " ".join(quadruped), "a horse has no arms to swing"
    assert motion_poses("walk", 4, "말 캐릭터")[0] == quadruped[0], "either language"

    # Naming a species is NOT enough. Game art draws animals upright far more often than on all
    # fours - the measured cat sheet came back walking on two legs while being told about its front
    # and hind legs. Only an explicit four-legged request, or an animal nobody draws standing.
    assert motion_poses("walk", 4, "a small orange cat with a striped tail") == biped
    assert "FRONT leg" in motion_poses("walk", 4, "a cat on all fours")[0]

    # No legs means deforming, not stepping - and NOT silence. Removing this subject's (wrong) leg
    # instructions dropped it from four distinct poses to ONE, because those instructions had been
    # the only thing asking for any change at all.
    limbless = motion_poses("idle breathing", 4, "a translucent blue slime blob")
    assert limbless and "squash" in limbless[0]
    assert "leg" not in " ".join(limbless)
    # An attack is not locomotion, so a limbless thing still gets the generic attack poses.
    assert motion_poses("an attack swing", 4, "a slime") == motion_poses("an attack swing", 4, "x")

    # Words, not substrings. "a football player" contains "ball", and matching on substrings
    # classified him as limbless and took his walk cycle away.
    assert motion_poses("walk", 4, "a football player")
    assert motion_poses("walk", 4, "a catapult"), "'cat' inside 'catapult' is not a cat"

    # A subject the lists do not recognise still gets the default vocabulary rather than nothing:
    # most game characters are humanoid, and a wrong guess here costs one less-varied sheet.
    assert motion_poses("walk", 4, "a boxy yellow robot") == biped


def test_a_surviving_background_is_re_rolled_instead_of_handed_back(monkeypatch, tmp_path):
    """Measured across 73 real sprite prompts: NOT ONE described a scene. "뿔 두 개 달린 붉은 대형
    슬라임, 크고 둥근 몸통" is exactly what the tool asks for, and its cut was refused anyway.

    So the agent's wording is not the problem, and telling the agent to reword it spends a model
    call and a round trip to arrive back at the same request. Another seed costs GPU time and
    nothing else - and the run has a clock for that, which is what stops this being unbounded.
    """
    from game_studio import agent_tools

    workspace = tmp_path / "game_html5_deadbeef"
    (workspace / "assets").mkdir(parents=True)
    monkeypatch.setattr(agent_tools, "_IMAGE_SECONDS", {})
    monkeypatch.setattr(agent_tools, "_note", lambda *_a: None)

    seeds: list[int] = []

    def render(positive, negative, seed, width, height, workspace_dir=""):
        seeds.append(seed)
        return b"png-bytes"

    # The first attempt keeps its background, the second does not.
    def finish(target, content, kind, facing):
        target.write_bytes(content)
        first = len(seeds) == 1
        return {"kind": kind, "facing": facing, "transparent": not first,
                "width": 512 if first else 260, "height": 512 if first else 327,
                "removed_share": 0.0 if first else 0.62,
                "source_width": 512, "source_height": 512}

    monkeypatch.setattr(agent_tools, "_render_png", render)
    monkeypatch.setattr(agent_tools, "_finish_asset", finish)

    state = {"generate_images": True, "output_dir": str(tmp_path),
             "workspace_dir": str(workspace), "art": {}}
    monkeypatch.setattr(agent_tools, "_comfy_asset_dir", lambda _s: workspace / "assets")

    result = agent_tools._generate_comfyui_image(
        prompt="뿔 두 개 달린 붉은 대형 슬라임", state=state, asset_name="boss", seed=11)

    assert len(seeds) == 2 and seeds[0] != seeds[1], "a different seed, not the same one again"
    assert "260x327" in result and "배경 제거됨" in result, "the good attempt is what is kept"
    assert "배경" not in result.split("—")[0] or "제거됨" in result

    # Twice and no more. A third attempt would be a third minute of GPU on a subject the model has
    # now failed twice, and the run's clock is needed for the game.
    seeds.clear()
    monkeypatch.setattr(agent_tools, "_finish_asset",
                        lambda target, content, kind, facing: (
                            target.write_bytes(content),
                            {"kind": kind, "facing": facing, "transparent": False,
                             "width": 512, "height": 512})[1])
    stubborn = agent_tools._generate_comfyui_image(
        prompt="뿔 두 개 달린 붉은 대형 슬라임", state=state, asset_name="boss2", seed=11)
    assert len(seeds) == 2
    assert "두 번 시도했지만" in stubborn and "Canvas" in stubborn


def test_image_prompts_are_asked_for_in_english_everywhere_they_are_written():
    """Measured over 46 paired generations of the same five objects, each described once in the
    Korean the agent actually wrote and once in English, across four seeds: every English subject
    keyed cleanly (26/26), three Korean ones did not (23/26). Median removal was identical at 84%
    when both worked.

    That is NOT a proven difference - Fisher's exact on the pooled result gives p=0.24, and a
    three-case gap at this sample size is well inside chance. It is acted on because the decision is
    asymmetric: English was never worse, every observed failure was in the other arm, and asking for
    it costs one sentence and no model call. A failure now costs a re-roll rather than a broken
    asset, so the saving is time rather than correctness.

    Three places write these prompts, and all three have to say it or the run is inconsistent.
    """
    from game_studio.agent_tools import generate_animation_frames, generate_comfyui_image
    from game_studio.prompts import ART_SYSTEM, CODE_SYSTEM, GODOT_CODE_SYSTEM

    # The art director, whose asset_plan descriptions the prompts are built from.
    assert "ENGLISH" in ART_SYSTEM
    # The code agent, in both engines.
    for prompt in (CODE_SYSTEM, GODOT_CODE_SYSTEM):
        assert "ENGLISH" in prompt
    # And the tools themselves, which is what the model actually reads at the moment it writes one.
    assert "ENGLISH" in generate_comfyui_image.description
    assert "ENGLISH" in generate_animation_frames.description

    # The game's own text stays Korean - this is about what the image model is asked for, not about
    # what the player reads.
    assert "Korean" in CODE_SYSTEM or "한국어" in ART_SYSTEM


def test_a_green_subject_gets_a_magenta_screen_instead_of_being_refused():
    """The limitation this removes was documented as unfixable and pushed onto the caller: the tool
    used to tell the agent "do not ask for a bright green object", because green that runs into the
    backdrop with no outline between them is cut away with it.

    A 2D game's bestiary is largely green - slimes, zombies, frogs, goblins - so that guard was
    losing exactly where it was needed. Magenta is the standard second key because nothing in a
    green subject is near it. Verified end to end: a saturated green slime and a green zombie both
    cut cleanly at 52% opaque.
    """
    from game_studio.sprites import (
        GREEN_KEY,
        MAGENTA_KEY,
        compose_prompt,
        compose_sheet_prompt,
        key_for,
    )

    for green in ("a green slime with angry eyes", "a zombie in torn clothes", "초록 개구리 캐릭터",
                  "an emerald dragon", "a goblin with a club"):
        assert key_for(green) == MAGENTA_KEY, green
    for other in ("a red brick wall", "an orange cat", "a knight in blue armour", "a gold coin"):
        assert key_for(other) == GREEN_KEY, other

    # Both prompt builders honour it, or an animated slime loses what a still one keeps.
    assert "magenta" in compose_prompt("a green slime", "sprite", "none", "flat art")[0]
    assert "magenta" in compose_sheet_prompt("a green slime", "a walk cycle", "right", 3)[0]
    assert "green screen" in compose_prompt("a red brick wall", "sprite", "none", "flat art")[0]


def test_the_cut_accepts_whichever_key_the_model_actually_painted():
    """Which key was asked for is not the question - the model may ignore it. What matters is the
    colour on the canvas, so the cut reads that and accepts either, by dominance rather than by
    distance to a nominal value. Measured green screens came back as (6,224,10), (22,254,91) and
    (117,212,113): unmistakably green, never the pure key."""
    from game_studio.sprites import _is_key

    for green in ((6, 224, 10), (22, 254, 91), (117, 212, 113)):
        assert _is_key(green), green
    for magenta in ((255, 0, 255), (230, 40, 220), (200, 90, 190)):
        assert _is_key(magenta), magenta
    # And the guard that makes the cut safe to trust still refuses everything else - a dark backdrop
    # behind a dark subject is what walked the fill straight through a car's bodywork.
    for flat in ((38, 40, 46), (245, 245, 245), (120, 120, 200), (200, 170, 120)):
        assert not _is_key(flat), flat


def test_a_sprite_is_not_generated_at_four_times_the_canvas_it_is_drawn_at():
    """The canvas decides the generation time - 1024x1024 is four times the pixels of 512x512 and
    takes about four times as long - and a sprite is drawn on screen at something like 64px.
    Measured: agents were asking for 1024x1024 objects, paying most of a run's ten-minute image
    budget for detail no player can see. A bigger canvas also gives the model more room to invent a
    scene in, which is the failure the cut exists to catch."""
    from game_studio.agent_tools import BACKDROP_CANVAS, SPRITE_CANVAS

    assert SPRITE_CANVAS <= 512 < BACKDROP_CANVAS, "a backdrop is the one thing seen full size"
    # Four times the pixels is four times the wait.
    assert (BACKDROP_CANVAS ** 2) / (SPRITE_CANVAS ** 2) >= 4


def mirrored_sheet():
    """A sheet whose second character is drawn facing the other way."""
    import numpy as np
    from PIL import Image as PILImage

    image = PILImage.new("RGB", (1152, 512), (10, 220, 15))
    pixels = np.asarray(image).copy()
    # An L-shaped figure: a tall body with a foot sticking out to one side, so mirroring it is
    # visible in the silhouette rather than only in the colours.
    def figure(x0, flip):
        # The body is centred and symmetric; only the foot says which way it faces, which is what
        # makes mirroring visible in the silhouette rather than only in the colours.
        pixels[150:400, x0 + 60:x0 + 140] = (200, 40, 40)
        foot = slice(x0, x0 + 60) if flip else slice(x0 + 140, x0 + 210)
        pixels[360:400, foot] = (200, 40, 40)
    figure(100, False)
    figure(500, True)
    figure(900, False)
    buffer = BytesIO()
    PILImage.fromarray(pixels).save(buffer, format="PNG")
    return buffer.getvalue()


def test_a_frame_drawn_facing_the_other_way_is_turned_back():
    """Asking for one facing is not the same as getting it. Measured on a delivered game: the
    ghost's frames 2, 3 and 4 all matched frame 1's MIRROR better than frame 1 itself (60/53,
    78/67, 86/69), and the player dinosaur turned around halfway through its own walk cycle - while
    the sprite manifest recorded every frame as facing right. The game flips sprites by that
    recorded facing, so a character walking right played its cycle facing backwards.

    Detecting facing from pixels in general is slow and wrong often enough to be useless, which is
    why this codebase fixes facing as a contract instead. CONSISTENCY is a different and much easier
    question: frame 1 is the reference and every later frame only has to agree with it. A frame that
    disagrees is mirrored back rather than discarded - the pose it holds is still a real pose.
    """
    import numpy as np
    from PIL import Image as PILImage

    from game_studio.sprites import slice_sheet

    frames = slice_sheet(mirrored_sheet())
    assert len(frames) == 3
    masks = [np.asarray(PILImage.open(BytesIO(f.png)).convert("RGBA"))[..., 3] > 8
             for f in frames]

    def agreement(a, b):
        return (a & b).sum() / max(1, (a | b).sum())

    for index, mask in enumerate(masks[1:], 2):
        direct, mirror = agreement(masks[0], mask), agreement(masks[0], mask[:, ::-1])
        assert direct >= mirror, f"frame {index} still faces the other way"


def test_a_near_symmetric_subject_is_left_alone():
    """A round slime matches its own mirror almost exactly whichever way it is drawn. Measured: one
    real slime frame scored 89% mirrored against 86% direct - three points, which is noise, and
    flipping on it would be noise pretending to be a fix. The margin is what stops that."""
    from game_studio.sprites import MIRROR_MARGIN, slice_sheet, unmirror

    assert 0.02 < MIRROR_MARGIN <= 0.10

    # A perfectly symmetric blob: direct and mirrored agree exactly, so nothing may change.
    circle = slice_sheet(sheet([(100, 150, 300, 400), (500, 150, 700, 400)]))
    assert len(circle) == 2
    assert [f.png for f in unmirror(circle)] == [f.png for f in circle]
    assert unmirror(circle[:1]) == circle[:1] and unmirror([]) == []


def test_a_frame_wearing_different_colours_is_dropped():
    """The prompt already says the clothing and colours are identical in every frame. The model
    agrees and then does it anyway: a measured four-frame soldier came back with a cream skirt in
    frame 1 and a red one in the other three, which flickers on every loop of the walk.

    Surveyed across 140 real frames on disk, a frame's colour distance to its NEAREST sibling has a
    median of 0.02 and a 95th percentile of 0.05. Exactly two frames in that set sat outside: the
    cream skirt at 0.27, and a slime whose frame 1 kept a cyan corner of its background at 0.64.
    Nothing lands between 0.07 and 0.27, so the line sits in open space - and it catches two
    different defects, because a changed costume and a surviving background are the same signal
    from the frame's point of view: this one does not belong with the others.
    """
    from game_studio.sprites import COLOUR_DRIFT, Cutout, consistent_colours

    def frame(body, trim):
        image = Image.new("RGBA", (120, 200), (0, 0, 0, 0))
        for x in range(30, 90):
            for y in range(20, 140):
                image.putpixel((x, y), (*body, 255))
            for y in range(140, 190):
                image.putpixel((x, y), (*trim, 255))
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        return Cutout(png=buffer.getvalue(), width=120, height=200, removed_share=0.5)

    red, cream = (200, 40, 40), (250, 245, 210)
    frames = [frame((60, 60, 70), cream)] + [frame((60, 60, 70), red) for _ in range(3)]
    kept = consistent_colours(frames)
    assert len(kept) == 3 and frames[0].png not in {f.png for f in kept}

    # A set that agrees with itself is left whole, whatever its colours are.
    assert len(consistent_colours([frame((60, 60, 70), red) for _ in range(4)])) == 4
    # Below three frames there is no majority to disagree with, so nothing is judged.
    assert len(consistent_colours(frames[:2])) == 2

    # Never cuts below two. A set where everything disagrees with everything is a failed sheet, and
    # the caller's own checks report that better than silently handing back one picture.
    motley = [frame((200, 40, 40), (200, 40, 40)), frame((40, 200, 40), (40, 200, 40)),
              frame((40, 40, 200), (40, 40, 200))]
    assert len(consistent_colours(motley)) == 3

    assert 0.07 < COLOUR_DRIFT < 0.27, "the threshold has to sit in the gap the survey found"


def test_the_prompt_names_the_costume_as_something_that_must_not_change():
    from game_studio.sprites import compose_sheet_prompt

    positive, negative = compose_sheet_prompt("a soldier in red armour", "a walk cycle", "right", 4)
    assert "SAME garment in the SAME colour on every frame" in positive
    for banned in ("recoloured costume", "changing outfit colour", "different clothing"):
        assert banned in negative, banned


def test_two_distinct_poses_is_a_short_animation_and_not_a_failure():
    """"Fewer than asked" and "not an animation" are different problems, and the warning used to
    call them the same thing - telling the agent to abandon animation and use a single still.

    Two poses IS a walk cycle: contact and passing alternating is the minimal one, and games have
    shipped on it for forty years. One pose is a still image whatever it is called.
    """
    import inspect

    from game_studio import agent_tools

    source = inspect.getsource(agent_tools._generate_animation_frames)
    assert "최소한의 걷기 주기는 2프레임입니다" in source
    assert "애니메이션이 아니라 정지 이미지" in source
    # The severe wording is reserved for the case that deserves it.
    severe = source.index("애니메이션이 아니라 정지 이미지")
    mild = source.index("최소한의 걷기 주기는 2프레임입니다")
    assert mild < severe, "the common case is handled before the hopeless one"


def test_everything_the_cut_depends_on_is_in_the_positive_prompt():
    """Z-Image Turbo is distilled and runs at cfg 1, where classifier-free guidance is off and the
    negative branch has nothing to steer away from - the negative prompt is simply inert.

    Measured on this machine, same prompts and seeds: cfg 1 averaged 14.4s with the chroma key
    succeeding 6/6, cfg 2 averaged 24.8s at 5/6. 42% faster and no worse, which is the model card's
    own recommendation ("Guidance should be 0 for the Turbo models").

    So anything the pipeline depends on has to be stated as something to DRAW rather than something
    to avoid. Two of these were learned the hard way: every character came back pointing, and stars
    kept appearing over their heads to be welded on by the cut.
    """
    from game_studio.comfyui import CFG
    from game_studio.sprites import compose_prompt, compose_sheet_prompt

    assert CFG <= 1.0, "above 1 the negative prompt starts working and the timings change"

    sprite, _ = compose_prompt("a knight in blue armour", "sprite", "right", "flat art")
    for essential in ("green screen", "single game sprite", "the subject alone",
                      "nothing floating around it", "arms relaxed at its sides",
                      "no shadow, no ground, no scenery"):
        assert essential in sprite, essential

    sheet, _ = compose_sheet_prompt("a knight", "a walk cycle", "right", 3, "flat art")
    for essential in ("green screen", "the character ALONE in every frame",
                      "EVERY frame faces the SAME way", "nothing floating around the character",
                      "the head and face always drawn", "identical in all 3 frames"):
        assert essential in sheet, essential


def test_a_backdrop_is_staged_as_deliberately_as_a_sprite():
    """A backdrop used to carry no staging at all - the subject, the house style, and a negative
    list that cfg 1 ignores. So the one instruction that matters for a background ("fill the frame,
    and put nobody in it") was never actually given, and a character painted into the backdrop is
    permanent: it cannot move and it cannot be cut out.

    The two prompts are opposites and have to stay opposites. A sprite says "the subject alone";
    a backdrop says "scenery only". Neither may inherit the other's staging.
    """
    from game_studio.sprites import compose_prompt

    backdrop, backdrop_negative = compose_prompt(
        "a ruined castle at dusk", "backdrop", "none", "flat 2D game art")
    sprite, _ = compose_prompt("a knight", "sprite", "right", "flat 2D game art")

    assert backdrop.startswith("a ruined castle at dusk"), "the subject still leads"
    assert "flat 2D game art" in backdrop, "the house style reaches the backdrop too"
    for demand in ("fills the entire canvas", "scenery only", "no characters", "no text"):
        assert demand in backdrop, (
            f"the backdrop must ask for {demand!r} in the positive prompt - at cfg 1 the "
            "negative list is not what decides the picture")
    assert backdrop_negative, "the negative list stays as a second line of defence"

    # Neither staging leaks into the other picture.
    assert "green screen" not in backdrop and "chroma" not in backdrop.lower()
    assert "fills the entire canvas" not in sprite and "scenery only" not in sprite
    assert "the subject alone" in sprite and "the subject alone" not in backdrop


def test_a_re_planned_background_is_generated_as_a_backdrop_not_cut_up_as_a_sprite():
    """A revision generates its required art directly, and that path hardcoded kind="sprite". A
    re-planned "forest-background" was therefore staged on a green screen and then had that screen
    cut away, so the picture came back with its sky removed - or with the cut refused and a warning
    about a background nobody wanted gone."""
    import inspect

    from game_studio.art_memory import guess_role
    from game_studio.graph import _generate_required_art

    source = inspect.getsource(_generate_required_art)
    assert 'kind="sprite"' not in source, "the kind must be read from the object, not assumed"
    assert "guess_role" in source
    assert guess_role("forest-background") == "backdrop"
    # guess_role is the only judge on this path, so what it misses, this path misses. "board-bg"
    # is a name a delivered run actually used.
    assert guess_role("board-bg") == "backdrop"
    assert guess_role("bgone-enemy") == "enemy", "a short hint must not match inside a word"
    assert guess_role("player") == "player"
