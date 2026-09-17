"""The replan/re-execute loop for missing art, and the enforcement that makes it terminate.

The loop it closes: verification says an object has no sprite, the art direction is re-planned to
name that object, the code agent generates it and wires it in, verification passes.

Before this, each of those three steps could fail open. The supervisor preferred `code` because
writing code always looks productive, and the code agent drew a rectangle for an object nothing had
planned. When art was re-planned, the new list was still a menu - the agent could skip the
generation for its own reasons. And nothing downstream checked, so the identical finding came back
a cycle later with the rethink budget already spent. These pin all three shut.
"""

import pytest

from game_studio.required_art import (
    MAX_REQUIRED_ASSETS,
    asset_slug,
    missing_required,
    required_assets,
)

PLAN = {"genre": "슈팅", "mechanics": ["이동", "사격", "회피"], "win_condition": "보스 처치",
        "loss_condition": "피격", "state_transitions": ["시작", "전투", "결과"],
        "acceptance_tests": ["이동", "사격", "보스"]}
PLAYABLE = ("<html><canvas>score restart keydown KeyW KeyA KeyS KeyD "
            "requestAnimationFrame</canvas></html>")


def state_for(tmp_path, **extra) -> dict:
    return {
        "model_id": "m", "engine": "html5", "rethink_cycles": 0, "repair_attempts": 0,
        "art_revised": False, "use_llm": True,
        "concept": {"title": "t", "elevator_pitch": "p", "player_goal": "g",
                    "controls": ["a", "b"], "core_loop": ["1", "2", "3"],
                    "difficulty_curve": "d", "visual_direction": "v"},
        "art": {"palette": {}, "image_prompt": "x", "asset_plan": [], "canvas_effects": []},
        "implementation_plan": PLAN,
        "output_dir": str(tmp_path), "workspace_dir": str(tmp_path),
        **extra,
    }


def test_an_asset_plan_entry_maps_to_the_file_the_generator_would_write():
    """asset_plan entries are "<name>: <description>", and the name has to survive into the same
    slug generate_comfyui_image turns asset_name into - or the check looks for a file that could
    never exist under that name."""
    from game_studio.agent_tools import _safe_asset_stem

    for entry, expected in (("player: 네온 삼각형 우주선", "player"),
                            ("enemy-drone: 붉은 드론", "enemy-drone"),
                            ("Boss Mech: 거대 기갑", "boss-mech"),
                            ("backdrop", "backdrop")):
        assert asset_slug(entry) == expected
        assert asset_slug(entry) == _safe_asset_stem(asset_slug(entry)), \
            "the slug has to be stable through the generator's own naming"


def test_an_asset_named_with_its_extension_does_not_become_a_second_asset():
    """The model names assets and does not always resist spelling them as files. Every
    non-alphanumeric character becomes a hyphen, so "enemy-goomba.png" used to arrive as the stem
    enemy-goomba-png and land on disk as enemy-goomba-png.png.

    One run shipped four sprites written twice under both spellings - generated twice, paid for
    twice - and referenced res://assets/enemy-goomba-png.png with only enemy-goomba.png beside it,
    which is a resource error the moment that code path runs.
    """
    from game_studio.agent_tools import _safe_asset_stem

    for spelling in ("enemy-goomba.png", "enemy goomba png", "enemy-goomba-png", "Enemy-Goomba.PNG"):
        assert _safe_asset_stem(spelling) == "enemy-goomba", spelling
    for spelling in ("backdrop.webp", "hero.jpg", "icon.svg"):
        assert "-" not in _safe_asset_stem(spelling), spelling

    # A plain name is untouched, and a bare extension is a word like any other rather than an
    # extension to strip - the stripping only fires on a trailing token behind something else, so
    # a name can never be emptied into the "asset" fallback by it.
    assert _safe_asset_stem("enemy-goomba") == "enemy-goomba"
    assert _safe_asset_stem("png") == "png"
    assert _safe_asset_stem(".png") == "png"
    assert _safe_asset_stem("") == "asset", "an empty name still has to become something"


def test_only_the_art_that_does_not_exist_yet_is_required():
    """A revision that re-lists the player's own sprite is not asking for it twice, and
    regenerating costs budget the object the finding was actually about then cannot have."""
    plan = ["player: 우주선", "enemy: 드론", "boss: 기갑"]
    assert required_assets(plan, ["player.png"]) == ["enemy", "boss"]
    assert required_assets(plan, ["player.png", "enemy.png", "boss.png"]) == []
    assert required_assets([], ["player.png"]) == []
    # The image budget is small and shared, so one revision cannot claim all of it.
    many = [f"obj{i}: 설명" for i in range(12)]
    assert len(required_assets(many, [])) == MAX_REQUIRED_ASSETS


def test_missing_required_reads_what_is_actually_on_disk(tmp_path):
    assets = tmp_path / "assets"
    assert missing_required(["enemy"], assets) == ["enemy"], "no folder means nothing generated"
    assets.mkdir()
    (assets / "enemy.png").write_bytes(b"x")
    assert missing_required(["enemy", "boss"], assets) == ["boss"]
    assert missing_required([], assets) == []


def test_a_finding_about_absent_art_goes_to_art_even_when_the_supervisor_says_code(tmp_path, monkeypatch):
    """A model asked to fix "the enemy has no sprite" reliably chooses to write code - it is the
    move that always looks productive - and the code agent then draws a rectangle for an object
    nothing planned. Only the art stage can add the object to the list."""
    import game_studio.graph as gm

    monkeypatch.setattr(gm, "_structured", lambda schema, *a, **kw: gm.SupervisorDecision(
        action="code", reason="코드로 고치겠습니다", instructions="적을 그리세요"))
    state = state_for(tmp_path, stage="qa",
                      qa={"status": "repair", "findings": ["적 스프라이트가 생성되지 않았습니다"]})
    assert gm.supervisor_node(state)["next_step"] == "art"

    # A sprite that exists but is never drawn is a coding problem, and must still go to code.
    drawn = state_for(tmp_path, stage="qa", qa={"status": "repair", "findings": [
        "Generated sprites are never drawn: enemy.png. Load each with new Image()"]})
    assert gm.supervisor_node(drawn)["next_step"] == "code"

    # And once art has already been re-planned it is off the menu, so the override cannot loop.
    spent = state_for(tmp_path, stage="qa", art_revised=True,
                      qa={"status": "repair", "findings": ["적 스프라이트가 생성되지 않았습니다"]})
    assert gm.supervisor_node(spent)["next_step"] != "art"


def test_a_re_planned_asset_list_is_a_mandate_not_a_menu(tmp_path, monkeypatch):
    """On the first pass the plan is a menu and the agent owns the budget - that stays true. A
    revision exists because verification said art was missing, so what it names is required, and
    the art stage produces it there and then rather than asking the code agent to."""
    import game_studio.graph as gm
    from game_studio.models import ArtDirection

    monkeypatch.setattr(gm, "create_art", lambda *a, **kw: ArtDirection(
        palette={}, image_prompt="x", canvas_effects=[],
        asset_plan=["player: 우주선", "enemy: 드론", "boss: 기갑"]))
    generated = []
    monkeypatch.setattr(gm, "_generate_required_art",
                        lambda state, workspace, required, plan: generated.extend(required))
    (tmp_path / "assets").mkdir()
    (tmp_path / "assets" / "player.png").write_bytes(b"x")

    first = gm.art_node(state_for(tmp_path, generate_images=True))
    assert "required_assets" not in first, "a first pass must stay a menu"
    assert generated == [], "and must not generate from a plan made before any code exists"

    revised = gm.art_node(state_for(tmp_path, generate_images=True, art_revision_needed=True,
                                    qa={"findings": ["적 스프라이트가 생성되지 않았습니다"]}))
    assert revised["required_assets"] == ["enemy", "boss"]
    assert revised["art_revised"] is True
    assert generated == ["enemy", "boss"], "the revision generates what it made mandatory"


def test_a_canvas_only_run_is_never_given_a_mandate_it_cannot_satisfy(tmp_path, monkeypatch):
    """With raster generation off, nothing in the pipeline can produce an image. Requiring one
    would block every remaining cycle on a finding that has no fix."""
    import game_studio.graph as gm
    from game_studio.models import ArtDirection

    monkeypatch.setattr(gm, "create_art", lambda *a, **kw: ArtDirection(
        palette={}, image_prompt="x", canvas_effects=[], asset_plan=["enemy: 드론"]))
    monkeypatch.setattr(gm, "_generate_required_art",
                        lambda *a: pytest.fail("nothing may be generated with images disabled"))

    revised = gm.art_node(state_for(tmp_path, generate_images=False, art_revision_needed=True,
                                    qa={"findings": ["적 스프라이트가 생성되지 않았습니다"]}))
    assert revised["required_assets"] == []
    assert revised["art_revised"] is True, "the revision still happened; only the mandate is absent"


def test_the_code_agent_is_told_which_generations_are_mandatory(tmp_path):
    """The requirement has to reach the agent as an instruction, not only as a later rejection."""
    import game_studio.graph as gm

    plain = gm._code_system_prompt(state_for(tmp_path))
    assert "MANDATORY ART" not in plain, "a first pass carries no mandate"

    for prompt in (gm._code_system_prompt(state_for(tmp_path, required_assets=["enemy", "boss"])),
                   gm._godot_system_prompt(state_for(tmp_path, required_assets=["enemy", "boss"]))):
        assert "MANDATORY ART" in prompt
        assert '"enemy"' in prompt and '"boss"' in prompt
        assert "generate_comfyui_image" in prompt


def test_the_agents_own_verification_refuses_to_pass_while_art_is_missing(tmp_path):
    """The first of two gates. The agent asks its own tool whether it is done, so this is where a
    skipped generation gets caught while there is still budget to fix it."""
    import json

    from game_studio.agent_tools import run_static_qa, write_game_file

    state = state_for(tmp_path, required_assets=["enemy"])
    write_game_file.invoke({"html": PLAYABLE, "state": state})
    report = json.loads(run_static_qa.invoke({"state": state}))
    assert report["status"] == "repair"
    assert "enemy" in report["findings"][0] and "generate_comfyui_image" in report["findings"][0]

    (tmp_path / "assets").mkdir(exist_ok=True)
    (tmp_path / "assets" / "enemy.png").write_bytes(b"x")
    cleared = json.loads(run_static_qa.invoke({"state": state}))
    assert not any("생성되지 않았습니다" in f for f in cleared["findings"])


def test_qa_is_the_backstop_when_the_agent_simply_runs_out_of_calls(tmp_path, monkeypatch):
    """The second gate. The agent's own tool can be left uncalled - it can exhaust its budget and
    stop - and a build that quietly shipped without the art QA had already asked for is the exact
    loop this mechanism exists to close."""
    import game_studio.graph as gm
    from game_studio.models import DesignReview

    monkeypatch.setattr(gm, "_structured", lambda *a, **kw: DesignReview(checks=[]))
    state = state_for(tmp_path, stage="code", required_assets=["enemy"], game_html=PLAYABLE)
    blocked = gm.qa_node(state)
    assert blocked["qa"]["status"] == "repair"
    assert "enemy" in blocked["qa"]["findings"][0]
    assert "enemy" in blocked["qa"]["repair_instructions"], "the repair has to name the file"

    (tmp_path / "assets").mkdir(exist_ok=True)
    (tmp_path / "assets" / "enemy.png").write_bytes(b"x")
    passed = gm.qa_node(state)
    assert not any("생성되지 않았습니다" in f for f in passed["qa"]["findings"])


def test_a_run_with_no_required_art_is_completely_unaffected(tmp_path, monkeypatch):
    """Every existing run has no required_assets, and must not pay for, or trip over, any of this."""
    import json

    import game_studio.graph as gm
    from game_studio.agent_tools import run_static_qa, write_game_file
    from game_studio.models import DesignReview

    state = state_for(tmp_path, stage="code", game_html=PLAYABLE)
    write_game_file.invoke({"html": PLAYABLE, "state": state})
    assert json.loads(run_static_qa.invoke({"state": state}))["status"] == "pass"

    monkeypatch.setattr(gm, "_structured", lambda *a, **kw: DesignReview(checks=[]))
    assert not any("생성되지" in f for f in gm.qa_node(state)["qa"]["findings"])


@pytest.mark.parametrize("engine", ["html5", "godot"])
def test_both_engines_enforce_the_same_list(tmp_path, engine):
    """The mandate is about art, which both engines generate the same way, so it cannot be a
    property of one build path."""
    import json

    from game_studio.godot_tools import run_godot_qa

    if engine == "godot":
        (tmp_path / "project.godot").write_text("config_version=5\n", encoding="utf-8")
        report = json.loads(run_godot_qa.invoke(
            {"state": state_for(tmp_path, engine="godot", required_assets=["boss"])}))
        # Either the art gate fires, or the engine is absent and says so - never a silent pass.
        assert report["status"] in {"repair", "skipped"}
        if report["status"] == "repair":
            assert "boss" in report["findings"][0]
    else:
        from game_studio.agent_tools import run_static_qa, write_game_file

        state = state_for(tmp_path, required_assets=["boss"])
        write_game_file.invoke({"html": PLAYABLE, "state": state})
        assert json.loads(run_static_qa.invoke({"state": state}))["status"] == "repair"


def test_the_revision_generates_in_the_orientation_the_game_already_uses(tmp_path):
    """A revision happens after a game exists, so the camera is settled: a top-down build's sprites
    face up and a side-on build's face right. The first pass genuinely cannot know that; this one
    reads it off what it already produced, and a mismatched facing breaks the rotation contract."""
    import json

    import game_studio.graph as gm

    assets = tmp_path / "assets"
    assets.mkdir()
    assert gm._established_facing(tmp_path) == "right", "no history falls back to the default"

    (assets / "sprites.json").write_text(json.dumps({
        "player.png": {"kind": "sprite", "facing": "up"},
        "enemy.png": {"kind": "sprite", "facing": "up"},
        "backdrop.png": {"kind": "backdrop", "facing": "none"},
    }), encoding="utf-8")
    assert gm._established_facing(tmp_path) == "up", "a top-down game keeps generating top-down"

    (assets / "sprites.json").write_text("{ broken", encoding="utf-8")
    assert gm._established_facing(tmp_path) == "right", "a corrupt manifest cannot fail the node"


def test_generating_the_required_art_costs_no_model_call(tmp_path, monkeypatch):
    """This is the whole point of moving it here. The objects are already decided, so there is
    nothing for a model to work out - and a measured run spent four of the code agent's calls
    making sprites and then had none left to wire them in."""
    import game_studio.agent_tools as tools
    import game_studio.graph as gm

    calls = []
    monkeypatch.setattr(tools, "_generate_comfyui_image",
                        lambda **kw: calls.append(kw) or f"Generated {kw['asset_name']}")
    monkeypatch.setattr(gm, "_model", lambda *a, **kw: pytest.fail("no model may be called"))
    monkeypatch.setattr(gm, "_structured", lambda *a, **kw: pytest.fail("no model may be called"))

    produced = gm._generate_required_art(
        state_for(tmp_path, generate_images=True), tmp_path, ["enemy", "boss"],
        ["player: 우주선", "enemy: 붉은 드론", "boss: 거대 기갑"])

    assert produced == ["enemy", "boss"]
    assert [c["asset_name"] for c in calls] == ["enemy", "boss"]
    # The plan's own description is the prompt - that is what the entry was written for.
    assert calls[0]["prompt"] == "enemy: 붉은 드론"
    assert all(c["kind"] == "sprite" and c["facing"] == "right" for c in calls)
    # Deterministic per object: a re-run reproduces the same art, and two sprites in one revision
    # do not come back as near-identical images.
    assert calls[0]["seed"] != calls[1]["seed"]
    # Derived from the object's name rather than a counter, so re-running the same revision
    # reproduces the same art instead of a different game every time.
    again = []
    monkeypatch.setattr(tools, "_generate_comfyui_image",
                        lambda **kw: again.append(kw["seed"]) or "ok")
    gm._generate_required_art(state_for(tmp_path, generate_images=True), tmp_path, ["enemy"],
                              ["enemy: 붉은 드론"])
    assert again == [calls[0]["seed"]]


def test_an_animation_frame_inherits_the_character_it_animates(tmp_path):
    """Written from scratch a walk cycle drifts: one run's walk1/walk2/jump came back as "cartoon
    platformer game sprite style, clean outline" while the player they animate was "retro 8-bit
    pixel art style, thick dark outline" - the same cat changing art style as it walked."""
    import json as _json

    from game_studio.agent_tools import _variant_prompt

    assets = tmp_path / "assets"
    assets.mkdir()
    (assets / "sprites.json").write_text(_json.dumps({
        "player.png": {"kind": "sprite", "facing": "right",
                       "prompt": "주황 줄무늬 고양이, 둥근 얼굴, 굵은 꼬리"},
    }, ensure_ascii=False), encoding="utf-8")

    frame = _variant_prompt(assets, "player", "왼발 앞으로 내딛는 달리기 자세")
    assert frame.startswith("주황 줄무늬 고양이"), "the character description carries over"
    assert "왼발 앞으로" in frame, "and this frame's pose is what changes"

    # A name nobody generated costs a less consistent frame, not a failed generation.
    assert _variant_prompt(assets, "없는것", "점프 자세") == "점프 자세"
    assert _variant_prompt(tmp_path / "빈폴더", "player", "점프 자세") == "점프 자세"


def test_the_agent_is_shown_what_it_already_wrote_in_this_game(tmp_path):
    """The house style clause fixes the art style; this fixes what it cannot cover - how much
    anatomy gets described, whether eyes are "big sparkling round" or "two dots". Measured, a run's
    sprites drifted in exactly those: the enemy got two clauses where the player got six, and they
    read as two different artists even where the style token agreed."""
    import json as _json

    from game_studio.agent_tools import list_game_assets

    assets = tmp_path / "assets"
    assets.mkdir()
    for name in ("player.png", "enemy.png"):
        (assets / name).write_bytes(b"x")
    (assets / "sprites.json").write_text(_json.dumps({
        "player.png": {"kind": "sprite", "facing": "right", "transparent": True,
                       "width": 300, "height": 300, "prompt": "주황 줄무늬 고양이, 둥근 얼굴"},
        "enemy.png": {"kind": "sprite", "facing": "right", "transparent": True,
                      "width": 280, "height": 280, "prompt": "회색 고양이, 찡그린 눈"},
    }, ensure_ascii=False), encoding="utf-8")

    listing = list_game_assets.invoke({"state": state_for(tmp_path)})
    assert "이 게임에서 이미 쓴 설명입니다" in listing
    assert "주황 줄무늬 고양이" in listing and "회색 고양이" in listing
    # The drawing contracts still come first - they are what the code needs to render at all.
    assert listing.index("assets/player.png") < listing.index("이 게임에서")

    # A run with no prompts recorded gets the listing it always got, with no empty block.
    (assets / "sprites.json").write_text("{}", encoding="utf-8")
    assert "이 게임에서" not in list_game_assets.invoke({"state": state_for(tmp_path)})
