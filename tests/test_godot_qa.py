"""Godot's structural QA: the checks the engine itself cannot make.

Headless Godot is the strongest verification in this pipeline - it compiles every script and
actually plays the game, reporting a null dereference with its file and line. But it can only
report what throws, and a project that throws nothing is not the same as a game. A scene whose
script is `func _ready(): pass` imports cleanly, runs its five seconds, exits zero, and used to be
reported as a pass: no input, no scoring, no win or loss, nothing to play.

The web path has static_qa for exactly this class of problem. This is Godot's.
"""

import pytest

from game_studio.godot import static_project_qa

CONFIG = ('config_version=5\n\n[application]\nconfig/name="T"\n'
          'run/main_scene="res://main.tscn"\n')
SCENE = ('[gd_scene load_steps=2 format=3]\n\n'
         '[ext_resource type="Script" path="res://main.gd" id="1"]\n\n'
         '[node name="Main" type="Node2D"]\nscript = ExtResource("1")\n')
PLAYABLE = '''extends Node2D
var score := 0
func _process(delta: float) -> void:
\tif Input.is_key_pressed(KEY_A) or Input.is_key_pressed(KEY_LEFT):
\t\tscore += 1
func restart() -> void:
\tget_tree().reload_current_scene()
'''


def project(tmp_path, script=PLAYABLE, config=CONFIG, scene=SCENE, **extra):
    (tmp_path / "project.godot").write_text(config, encoding="utf-8")
    (tmp_path / "main.tscn").write_text(scene, encoding="utf-8")
    (tmp_path / "main.gd").write_text(script, encoding="utf-8")
    for name, body in extra.items():
        (tmp_path / name.replace("__", ".")).write_text(body, encoding="utf-8")
    return tmp_path


def test_a_complete_game_passes(tmp_path):
    check = static_project_qa(project(tmp_path))
    assert check.ok, check.findings


def test_a_project_that_runs_and_does_nothing_is_not_a_game(tmp_path):
    """The gap this whole module exists for. It compiles, it runs, it exits zero, and there is
    nothing to play - which is exactly what the engine has no way to report."""
    check = static_project_qa(project(tmp_path, script="extends Node2D\nfunc _ready() -> void:\n\tpass\n"))
    assert not check.ok
    assert any("_process" in finding for finding in check.findings), \
        "a game that never updates has to be caught"
    assert any("입력" in finding for finding in check.findings), \
        "and one that never reads input"


def test_a_frame_loop_without_input_is_still_unplayable(tmp_path):
    check = static_project_qa(project(
        tmp_path, script="extends Node2D\nvar s := 0\nfunc _process(d: float) -> void:\n\ts += 1\n"))
    assert not check.ok
    assert any("입력" in finding for finding in check.findings)
    assert not any("_process" in finding for finding in check.findings), \
        "the loop it does have must not be reported as missing"


NETWORKING = ('extends Node2D\nfunc _process(d: float) -> void:\n\t{}\n'
              '\tif Input.is_action_pressed("ui_left"): pass\n')


@pytest.mark.parametrize("script,expected", [
    (NETWORKING.format("var h := HTTPRequest.new()"), "HTTPRequest"),
    (NETWORKING.format('var u := "https://example.com"'), "https://"),
])
def test_a_generated_game_has_to_be_standalone(tmp_path, script, expected):
    """The same rule the web path enforces: no external network dependency, whatever the engine."""
    check = static_project_qa(project(tmp_path, script=script))
    assert not check.ok
    assert any(expected in finding for finding in check.findings)


def test_a_reference_to_a_file_nobody_wrote_is_caught_before_it_is_reached(tmp_path):
    """This is the one the headless run genuinely cannot do. Five seconds of play only touches the
    code path it happens to reach; a scene loaded from a branch that needs input is never opened,
    and the broken path surfaces on a player's machine instead."""
    check = static_project_qa(project(
        tmp_path,
        script='extends Node2D\nvar t = "res://scenes/boss.tscn"\n'
               'func _process(d: float) -> void:\n\tif Input.is_key_pressed(KEY_A): pass\n'))
    assert not check.ok
    assert any("scenes/boss.tscn" in finding for finding in check.findings)

    # A path that does exist is not reported.
    (tmp_path / "scenes").mkdir()
    (tmp_path / "scenes" / "boss.tscn").write_text("[gd_scene format=3]\n", encoding="utf-8")
    assert static_project_qa(tmp_path).ok


def test_a_missing_main_scene_is_caught_without_starting_the_engine(tmp_path):
    no_scene = static_project_qa(project(
        tmp_path, config='config_version=5\n\n[application]\nconfig/name="T"\n'))
    assert not no_scene.ok and any("main_scene" in f for f in no_scene.findings)

    # Named but absent is a different failure from never named, and both have to be caught.
    project(tmp_path)
    (tmp_path / "main.tscn").unlink()
    gone = static_project_qa(tmp_path)
    assert not gone.ok and any("메인 장면" in f for f in gone.findings)

    empty = static_project_qa(tmp_path / "nothing-here")
    assert not empty.ok and "project.godot" in empty.findings[0]


def test_generated_art_the_project_never_draws_is_reported(tmp_path):
    """Reported, and non-blocking. The game runs either way; what was lost is the cost of the art,
    and failing the release over it spent the rethink budget and shipped with it open anyway."""
    check = static_project_qa(project(tmp_path), sprites=["enemy.png"])
    assert check.ok, "waste is not breakage"
    assert any("enemy.png" in finding for finding in check.findings)

    drawn = static_project_qa(project(
        tmp_path,
        script=PLAYABLE + '\nvar tex = preload("res://assets/enemy.png")\n'), sprites=["enemy.png"])
    assert not any("참조하지 않습니다" in finding for finding in drawn.findings)


def test_control_and_polish_gaps_are_advisory_not_blocking(tmp_path):
    """static_qa's hard-won lesson, applied here: a keyword check that failed a working game used
    to cost a whole repair cycle. And WASD is genuinely different on this engine - Godot's
    physical_keycode is layout independent, so the Korean-IME defect the web rule exists for
    cannot happen, which makes arrows-only a design choice rather than a broken control."""
    arrows_only = '''extends Node2D
func _process(delta: float) -> void:
\tif Input.is_action_pressed("ui_left"):
\t\tpass
'''
    check = static_project_qa(project(tmp_path, script=arrows_only))
    assert check.ok, "no score, no restart and no WASD must not block a release"
    # These three are keyword guesses, as often wrong as right, so they stay out of a clean report.
    # Unused art is not a guess and does appear - see the test below.
    assert check.findings == []


def test_the_structural_check_runs_before_the_engine_is_even_needed(tmp_path, monkeypatch):
    """It is free, and its findings are the ones an engine run cannot produce - so a machine with
    no Godot at all still gets them."""
    import game_studio.graph as gm

    monkeypatch.setattr(gm, "godot_available", lambda: False)
    monkeypatch.setattr(gm, "check_scripts",
                        lambda *a: pytest.fail("the engine must not be reached"))
    report = gm._godot_report(project(
        tmp_path, script="extends Node2D\nfunc _ready() -> void:\n\tpass\n"))
    assert report.status == "repair", "a hollow project fails even with no engine installed"
    assert any("입력" in finding for finding in report.findings)


def test_the_agents_own_tool_reports_the_same_structural_verdict(tmp_path):
    """The agent has to catch this inside its loop, while it still has calls left to fix it."""
    import json

    from game_studio.godot_tools import run_godot_qa

    project(tmp_path, script="extends Node2D\nfunc _ready() -> void:\n\tpass\n")
    state = {"engine": "godot", "output_dir": str(tmp_path), "workspace_dir": str(tmp_path),
             "concept": {"title": "t", "elevator_pitch": "p", "player_goal": "g",
                         "controls": ["a", "b"], "core_loop": ["1", "2", "3"],
                         "difficulty_curve": "d", "visual_direction": "v"}}
    report = json.loads(run_godot_qa.invoke({"state": state}))
    assert report["status"] == "repair"
    assert report["stage"] == "project-structure"
    assert any("입력" in finding for finding in report["findings"])


def test_a_runtime_assembled_asset_path_is_not_called_a_missing_file(tmp_path):
    """The failure that ended a real run. A falling-block puzzle names its pieces in a loop -
    load("res://assets/block-" + kind + ".png") - so the literal "block-i.png" appears nowhere and
    the source carries a fragment, "res://assets/block-", that names no file and never will.

    Reading that as unused art and a missing resource failed a build whose art was perfectly fine,
    spent the entire rethink budget failing to "fix" it, and shipped with the finding still open.
    """
    assets = tmp_path / "assets"
    assets.mkdir()
    sprites = ["block-i.png", "block-l.png", "block-o.png"]
    for sprite in sprites:
        (assets / sprite).write_bytes(b"x")

    for loader in ('load("res://assets/block-" + kind + ".png")',
                   'load("res://assets/block-%s.png" % kind)',
                   'load("res://assets/block-{0}.png".format(kind))'):
        check = static_project_qa(project(
            tmp_path, script=PLAYABLE + f"\nfunc tex(kind: String):\n\treturn {loader}\n"), sprites)
        blocking = [f for f in check.findings if not f.startswith("참고")]
        assert check.ok, f"{loader} must not fail the build: {blocking}"
        assert not any("존재하지 않는 리소스" in f for f in check.findings), \
            "a path fragment is not a missing file"
        assert not any("참조하지 않습니다" in f for f in check.findings), \
            "art loaded by an assembled path is used art"


def test_a_missing_resource_says_which_file_to_create(tmp_path):
    """Every one of these is a single file away from running, and the two shapes are both
    recognisable from the path. A run shipped referencing four resources it never created - three
    .tscn scenes whose .gd scripts were sitting right there, and a sprite under a mangled name -
    and spent its whole budget with all four still open, because a bare missing path does not say
    whether the fix is a scene, a sprite, or a typo.
    """
    script = PLAYABLE + (
        '\nfunc spawn():\n'
        '\treturn load("res://coin_effect.tscn")\n'
        'func art():\n'
        '\treturn load("res://assets/enemy-goomba.png")\n'
        'func gone():\n'
        '\treturn load("res://data/level.cfg")\n'
    )
    # The script the agent wrote and never packaged into a scene.
    (tmp_path / "coin_effect.gd").write_text("extends Node2D\n", encoding="utf-8")
    findings = static_project_qa(project(tmp_path, script=script)).findings

    scene = next(f for f in findings if "coin_effect.tscn" in f)
    assert "coin_effect.gd는 있는데" in scene, "the sibling script is the whole diagnosis"
    assert "write_godot_file" in scene, "and the tool that fixes it has to be named"

    sprite = next(f for f in findings if "enemy-goomba.png" in f)
    assert 'asset_name="enemy-goomba"' in sprite, "named without the extension it would mangle"
    assert "generate_comfyui_image" in sprite

    # Neither shape applies, so it stays a plain report rather than inventing advice.
    plain = next(f for f in findings if "level.cfg" in f)
    assert plain == "존재하지 않는 리소스를 참조합니다: res://data/level.cfg"

    # A .tscn with no sibling script is not the scene-packaging case either.
    orphan = static_project_qa(project(
        tmp_path, script=PLAYABLE + '\nfunc x():\n\treturn load("res://nowhere.tscn")\n')).findings
    assert any(f == "존재하지 않는 리소스를 참조합니다: res://nowhere.tscn" for f in orphan)


def test_art_that_really_is_unused_is_reported_but_does_not_block(tmp_path):
    """It is waste worth naming, not breakage. The game runs. Blocking on it spent the whole
    rethink budget on "you did not use art you paid for" and then shipped with the finding open
    anyway - the release failed and the waste was not fixed."""
    assets = tmp_path / "assets"
    assets.mkdir()
    (assets / "unused-boss.png").write_bytes(b"x")

    check = static_project_qa(project(tmp_path), ["unused-boss.png"])
    assert check.ok, "unused art must never fail a release"
    assert any("unused-boss.png" in finding for finding in check.findings), \
        "but it has to be reported, on an otherwise clean project too"
    assert all(finding.startswith("참고") for finding in check.findings), "as an advisory"


def test_a_genuinely_broken_resource_path_still_blocks(tmp_path):
    """Loosening the check must not lose the case it was written for."""
    check = static_project_qa(project(
        tmp_path, script=PLAYABLE + '\nvar boss = preload("res://scenes/boss.tscn")\n'))
    assert not check.ok
    assert any("scenes/boss.tscn" in finding for finding in check.findings)


def test_a_web_build_that_produced_nothing_leaves_nothing_behind(tmp_path, monkeypatch):
    """An empty build/ is worse than no build/. It is what the dashboard and the restore path look
    in to decide whether a run has something playable in the browser, so a directory the export
    created and then failed to fill reads as a broken page rather than an honest absence - two
    finished runs were sitting on disk with exactly that."""
    import game_studio.godot as gd

    (tmp_path / "project.godot").write_text("config_version=5\n", encoding="utf-8")
    monkeypatch.setattr(gd, "godot_available", lambda: True)
    monkeypatch.setattr(gd, "godot_version", lambda: "4.7.2.stable")
    monkeypatch.setattr(gd, "_run", lambda *a: (1, "no export_templates found"))

    ok, note = gd.export_web(tmp_path, tmp_path / "build")
    assert not ok and "템플릿" in note
    assert not (tmp_path / "build").exists(), "a build that made nothing must not leave a folder"

    # A successful export keeps everything, obviously.
    def succeed(args, _timeout):
        out = tmp_path / "build"
        out.mkdir(parents=True, exist_ok=True)
        (out / "index.html").write_text("<html></html>", encoding="utf-8")
        return 0, ""

    monkeypatch.setattr(gd, "_run", succeed)
    ok, note = gd.export_web(tmp_path, tmp_path / "build")
    assert ok and (tmp_path / "build" / "index.html").is_file()


def test_packaging_never_turns_a_finished_project_into_a_failed_run(tmp_path, monkeypatch):
    """Packaging runs last, on a project that is already complete on disk. A missing engine, a
    locked file or an export that dies in a way nobody anticipated all mean "no web build" - which
    the dashboard already knows how to show - and none of them is worth converting a delivered game
    into a failed run."""
    import game_studio.graph as gm

    (tmp_path / "project.godot").write_text("config_version=5\n", encoding="utf-8")
    monkeypatch.setattr(gm, "godot_version", lambda: "4.7.2.stable")
    monkeypatch.setattr(gm, "write_launch_script", lambda target: target / "run.bat")
    monkeypatch.setattr(gm, "export_web",
                        lambda *a: (_ for _ in ()).throw(RuntimeError("엔진이 죽었습니다")))

    manifest = {}
    produced = gm._package_godot({"engine": "godot"}, tmp_path, manifest)

    assert produced["godot_project_path"].endswith("project.godot"), "the project still ships"
    assert produced["launch_script_path"].endswith("run.bat"), "and so does its launcher"
    assert "game_path" not in produced
    assert manifest["web_export"]["ok"] is False
    assert "엔진이 죽었습니다" in manifest["web_export"]["detail"], "and the reason travels with it"

    # A launcher that cannot be written is the same kind of non-event.
    monkeypatch.setattr(gm, "write_launch_script",
                        lambda target: (_ for _ in ()).throw(OSError("읽기 전용")))
    monkeypatch.setattr(gm, "export_web", lambda *a: (False, "템플릿 없음"))
    degraded = gm._package_godot({"engine": "godot"}, tmp_path, {})
    assert degraded == {"godot_project_path": str(tmp_path / "project.godot")}


def test_a_type_name_gdscript_does_not_have_is_caught_before_the_engine_runs(tmp_path):
    """The failure this was written for: `dict` on line 377 of a delivered main.gd.

    An unresolvable type is a PARSE error, not a style problem. The script does not load at all, so
    Godot brings the scene up and nothing in it runs - no _ready, no _process, no input, no drawing.
    The window opens black and the game looks like it was never written. That run spent both of its
    rethink cycles on the two resulting error lines and shipped nothing, over one word.

    Checked statically because it costs no model call and no engine start, and because naming the
    replacement is the part the engine's own message leaves out: "Could not find type 'dict' in the
    current scope" never says that the answer is "Dictionary".
    """
    from game_studio.godot import foreign_types

    source = tmp_path / "main.gd"
    found = foreign_types({source: """\
extends Node2D
var slots: dict = {}
var names: list = []
func describe(data: dict) -> str:
	return ''
func tally(rows: Array[str]) -> number:
	return 0.0
func check(node) -> void:
	if node is dict:
		pass
"""})
    wrong = [finding.split("'")[1] for finding in found]
    assert sorted(wrong) == ["dict", "dict", "dict", "list", "number", "str", "str"], wrong
    assert "main.gd:2" in found[0], "the line number is what makes this one edit instead of a hunt"
    assert "Dictionary" in found[0], "saying what is wrong without saying what is right is the "\
        "engine's own message, which was not enough"
    assert "화면이 빈 채로" in found[0], "the symptom has to be named, or nobody connects the two"


def test_the_type_check_does_not_fire_on_correct_gdscript(tmp_path):
    """Every one of these is a blocking finding, so a false one costs a repair cycle on working
    code. The traps are real: `int`, `float` and `bool` ARE GDScript types; a colon ends every
    block header, so a variable named `list` on the next line must not be read as a type; and a
    Label can legitimately say "Score: str"."""
    from game_studio.godot import foreign_types

    source = tmp_path / "main.gd"
    assert foreign_types({source: """\
extends Node2D
var speed := 400.0
var score: int = 0
var alive: bool = true
var caption: String = "Score: str"
var slots: Dictionary = {}
var items: Array[String] = []
func _ready() -> void:
	set_process(true)
func _process(delta: float) -> void:
	if alive:
		list_of_things()
func describe(data: Dictionary, names: Array) -> String:
	# dict is what python calls it
	return str(data.size())
func _on_body(body: Node2D) -> void:
	if body is CharacterBody2D:
		pass
"""}) == []

    # .tscn and .tres carry colons everywhere and no GDScript annotations at all.
    assert foreign_types({tmp_path / "main.tscn": '[node name="X" type="Node2D"]\nscript = null\n'}) == []


def test_the_type_check_is_wired_into_the_static_pass(tmp_path):
    """It has to reach an actual verdict, not merely exist."""
    from game_studio.godot import static_project_qa

    (tmp_path / "project.godot").write_text(
        'config_version=5\n\n[application]\nrun/main_scene="res://main.tscn"\n', encoding="utf-8")
    (tmp_path / "main.tscn").write_text('[gd_scene format=3]\n[node name="Main" type="Node2D"]\n',
                                        encoding="utf-8")
    (tmp_path / "main.gd").write_text("""\
extends Node2D
var slots: dict = {}
func _process(delta: float) -> void:
	if Input.is_action_pressed('ui_right'):
		pass
    """, encoding="utf-8")

    check = static_project_qa(tmp_path)
    assert not check.ok, "a script that cannot parse is not a passing project"
    assert any("Dictionary" in finding for finding in check.findings), check.findings
