"""The Godot engine option: what changes, and what deliberately does not.

A run picks its engine once, at the start. Every planning stage - the concept, the implementation
contract, the human approval, the art direction - is shared, and only the three stages that touch
the artifact diverge: building it, verifying it, and packaging it. These pin that boundary, because
the easiest way to break this feature is to let the Godot path leak into the HTML one.

The engine's own behaviour is pinned against captured output rather than by invoking Godot, so the
suite stays fast and runs on a machine that has no engine installed. The one test that does shell
out is skipped when there is nothing to shell out to.
"""

import json

import pytest

from game_studio.godot import GodotCheck, parse_errors
from game_studio.godot_tools import WRITABLE_SUFFIXES, _resolve


def state_for(tmp_path, engine="godot") -> dict:
    return {
        "engine": engine,
        "concept": {"title": "테스트", "elevator_pitch": "p", "player_goal": "g",
                    "controls": ["a", "b"], "core_loop": ["1", "2", "3"],
                    "difficulty_curve": "d", "visual_direction": "v"},
        "output_dir": str(tmp_path), "workspace_dir": str(tmp_path),
    }


# Real stderr from Godot 4.7.2, captured from a project with each defect introduced on purpose.
PARSE_ERROR = """Godot Engine v4.7.2.stable.official.ed1daf0bf
SCRIPT ERROR: Parse Error: Expected expression for variable initial value after "=".
   at: GDScript::reload (res://main.gd:3)
ERROR: Failed to load script "res://main.gd" with error "Parse error".
   at: load (modules/gdscript/gdscript_resource_format.cpp:46)
"""
RUNTIME_ERROR = """Godot Engine v4.7.2.stable.official.ed1daf0bf
SCRIPT ERROR: Cannot call method 'get_child' on a null value.
   at: _ready (res://main.gd:4)
   GDScript backtrace (most recent call first):
       [0] _ready (res://main.gd:4)
"""
CLEAN_RUN = """Godot Engine v4.7.2.stable.official.ed1daf0bf
[  83% ] first_scan_filesystem | 파일 스캔 시작 중...
[ DONE ] first_scan_filesystem
"""


def test_engine_errors_are_reported_with_the_file_and_line_that_caused_them():
    """A finding without an address is not actionable. Godot prints the message and the location on
    separate lines, so joining them is what turns engine output into something to fix."""
    findings = parse_errors(PARSE_ERROR)
    assert findings, "a parse error has to be reported"
    assert "res://main.gd:3" in findings[0] and "Parse Error" in findings[0]

    runtime = parse_errors(RUNTIME_ERROR)
    assert len(runtime) == 1, "one error, not one per backtrace line"
    assert "null value" in runtime[0] and "res://main.gd:4" in runtime[0]

    assert parse_errors(CLEAN_RUN) == [], "progress chatter is not a defect"


def test_a_godot_path_cannot_escape_the_run_workspace(tmp_path):
    """The path arrives as free text from a model, so this is a boundary, not a formality."""
    state = state_for(tmp_path)
    assert _resolve(state, "res://main.gd") == (tmp_path / "main.gd").resolve()
    assert _resolve(state, "scenes/level.tscn") == (tmp_path / "scenes" / "level.tscn").resolve()
    for escape in ("../outside.gd", "res://../../x.gd", "..\\..\\evil.gd", "C:/Windows/evil.gd", ""):
        with pytest.raises(ValueError):
            _resolve(state, escape)
    # A leading slash is project-relative, the way res:// is - contained, not an escape. Refusing it
    # would be wrong; letting it reach the real filesystem root would be the bug.
    assert _resolve(state, "/etc/passwd.gd") == (tmp_path / "etc" / "passwd.gd").resolve()


def test_only_text_a_model_can_actually_author_is_writable(tmp_path):
    """Godot's binary formats and its .import sidecars are produced by the engine on import. A
    model writing one by hand produces a corrupt project that fails in a way nothing can explain."""
    assert {".gd", ".tscn", ".godot"} <= WRITABLE_SUFFIXES
    assert not {".png", ".res", ".scn", ".import", ".exe"} & WRITABLE_SUFFIXES
    for rejected in ("sprite.png", "player.res", "main.tscn.import"):
        with pytest.raises(ValueError):
            _resolve(state_for(tmp_path), rejected)


def test_the_code_agent_gets_the_engine_it_was_asked_for(tmp_path, monkeypatch):
    """The engine decides the toolset and the contract, and nothing else about the build node."""
    import game_studio.graph as gm

    captured = {}

    def build(model_id, tools, prompt, retry_on):
        captured["tools"] = [tool.name for tool in tools]
        captured["prompt"] = prompt
        raise RuntimeError("stop after wiring")

    monkeypatch.setattr(gm, "build_code_agent", build)
    state = {**state_for(tmp_path), "brief": "b", "art": {"palette": {}, "image_prompt": "x",
             "asset_plan": [], "canvas_effects": []}, "implementation_plan": {}, "use_llm": True}
    with pytest.raises(RuntimeError):
        gm.code_node(state)
    assert "write_godot_file" in captured["tools"] and "run_godot_qa" in captured["tools"]
    assert "write_game_file" not in captured["tools"], "the HTML tools must not leak in"
    assert "Godot 4" in captured["prompt"]

    captured.clear()
    with pytest.raises(RuntimeError):
        gm.code_node({**state, "engine": "html5"})
    assert "write_game_file" in captured["tools"]
    assert "write_godot_file" not in captured["tools"], "and the Godot tools must not leak out"


def test_an_unknown_engine_builds_the_html_game_rather_than_failing(tmp_path):
    """The field can arrive from an older checkpoint or a hand-made payload. An unrecognised value
    is not worth losing a run over."""
    import game_studio.graph as gm

    assert gm._engine({}) == gm.HTML5
    assert gm._engine({"engine": "unity"}) == gm.HTML5
    assert gm._engine({"engine": "GODOT"}) == gm.GODOT, "the choice is case-insensitive"


def test_a_godot_build_is_repaired_by_the_code_agent_not_by_a_text_rewrite():
    """repair is one model call answering with the whole game as a blob of text. That is a complete
    artifact for HTML and a meaningless one for a project of scenes and scripts."""
    import game_studio.graph as gm

    fresh = {"repair_attempts": 0, "rethink_cycles": 0}
    assert "repair" in gm._affordable_actions({**fresh, "engine": "html5"})
    assert "repair" not in gm._affordable_actions({**fresh, "engine": "godot"})
    # The expensive move is still available, or a failed Godot build would have no way back at all.
    assert "code" in gm._affordable_actions({**fresh, "engine": "godot"})


def test_a_missing_engine_is_an_honest_skip_not_a_false_pass(tmp_path, monkeypatch):
    """A machine without Godot must not silently report that the game runs."""
    import game_studio.graph as gm

    monkeypatch.setattr(gm, "godot_available", lambda: False)
    report = gm._godot_report(tmp_path)
    assert report.status == "pass", "a missing engine cannot block a build"
    assert any("찾을 수 없어" in finding for finding in report.findings), "but it has to be said"


def test_engine_verification_runs_scripts_first_then_the_game(tmp_path, monkeypatch):
    """A parse error hides every runtime error after it, so the game is only run once every script
    compiles - and the run's findings are what the build is judged on."""
    import game_studio.graph as gm

    monkeypatch.setattr(gm, "godot_available", lambda: True)
    monkeypatch.setattr(gm, "check_scripts",
                        lambda _p: GodotCheck(ok=False, findings=["broken.gd:3"], output=""))
    monkeypatch.setattr(gm, "run_project", lambda *a, **kw: pytest.fail("must not run a broken build"))
    failed = gm._godot_report(tmp_path)
    assert failed.status == "repair" and failed.findings == ["broken.gd:3"]

    monkeypatch.setattr(gm, "check_scripts", lambda _p: GodotCheck(ok=True, findings=[], output=""))
    monkeypatch.setattr(gm, "run_project",
                        lambda *a, **kw: GodotCheck(ok=False, findings=["null in _ready"], output=""))
    ran = gm._godot_report(tmp_path)
    assert ran.status == "repair" and ran.findings == ["null in _ready"]


def test_the_web_build_is_a_bonus_and_its_absence_is_explained(tmp_path, monkeypatch):
    """Godot ships export templates only inside a ~1GB all-platform archive, so a machine with the
    editor very often cannot export. The project is complete either way and must not be reported as
    a failure - but the dashboard has nothing to embed, so it has to say why."""
    import game_studio.graph as gm

    monkeypatch.setattr(gm, "godot_version", lambda: "4.7.2.stable")
    monkeypatch.setattr(gm, "export_web", lambda *a: (False, "익스포트 템플릿이 설치되어 있지 않습니다"))
    state = {**state_for(tmp_path), "art": {}, "implementation_plan": {},
             "qa": {"status": "pass", "findings": []}, "design_review": {"checks": []}}
    produced = gm.package_node(state)

    assert produced["godot_project_path"].endswith("project.godot")
    assert "game_path" not in produced, "there is no web build to play"
    manifest = json.loads((tmp_path / "production-manifest.json").read_text(encoding="utf-8"))
    assert manifest["engine"] == "godot"
    assert manifest["web_export"]["ok"] is False
    assert "템플릿" in manifest["web_export"]["detail"], "the reason has to travel with the game"

    # With templates present the dashboard gets something to embed.
    monkeypatch.setattr(gm, "export_web", lambda *a: (True, "ok"))
    assert gm.package_node(state)["game_path"].endswith("index.html")


def test_the_html_path_is_untouched_by_any_of_this(tmp_path):
    """The whole point of the option is that the existing pipeline keeps working exactly as it did."""
    import game_studio.graph as gm

    state = {**state_for(tmp_path, engine="html5"), "art": {}, "implementation_plan": {},
             "qa": {"status": "pass", "findings": []}, "design_review": {"checks": []},
             "game_html": "<html><canvas></canvas></html>"}
    produced = gm.package_node(state)
    assert produced == {"game_path": str(tmp_path / "index.html")}
    assert (tmp_path / "index.html").read_text(encoding="utf-8") == state["game_html"]
    assert "godot_project_path" not in produced
    manifest = json.loads((tmp_path / "production-manifest.json").read_text(encoding="utf-8"))
    assert manifest["engine"] == "html5" and "web_export" not in manifest


def test_a_finished_godot_folder_can_be_played_without_the_dashboard(tmp_path, monkeypatch):
    """run.bat is the deliverable's own front door: the folder has to play on a machine with no
    dashboard, no repository and no Python, so the launcher goes in the folder itself."""
    import game_studio.godot as gd

    monkeypatch.setattr(gd, "godot_executable", lambda: r"C:\dev\godot\Godot_v4.7.2-stable_win64_console.exe")
    script = gd.write_launch_script(tmp_path)
    assert script is not None and script.name == "run.bat"
    body = script.read_text(encoding="utf-8")

    # --path runs the game. Godot's own help says -e/--editor starts "the editor instead of running
    # the scene", so its absence is what makes this a play button rather than an edit button.
    assert "--path" in body
    assert " -e " not in body and "--editor" not in body
    # Its own directory, so the folder can be moved or copied anywhere.
    assert "%~dp0" in body
    assert str(tmp_path) not in body, "the project path must not be baked in"
    # The windowed build, not the console one used for verification: a player wants a game, not a
    # terminal full of engine output.
    assert "_console.exe" not in body
    # Written UTF-8, so the console has to be told before any Korean message is printed.
    assert body.index("chcp 65001") < body.index("Godot 실행 파일을 찾을 수 없습니다")
    assert "GODOT_EXECUTABLE" in body, "the engine location has to be overridable"


def test_no_engine_means_no_launcher_rather_than_a_broken_one(tmp_path, monkeypatch):
    import game_studio.godot as gd

    monkeypatch.setattr(gd, "godot_executable", lambda: None)
    assert gd.write_launch_script(tmp_path) is None
    assert not (tmp_path / "run.bat").exists()


def test_the_launcher_ships_even_when_the_audit_never_passed(tmp_path, monkeypatch):
    """A reviewer can only judge a game they can start. A Godot run that spent its budget still
    produced a project, so the project and its launcher are the deliverable - labelled, not hidden."""
    import game_studio.graph as gm

    (tmp_path / "project.godot").write_text("config_version=5\n", encoding="utf-8")
    monkeypatch.setattr(gm, "godot_version", lambda: "4.7.2.stable")
    monkeypatch.setattr(gm, "write_launch_script", lambda target: target / "run.bat")
    produced = gm.abandoned_node({**state_for(tmp_path), "art": {},
                                  "qa": {"status": "repair", "findings": ["미충족: 가속"]}})

    assert produced["godot_project_path"].endswith("project.godot")
    assert produced["launch_script_path"].endswith("run.bat")
    assert produced["qa_report_path"].endswith("qa-report.json")
    manifest = json.loads((tmp_path / "production-manifest.json").read_text(encoding="utf-8"))
    assert manifest["generation_mode"] == "model_generated_qa_failed"
    assert manifest["qa_outstanding"] == ["미충족: 가속"], "the open items travel with the game"
    # And an HTML run that produced nothing still must not claim an artifact.
    empty = gm.abandoned_node({**state_for(tmp_path, engine="html5"), "art": {}, "qa": {"findings": []}})
    assert "game_path" not in empty and "godot_project_path" not in empty


def test_the_run_button_can_only_ever_start_the_launcher_this_pipeline_wrote(tmp_path, monkeypatch):
    """This is the one endpoint in the dashboard that starts a local process, so nothing from the
    request may reach the command line: not a path, not an argument, not a run id that is not a
    plain token."""
    from fastapi.testclient import TestClient

    from game_studio import server

    monkeypatch.setattr(server, "GAME_OUTPUT_ROOT", tmp_path)
    launched = []
    monkeypatch.setattr(server, "_spawn_detached", lambda script, cwd: launched.append((script, cwd)))

    run_dir = tmp_path / "abc123"
    run_dir.mkdir()
    with TestClient(server.app) as client:
        assert client.post("/api/runs/abc123/launch").status_code == 404, "no launcher, no launch"
        (run_dir / "run.bat").write_text("@echo off\n", encoding="utf-8")
        assert client.post("/api/runs/abc123/launch").status_code == 202
        for hostile in ("../../evil", "..%2F..%2Fevil", "abc123/../../evil", "a b"):
            assert client.post(f"/api/runs/{hostile}/launch").status_code == 404

    assert len(launched) == 1, "only the one real launcher ever ran"
    script, cwd = launched[0]
    assert script == (run_dir / "run.bat").resolve() and cwd == run_dir.resolve()
