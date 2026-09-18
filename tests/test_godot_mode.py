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
import pathlib
import time

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
    # The structural check runs first and needs no engine, so it has to pass for the skip to be
    # the thing under test.
    monkeypatch.setattr(gm, "static_project_qa",
                        lambda *a, **kw: GodotCheck(ok=True, findings=[], output=""))
    report = gm._godot_report(tmp_path)
    assert report.status == "pass", "a missing engine cannot block a build"
    assert any("찾을 수 없어" in finding for finding in report.findings), "but it has to be said"


def test_engine_verification_runs_scripts_first_then_the_game(tmp_path, monkeypatch):
    """A parse error hides every runtime error after it, so the game is only run once every script
    compiles - and the run's findings are what the build is judged on."""
    import game_studio.graph as gm

    monkeypatch.setattr(gm, "godot_available", lambda: True)
    monkeypatch.setattr(gm, "static_project_qa",
                        lambda *a, **kw: GodotCheck(ok=True, findings=[], output=""))
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
    monkeypatch.setattr(gm, "export_web", lambda *a: (False, "익스포트 템플릿이 설치되어 있지 않습니다"))
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


def test_a_qa_failed_godot_run_is_packaged_exactly_like_a_passing_one(tmp_path, monkeypatch):
    """The two exits used to diverge where it could not be seen: a passing run got a web build
    attempt and a failing one was never even offered a build. The folder looked identically
    packaged - project, launcher, manifest all present - so the missing build read as "export
    templates are absent" rather than "this path never asked". A reviewer then could not open the
    game in the browser to judge the very findings they were handed.
    """
    import game_studio.graph as gm

    calls = []
    monkeypatch.setattr(gm, "godot_version", lambda: "4.7.2.stable")
    monkeypatch.setattr(gm, "write_launch_script", lambda target: target / "run.bat")
    monkeypatch.setattr(gm, "export_web", lambda project, out: calls.append(project) or (True, "ok"))

    passed = tmp_path / "ok"
    passed.mkdir()
    (passed / "project.godot").write_text("config_version=5\n", encoding="utf-8")
    ship = gm.package_node({**state_for(passed), "art": {}, "implementation_plan": {},
                            "qa": {"status": "pass", "findings": []},
                            "design_review": {"checks": []}})

    failed = tmp_path / "ng"
    failed.mkdir()
    (failed / "project.godot").write_text("config_version=5\n", encoding="utf-8")
    draft = gm.abandoned_node({**state_for(failed), "art": {},
                               "qa": {"status": "repair", "findings": ["미충족: 가속"]}})

    assert len(calls) == 2, "both exits have to attempt the web build"
    # Same keys, same artifacts - only the paths and the honesty about the verdict differ.
    assert set(ship) == set(draft) - {"qa_report_path", "trace_notes"}
    assert draft["game_path"].endswith("index.html"), "a failed run is still playable in browser"

    for folder, mode in ((passed, "model_generated"), (failed, "model_generated_qa_failed")):
        manifest = json.loads((folder / "production-manifest.json").read_text(encoding="utf-8"))
        assert manifest["web_export"] == {"ok": True, "detail": "ok"}
        assert manifest["launch_script"].endswith("run.bat")
        assert manifest["godot_version"] == "4.7.2.stable"
        # What must stay different: the failed run never claims it passed.
        assert manifest["generation_mode"] == mode
    outstanding = json.loads((failed / "production-manifest.json").read_text(encoding="utf-8"))
    assert outstanding["qa_outstanding"] == ["미충족: 가속"]


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


def test_the_launch_flags_are_a_combination_windows_actually_accepts():
    """WinError 87. CREATE_NEW_CONSOLE and DETACHED_PROCESS are mutually exclusive on Windows -
    ORing them is not "more detached", it is an invalid pair that CreateProcess rejects outright,
    before the launcher ever runs. The run button failed with "매개 변수가 틀립니다" every time.
    """
    import os
    import subprocess

    from game_studio import server

    passed = {}
    original = subprocess.Popen

    class Recorder(original):
        def __init__(self, *args, **kwargs):
            passed.update(kwargs)
            super().__init__(["cmd", "/c", "exit"] if os.name == "nt" else ["true"],
                             **{k: v for k, v in kwargs.items() if k != "cwd"})

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(subprocess, "Popen", Recorder)
    try:
        server._spawn_detached(__import__("pathlib").Path("x.bat"),
                               __import__("pathlib").Path("."))
    finally:
        monkeypatch.undo()

    if os.name == "nt":
        flags = passed.get("creationflags", 0)
        assert flags & subprocess.CREATE_NEW_CONSOLE, "the game needs its own console"
        assert not flags & subprocess.DETACHED_PROCESS, \
            "the two are mutually exclusive; together they are WinError 87"
        # Proven invalid rather than asserted to be.
        with pytest.raises(OSError):
            subprocess.Popen(
                ["cmd", "/c", "exit"],
                creationflags=subprocess.CREATE_NEW_CONSOLE | subprocess.DETACHED_PROCESS,
            ).wait(timeout=30)

    # run.bat pauses when the engine is missing or the game errors, so its output is the only
    # explanation a failed launch ever gets. Discarding it leaves a process waiting forever on a
    # prompt nobody can see.
    assert subprocess.DEVNULL not in (passed.get("stdout"), passed.get("stderr"))


def test_the_run_button_really_starts_the_launcher_it_was_given(tmp_path, monkeypatch):
    """End to end through the endpoint, with a launcher that proves it ran. The flag bug returned
    HTTP 500 from a call that looked correct everywhere else, so the wiring is only worth trusting
    when something on disk changes."""
    import sys

    from fastapi.testclient import TestClient

    from game_studio import server

    monkeypatch.setattr(server, "GAME_OUTPUT_ROOT", tmp_path)
    run_dir = tmp_path / "proofrun"
    run_dir.mkdir()
    proof = run_dir / "PROOF.txt"
    script = run_dir / "run.bat"
    if sys.platform == "win32":
        script.write_text(f'@echo off\r\necho ran> "{proof}"\r\n', encoding="utf-8")
    else:
        script.write_text(f'#!/bin/sh\necho ran > "{proof}"\n', encoding="utf-8")
        script.chmod(0o755)

    with TestClient(server.app) as client:
        assert client.post("/api/runs/proofrun/launch").status_code == 202

    for _ in range(100):
        if proof.exists():
            break
        time.sleep(0.1)
    assert proof.exists(), "the launcher has to actually run, not just return 202"


def _godot_web_run(root: pathlib.Path, run_id: str = "abc123") -> pathlib.Path:
    """A delivered Godot run as packaging leaves it: the project, plus a web export in build/."""
    folder = root / f"미로_godot_{run_id}"
    (folder / "build").mkdir(parents=True)
    (folder / "assets").mkdir()
    (folder / "production-manifest.json").write_text('{"engine": "godot"}', encoding="utf-8")
    (folder / "project.godot").write_text("config_version=5\n", encoding="utf-8")
    (folder / "main.gd").write_text("extends Node2D\n", encoding="utf-8")
    (folder / "assets" / "ghost.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    (folder / "assets" / "ghost.png.import").write_text("[remap]\n", encoding="utf-8")
    (folder / "build" / "index.html").write_text("<html><canvas></canvas></html>", encoding="utf-8")
    (folder / "build" / "index.js").write_text("// engine loader", encoding="utf-8")
    (folder / "build" / "index.wasm").write_bytes(b"\x00asm\x01\x00\x00\x00")
    (folder / "build" / "index.pck").write_bytes(b"GDPC")
    return folder


def test_a_godot_web_build_is_served_the_way_the_browser_asks_for_it(tmp_path, monkeypatch):
    """Export templates make the web build exist; this is what makes it RUN.

    A Godot export is not one page, it is a page plus four files the browser fetches on its own -
    index.js, index.wasm, index.pck and the audio worklets - and two things stopped every one of
    them. The export lives in build/, but assets were resolved against the run folder, so
    "index.wasm" beside the page was looked for a directory too high. And .wasm and .pck were not
    on the allowlist at all, so even the right path was refused.

    The result was a page that loaded and then died fetching its own engine, which reads as "the
    generated game is broken" and is nothing of the kind.
    """
    from fastapi.testclient import TestClient

    from game_studio import server

    monkeypatch.setattr(server, "GAME_OUTPUT_ROOT", tmp_path)
    server._RUN_FOLDERS.clear()
    _godot_web_run(tmp_path)

    with TestClient(server.app) as client:
        page = client.get("/games/abc123/")
        assert page.status_code == 200 and "<canvas>" in page.text, "the export's page, not the project"

        for name in ("index.js", "index.wasm", "index.pck"):
            assert client.get(f"/games/abc123/{name}").status_code == 200, (
                f"{name} is fetched by the page as its own sibling, out of build/")

        # A .wasm served as anything else cannot be stream-compiled, and a 38MB engine binary is
        # exactly the case that depends on it.
        assert client.get("/games/abc123/index.wasm").headers["content-type"] == "application/wasm"

        # The run folder stays reachable too: an HTML5 run's page IS the run folder, and the sprite
        # review panel reads assets/ from there on every engine.
        assert client.get("/games/abc123/assets/ghost.png").status_code == 200


def test_serving_a_web_build_does_not_open_the_project_up(tmp_path, monkeypatch):
    """Two bases to resolve against is two chances to escape, so containment is checked on the
    resolved path rather than on the request. And the allowlist stays an allowlist: this route
    serves out of the user's own output folder, and "anything under the run folder" would hand out
    the engine config and the source with it."""
    from fastapi.testclient import TestClient

    from game_studio import server

    monkeypatch.setattr(server, "GAME_OUTPUT_ROOT", tmp_path)
    server._RUN_FOLDERS.clear()
    _godot_web_run(tmp_path)
    (tmp_path.parent / "secret.txt").write_text("not yours", encoding="utf-8")

    with TestClient(server.app) as client:
        for private in ("project.godot", "main.gd", "assets/ghost.png.import"):
            assert client.get(f"/games/abc123/{private}").status_code == 404, private
        for escape in ("../secret.txt", "../../secret.txt", "build/../../secret.txt",
                       "..%2F..%2Fsecret.txt"):
            assert client.get(f"/games/abc123/{escape}").status_code == 404, escape


def test_every_godot_project_ships_a_font_that_can_draw_korean(tmp_path, monkeypatch):
    """Godot's built-in font has no Hangul and every label this pipeline writes is Korean. On the
    desktop a system-font fallback sometimes hides that; a web export has no system fonts at all,
    so a delivered build came back with its entire HUD as tofu boxes - 목숨, 속도, GAME OVER - while
    the game underneath ran perfectly.

    Written by the pipeline rather than asked of the code agent, for the reason every other
    mechanical guarantee here is: a step the model can forget is one that will be forgotten, and
    this one fails silently - the build succeeds and the text is unreadable.
    """
    from game_studio import godot

    project = tmp_path / "game"
    project.mkdir()
    (project / "project.godot").write_text(
        'config_version=5\n\n[application]\nrun/main_scene="res://main.tscn"\n', encoding="utf-8")
    imported = []
    monkeypatch.setattr(godot, "_run", lambda args, timeout: (imported.append(args), (0, ""))[1])

    note = godot.install_korean_font(project)
    assert (project / godot.FONT_DIR / godot.FONT_FILE).is_file(), note
    assert (project / godot.FONT_DIR / "OFL.txt").is_file(), "the licence travels with the font"

    config = (project / "project.godot").read_text(encoding="utf-8")
    assert '[gui]' in config and 'theme/custom_font="res://fonts/' in config
    assert 'run/main_scene="res://main.tscn"' in config, "the agent's own settings survive"

    # A file dropped into the project is not yet a resource: without this pass Godot answers
    # "No loader found for resource" and silently falls back to the built-in font - measured, with
    # the font sitting in the .pck the whole time.
    assert any("--import" in args for args in imported), "the import pass has to actually run"


def test_the_font_setting_is_replaced_rather_than_added_twice(tmp_path, monkeypatch):
    """Packaging runs after the agent has stopped writing project.godot, and a revision packages
    again. Two [gui] sections, or two custom_font lines, is a file Godot reads the wrong half of."""
    from game_studio import godot

    project = tmp_path / "game"
    project.mkdir()
    (project / "project.godot").write_text(
        'config_version=5\n\n[gui]\n\ntheme/custom_font="res://old.ttf"\n', encoding="utf-8")
    monkeypatch.setattr(godot, "_run", lambda args, timeout: (0, ""))

    godot.install_korean_font(project)
    config = (project / "project.godot").read_text(encoding="utf-8")
    assert config.count("[gui]") == 1
    assert config.count("theme/custom_font") == 1
    assert "old.ttf" not in config
