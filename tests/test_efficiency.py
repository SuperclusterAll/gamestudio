"""What the pipeline stopped paying for, and the memory it gained.

Every item here was measured before it was changed. A measured run spent 85% of its tokens on
input, re-sending the same system prompt and the same ~2,700 tokens of tool definitions on each of
the code agent's twenty calls; the build loop spent a model call asking for a verdict that was
already knowable at write time; the code agent's whole transcript - whole games included, as
write_game_file arguments - was written into every checkpoint and read by nothing; and a pending
human approval, which the graph is designed to hold durably, died with the process.
"""

import json
import pathlib

import pytest


def test_the_stable_prefix_is_cached_on_the_models_that_support_it(monkeypatch):
    """Prompt caching is the one lever that hits the dominant cost directly. Measured against a
    real Bedrock call on this prefix: the second request read 6,520 of its 6,523 input tokens from
    cache, at a tenth of the input rate."""
    from game_studio.agents import cache_control_for

    monkeypatch.delenv("BEDROCK_PROMPT_CACHE_TTL", raising=False)
    assert cache_control_for("global.anthropic.claude-sonnet-4-6") == {
        "cache_control": {"ttl": "5m"}}
    assert cache_control_for("global.anthropic.claude-haiku-4-5-20251001-v1:0")
    # Nova has its own rules for where a cachePoint may sit and rejects an unsupported combination
    # outright. A cost optimisation must never be able to fail a run.
    assert cache_control_for("us.amazon.nova-2-lite-v1:0") == {}
    assert cache_control_for(None) == {} and cache_control_for("") == {}


def test_caching_can_be_switched_off_without_touching_code(monkeypatch):
    import importlib

    from game_studio import agents

    monkeypatch.setenv("BEDROCK_PROMPT_CACHE_TTL", "off")
    importlib.reload(agents)
    try:
        assert agents.cache_control_for("global.anthropic.claude-sonnet-4-6") == {}
    finally:
        monkeypatch.delenv("BEDROCK_PROMPT_CACHE_TTL")
        importlib.reload(agents)


def test_cached_input_is_not_billed_as_fresh_input():
    """Counting a cache read at the full input price makes the estimate wrong in exactly the
    situation caching exists to create, and hides whether caching is engaging at all."""
    from game_studio.agents import CACHE_READ_RATE, _UsageTracker, price_per_mtok

    model = "global.anthropic.claude-sonnet-4-6"
    reported = {}

    class Message:
        response_metadata = {"model_name": model}
        usage_metadata = {"input_tokens": 10_000, "output_tokens": 0,
                          "input_token_details": {"cache_read": 9_000, "cache_creation": 0}}

    class Generation:
        message = Message()

    class Response:
        generations = [[Generation()]]
        llm_output = None

    import game_studio.agents as agents_module

    class Writer:
        def __call__(self, payload):
            reported.update(payload)

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr("langgraph.config.get_stream_writer", lambda: Writer())
    try:
        _UsageTracker().on_llm_end(Response())
    finally:
        monkeypatch.undo()

    assert reported["cache_read_tokens"] == 9_000
    price_in = price_per_mtok(model)[0]
    expected = (1_000 * price_in + 9_000 * price_in * CACHE_READ_RATE) / 1_000_000
    assert reported["cost_usd"] == pytest.approx(expected)
    # The whole point: the same tokens billed as fresh would cost far more.
    assert reported["cost_usd"] < (10_000 * price_in / 1_000_000) / 2
    assert agents_module.CACHE_READ_RATE < 1, "a cache read must be cheaper than a fresh token"


def test_a_write_answers_with_its_own_verdict_instead_of_costing_another_turn(tmp_path):
    """static_qa is a keyword scan plus a JavaScript parse - no model call, and cached per content.
    Making the agent spend a turn to ask for it doubled the round trips of the whole build loop."""
    from game_studio.agent_tools import repair_html, write_game_file

    state = {"concept": {"title": "t", "elevator_pitch": "p", "player_goal": "g",
                         "controls": ["a", "b"], "core_loop": ["1", "2", "3"],
                         "difficulty_curve": "d", "visual_direction": "v"},
             "output_dir": str(tmp_path), "workspace_dir": str(tmp_path)}

    broken = "<html><canvas></canvas></html>"
    result = write_game_file.invoke({"html": broken, "state": state})
    assert "정적 QA 실패" in result, "the verdict comes back with the write"
    assert "Missing" in result, "and names what is actually wrong"
    assert "run_static_qa" not in result, "the agent must not be sent to ask again"

    good = ("<html><canvas>score restart keydown KeyW KeyA KeyS KeyD "
            "requestAnimationFrame</canvas></html>")
    assert "정적 QA 통과" in repair_html.invoke({"html": good, "state": state})


def test_the_tool_list_carries_nothing_the_model_gains_from_calling():
    """Every tool's description is re-sent on every model call, so a tool that only writes a file
    the agent never reads is a standing tax on the whole loop."""
    from game_studio.agent_tools import GAME_TOOLS
    from game_studio.godot_tools import GODOT_TOOLS

    for tools in (GAME_TOOLS, GODOT_TOOLS):
        names = [tool.name for tool in tools]
        assert "generate_asset" not in names, "persisting the art plan needs no model"
        # What the agent genuinely cannot do without a tool is still there.
        assert "generate_comfyui_image" in names and "list_game_assets" in names


def test_the_art_plan_still_reaches_disk_without_a_tool(tmp_path):
    from game_studio.graph import _persist_art_plan
    from game_studio.models import ArtDirection

    _persist_art_plan(tmp_path, ArtDirection(palette={"bg": "#000"}, image_prompt="x",
                                             asset_plan=["player: 우주선"], canvas_effects=[]))
    saved = json.loads((tmp_path / "assets" / "canvas-art-plan.json").read_text(encoding="utf-8"))
    assert saved["asset_plan"] == ["player: 우주선"]


def test_the_trace_trail_is_bounded(tmp_path):
    """It rides along in every checkpoint and in the published manifest, and the director's entire
    brief used to go into it verbatim."""
    import game_studio.graph as gm

    trail = []
    for index in range(gm.TRACE_NOTE_LIMIT * 3):
        trail = gm._note_trail({"trace_notes": trail}, f"entry {index}")
    assert len(trail) == gm.TRACE_NOTE_LIMIT
    assert trail[-1] == f"entry {gm.TRACE_NOTE_LIMIT * 3 - 1}", "the newest entries are kept"


def test_a_pending_approval_survives_a_restart(tmp_path, monkeypatch):
    """The approval gate is a durable interrupt - the graph stops and waits for a human Command
    that may arrive days later - but durability was only ever as good as the checkpointer behind
    it, and an in-memory one lasts exactly as long as the process. A reviewer who left a design
    document open overnight and restarted the dashboard lost the run."""
    import importlib

    import game_studio.graph as gm
    from game_studio.models import ArtDirection, DesignReview, GameConcept, ImplementationPlan

    monkeypatch.setenv("CHECKPOINT_DB", str(tmp_path / "ck.sqlite"))
    from game_studio import server
    importlib.reload(server)

    concept = GameConcept(title="지속성", elevator_pitch="p", player_goal="g",
                          controls=["a", "b"], core_loop=["1", "2", "3"],
                          difficulty_curve="d", visual_direction="v")
    plan = ImplementationPlan(genre="퍼즐", mechanics=["a", "b", "c"], win_condition="w",
                              loss_condition="l", state_transitions=["1", "2", "3"],
                              acceptance_tests=["t1", "t2", "t3"])
    monkeypatch.setattr(gm, "run_director", lambda *a, **kw: "plan")
    monkeypatch.setattr(gm, "create_concept", lambda *a, **kw: concept)
    monkeypatch.setattr(gm, "create_art", lambda *a, **kw: ArtDirection(
        palette={}, image_prompt="x", asset_plan=[], canvas_effects=[]))
    monkeypatch.setattr(gm, "_structured", lambda schema, *a, **kw:
                        plan if schema is ImplementationPlan else DesignReview(checks=[]))

    config = {"configurable": {"thread_id": "survive"}, "recursion_limit": 200}
    payload = {"brief": "b", "output_dir": str(tmp_path), "workspace_dir": str(tmp_path),
               "use_llm": True, "generate_images": False, "repair_attempts": 0, "trace_notes": []}

    gm.build_graph(server._checkpointer()).invoke(payload, config)
    # A brand new graph and a brand new checkpointer, as a restarted dashboard would build.
    revived = gm.build_graph(server._checkpointer()).get_state(config)
    assert revived.next == ("approval",), "the run is still waiting for its human"
    assert revived.values["design_document"]["title"] == "지속성"

    monkeypatch.delenv("CHECKPOINT_DB")
    importlib.reload(server)


def test_a_dashboard_that_cannot_write_its_database_still_runs(monkeypatch, tmp_path):
    """Durability is worth having and not worth failing over."""
    import importlib

    from game_studio import server

    monkeypatch.setenv("CHECKPOINT_DB", str(tmp_path / "missing-dir" / "x" / "ck.sqlite"))
    importlib.reload(server)
    try:
        monkeypatch.setattr(server.Path, "mkdir",
                            lambda *a, **kw: (_ for _ in ()).throw(OSError("read-only")))
        assert server._checkpointer() is not None
    finally:
        monkeypatch.undo()
        monkeypatch.delenv("CHECKPOINT_DB", raising=False)
        importlib.reload(server)


def test_the_studio_remembers_what_it_already_shipped(tmp_path):
    """Assigning a genre from the run seed spreads runs across the reference table, but inside one
    genre the model still reaches for the same design - it has no way to know what it built last
    time. The manifests already record exactly that, so the output folder is the memory."""
    import time

    from game_studio.agents import RECENT_TITLE_LIMIT, recent_productions

    for index, (title, genre) in enumerate([("별똥별 사냥꾼", "슈팅"), ("블록 낙하", "퍼즐"),
                                            ("불꽃 생존자", "액션 생존")]):
        run = tmp_path / f"run{index}"
        run.mkdir()
        (run / "production-manifest.json").write_text(json.dumps(
            {"concept": {"title": title}, "implementation_plan": {"genre": genre}},
            ensure_ascii=False), encoding="utf-8")
        time.sleep(0.02)

    recent = recent_productions(tmp_path)
    assert recent[0] == "불꽃 생존자 (액션 생존)", "newest first"
    assert len(recent) == 3 and "별똥별 사냥꾼 (슈팅)" in recent
    assert len(recent) <= RECENT_TITLE_LIMIT

    # Nothing to remember, and nothing to crash on.
    assert recent_productions(None) == []
    assert recent_productions(tmp_path / "does-not-exist") == []
    broken = tmp_path / "broken"
    broken.mkdir()
    (broken / "production-manifest.json").write_text("{ not json", encoding="utf-8")
    assert len(recent_productions(tmp_path)) == 3, "a corrupt manifest is skipped, not fatal"


def test_the_idea_agent_is_told_what_not_to_repeat(tmp_path, monkeypatch):
    from game_studio import agents

    run = tmp_path / "run0"
    run.mkdir()
    (run / "production-manifest.json").write_text(json.dumps(
        {"concept": {"title": "별똥별 사냥꾼"}, "implementation_plan": {"genre": "슈팅"}},
        ensure_ascii=False), encoding="utf-8")

    captured = {}
    monkeypatch.setattr(agents, "_structured",
                        lambda schema, system, user, *a, **kw: captured.update(user=user))
    agents.create_concept("Requested genre: 퍼즐\nRun seed: x", True, "m", output_root=tmp_path)

    assert "별똥별 사냥꾼 (슈팅)" in captured["user"]
    assert "뚜렷하게 다른 게임" in captured["user"]


def test_an_explicit_request_outranks_every_nudge_this_module_adds(tmp_path, monkeypatch):
    """Asked for a faithful Tetris, the pipeline answered with a different game. Three things were
    pushing it there at once: the standing prompt wants one twist of the borrowed loop and an
    original work, and the recent-productions nudge was telling it to avoid the puzzle game the
    studio shipped last week. All three are right in the absence of a request and wrong against one.
    """
    from game_studio import agents

    run = tmp_path / "r0"
    run.mkdir()
    (run / "production-manifest.json").write_text(json.dumps(
        {"concept": {"title": "블록 낙하 합산"}, "implementation_plan": {"genre": "퍼즐"}},
        ensure_ascii=False), encoding="utf-8")

    captured = {}
    monkeypatch.setattr(agents, "_structured",
                        lambda schema, system, user, *a, **kw: captured.update(user=user,
                                                                              system=system))

    asked = ("Requested genre: 퍼즐\n"
             "Player brief: 테트리스랑 완전히 똑같은 게임을 만들어줘\nRun seed: a")
    agents.create_concept(asked, True, "m", output_root=tmp_path)
    assert "다른 모든 지침보다 우선" in captured["user"], "the request has to be given priority"
    assert "뚜렷하게 다른" not in captured["user"], \
        "variety must never be optimised for against an explicit request"
    assert "독창적인 변형을 더하지 말고" in captured["user"]
    # And the standing prompt has to say the same thing, or the two fight each other.
    assert "outranks every other instruction" in captured["system"]

    captured.clear()
    auto = ("Requested genre: 자동 기획\n"
            "Player brief: 사용자 경험 없이 독자적으로 기획하세요.\nRun seed: a")
    agents.create_concept(auto, True, "m", output_root=tmp_path)
    assert "뚜렷하게 다른" in captured["user"], "with no request, variety is exactly what to want"
    assert "다른 모든 지침보다 우선" not in captured["user"]


def test_whether_a_human_wrote_the_brief_is_answered_precisely():
    """It decides which way two separate nudges point, so guessing from length would not do."""
    from game_studio.agents import player_requested

    assert player_requested(
        "Requested genre: 퍼즐\nPlayer brief: 테트리스처럼 만들어줘\nRun seed: x") == "테트리스처럼 만들어줘"
    for empty in ("Player brief: 사용자 경험 없이 독자적으로 기획하세요.",
                  "Player brief: 사용자 경험 미입력 — Agent 자동 기획",
                  "Requested genre: 퍼즐"):
        assert player_requested(empty) == ""


def test_finished_runs_come_back_after_a_restart(tmp_path, monkeypatch):
    """Checkpoints became durable and the run list did not follow: StudioService.runs is an
    in-memory dict nothing repopulated, so every restart emptied the dashboard. A finished Godot
    project with its run.bat sitting on disk became unreachable - the run could not be selected,
    so its launch button was never shown, and pressing where it used to be did nothing at all.
    """
    from game_studio import server

    monkeypatch.setattr(server, "GAME_OUTPUT_ROOT", tmp_path)

    godot = tmp_path / "a1b2c3d4e5f6"
    godot.mkdir()
    (godot / "project.godot").write_text("config_version=5\n", encoding="utf-8")
    (godot / "run.bat").write_text("@echo off\n", encoding="utf-8")
    (godot / "production-manifest.json").write_text(json.dumps({
        "engine": "godot", "concept": {"title": "복원된 게임", "elevator_pitch": "p"},
        "implementation_plan": {"genre": "퍼즐"}, "qa": {"status": "pass", "findings": []},
        "code_model_id": "m", "generation_mode": "model_generated",
    }, ensure_ascii=False), encoding="utf-8")

    html = tmp_path / "b1b2c3d4e5f6"
    html.mkdir()
    (html / "index.html").write_text("<html></html>", encoding="utf-8")
    (html / "production-manifest.json").write_text(json.dumps({
        "engine": "html5", "concept": {"title": "웹 게임"}, "qa": {"status": "repair"},
        "generation_mode": "model_generated_qa_failed", "qa_outstanding": ["미충족: 가속"],
    }, ensure_ascii=False), encoding="utf-8")

    # Folders that produced nothing are not runs; neither is a stray directory.
    (tmp_path / "c1b2c3d4e5f6").mkdir()
    (tmp_path / "ember-vault").mkdir()

    restored = server._restore_finished_runs()
    assert set(restored) == {"a1b2c3d4e5f6", "b1b2c3d4e5f6"}

    g = restored["a1b2c3d4e5f6"]
    assert g.engine == "godot" and g.status == "completed"
    assert g.state["launch_script_path"] == str(godot / "run.bat"), "the launch button needs this"
    assert g.state["godot_project_path"].endswith("project.godot")
    assert "game_path" not in g.state, "no web build, so nothing to embed"
    assert g.state["design_document"]["title"] == "복원된 게임"

    h = restored["b1b2c3d4e5f6"]
    assert h.status == "qa_failed", "a run that shipped with open findings says so"
    assert h.state["game_path"].endswith("index.html")


def test_a_restored_run_offers_only_artifacts_that_still_exist(tmp_path, monkeypatch):
    """A folder the user has since cleaned out must not be offered as playable."""
    from game_studio import server

    monkeypatch.setattr(server, "GAME_OUTPUT_ROOT", tmp_path)
    run = tmp_path / "d1b2c3d4e5f6"
    run.mkdir()
    (run / "production-manifest.json").write_text(json.dumps({
        "engine": "godot", "concept": {"title": "t"}, "qa": {},
        "generation_mode": "model_generated",
    }, ensure_ascii=False), encoding="utf-8")

    state = server._restore_finished_runs()["d1b2c3d4e5f6"].state
    assert "launch_script_path" not in state and "godot_project_path" not in state
    assert "game_path" not in state


def test_a_director_that_fails_returns_no_brief_rather_than_its_error(monkeypatch):
    """This return value is handed to the planners as "Production director's brief", so a failure
    has to be silence rather than an explanation. A run actually opened by telling the idea agent
    that its production brief was:

        Director fallback: TypeError: 'Overwrite' object is not iterable

    That particular TypeError is gone with the streaming loop that raised it - the director no
    longer streams a deepagents graph whose middleware bypasses its own reducer - but the reason
    the sentence reached the planners at all was this return value, and that is what is pinned
    here. Every stage after the director works without a brief.
    """
    from game_studio import agents

    monkeypatch.setattr(agents, "DIRECTOR_TIMEOUT_SECONDS", 60)
    monkeypatch.setattr(agents, "_model",
                        lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("Bedrock 거부")))
    reported = []
    brief = agents.run_director("테트리스", True, "m",
                                on_step=lambda node, tools, text: reported.append(text))

    assert brief == "", "a failure must not become the plan the planners work from"
    assert any("총괄 감독 실패" in text and "Bedrock 거부" in text for text in reported),         "and it still has to be visible - silently swallowing it is the other failure"


def test_a_finished_run_records_what_it_actually_consumed(tmp_path, monkeypatch):
    """Only this process ever sees these totals - the usage callback reports each model call onto
    the run's stream and the server adds them up, so the graph node that writes the manifest has no
    access to them. They lived in an in-memory dict and died with the process, which made every
    question about consumption unanswerable after the fact: is 50 calls the right budget, does
    Godot cost more than the Canvas path, did that model change help. All of them need runs to
    compare and there were none, because nothing was kept.
    """
    import json as _json

    from game_studio import server

    monkeypatch.setattr(server, "GAME_OUTPUT_ROOT", tmp_path)
    folder = tmp_path / "3f0d505d3038"
    folder.mkdir()
    manifest = folder / "production-manifest.json"
    manifest.write_text(_json.dumps({"engine": "godot", "concept": {"title": "블록 강하"},
                                     "implementation_plan": {"genre": "퍼즐"}}), encoding="utf-8")

    run = server.Run(id="3f0d505d3038", genre="퍼즐", brief="b", model_id="m",
                     config={}, engine="godot", status="completed")
    run.usage = {"input_tokens": 480_000, "output_tokens": 22_405,
                 "total_tokens": 502_405, "cost_usd": 0.6027, "calls": 19}
    run.usage_by_step = {"code": {"calls": 12}, "qa": {"calls": 3}}
    server._record_usage(run)

    written = _json.loads(manifest.read_text(encoding="utf-8"))["usage"]
    assert written["calls"] == 19 and written["total_tokens"] == 502_405
    assert written["by_step"]["code"]["calls"] == 12, "per-step totals are how an engine is compared"
    assert written["status"] == "completed" and written["engine"] == "godot"
    assert written["recorded_at"], "a measurement without a date cannot be a trend"
    assert _json.loads(manifest.read_text(encoding="utf-8"))["concept"]["title"] == "블록 강하"

    # And it comes back with the run, so a restarted dashboard shows the totals rather than zeroes.
    restored = server._restore_finished_runs()["3f0d505d3038"]
    assert restored.usage["calls"] == 19 and restored.usage["total_tokens"] == 502_405
    assert restored.usage_by_step["code"]["calls"] == 12

    # A run that never called a model has nothing to record, and must not write an empty block.
    quiet = server.Run(id="3f0d505d3038", genre="q", brief="b", model_id="m", config={})
    manifest.write_text(_json.dumps({"engine": "godot"}), encoding="utf-8")
    server._record_usage(quiet)
    assert "usage" not in _json.loads(manifest.read_text(encoding="utf-8"))


def test_packaging_does_not_erase_what_it_did_not_write(tmp_path):
    """Packaging composes a fresh manifest from the run's own state and writes it over whatever was
    there. Two fields are written by somebody else and were being destroyed by that: `adoption`,
    which the dashboard writes when a revision starts - so every successful rework deleted the
    record of having been reworked - and `usage`, which only the server can supply and which is
    added after this file is written.
    """
    import json as _json

    from game_studio.graph import _write_manifest

    path = tmp_path / "production-manifest.json"
    path.write_text(_json.dumps({
        "concept": {"title": "옛 제목"},
        "adoption": {"adopted": True, "note": "점프를 가볍게"},
        "usage": {"calls": 19, "total_tokens": 502_405},
    }, ensure_ascii=False), encoding="utf-8")

    _write_manifest(tmp_path, {"concept": {"title": "새 제목"}, "generation_mode": "model_generated"})

    after = _json.loads(path.read_text(encoding="utf-8"))
    assert after["concept"]["title"] == "새 제목", "the run's own fields still win"
    assert after["generation_mode"] == "model_generated"
    assert after["adoption"]["note"] == "점프를 가볍게", "and what it did not write survives"
    assert after["usage"]["calls"] == 19

    # A run that does supply one of them overwrites it, and a first write needs no previous file.
    _write_manifest(tmp_path, {"usage": {"calls": 3}})
    assert _json.loads(path.read_text(encoding="utf-8"))["usage"]["calls"] == 3
    fresh = tmp_path / "새폴더"
    fresh.mkdir()
    _write_manifest(fresh, {"concept": {}})
    assert (fresh / "production-manifest.json").is_file()


def test_a_revision_adds_to_the_games_cost_instead_of_replacing_it(tmp_path, monkeypatch):
    """A revision runs in the folder of the game it is revising and under that game's id, so
    overwriting meant a three-call revision that failed replaced the delivered game's 47 calls with
    its own - and stamped status "failed" onto a manifest whose generation_mode still said the game
    shipped. The dashboard read that back and under-reported the game's cost permanently.
    """
    import json as _json

    from game_studio import server

    monkeypatch.setattr(server, "GAME_OUTPUT_ROOT", tmp_path)
    folder = tmp_path / "블록-강하_godot_3f0d505d3038"
    folder.mkdir()
    manifest = folder / "production-manifest.json"
    manifest.write_text(_json.dumps({"generation_mode": "model_generated"}), encoding="utf-8")

    def record(calls, tokens, cost, status):
        run = server.Run(id="3f0d505d3038", genre="g", brief="b", model_id="m", config={},
                         engine="godot", status=status)
        run.usage = {"input_tokens": tokens - 100, "output_tokens": 100,
                     "total_tokens": tokens, "cost_usd": cost, "calls": calls}
        run.usage_by_step = {"code": {"calls": calls}}
        server._record_usage(run)

    record(47, 1_200_000, 2.10, "completed")   # the delivered game
    record(3, 1_000, 0.01, "failed")           # a revision that died early
    record(12, 300_000, 0.55, "completed")     # a revision that worked

    usage = _json.loads(manifest.read_text(encoding="utf-8"))["usage"]
    assert usage["calls"] == 62, "what this game cost is the sum of every run that built it"
    assert usage["total_tokens"] == 1_501_000
    assert usage["cost_usd"] == 2.66
    # Each attempt is still there individually, because both questions get asked.
    assert [(a["calls"], a["status"]) for a in usage["runs"]] == [
        (47, "completed"), (3, "failed"), (12, "completed")]
    assert usage["status"] == "completed", "the top level describes the latest attempt"

    # The cumulative figure is what a restarted dashboard shows for the game.
    manifest.write_text(_json.dumps({**_json.loads(manifest.read_text(encoding="utf-8")),
                                     "concept": {"title": "블록 강하"},
                                     "implementation_plan": {"genre": "퍼즐"}},
                                    ensure_ascii=False), encoding="utf-8")
    assert server._restore_finished_runs()["3f0d505d3038"].usage["calls"] == 62

    # Totals are carried forward, not re-summed from the list, so trimming it cannot corrupt them.
    for _ in range(server.USAGE_ATTEMPT_HISTORY + 5):
        record(1, 10, 0.001, "completed")
    grown = _json.loads(manifest.read_text(encoding="utf-8"))["usage"]
    assert len(grown["runs"]) == server.USAGE_ATTEMPT_HISTORY
    assert grown["calls"] == 62 + server.USAGE_ATTEMPT_HISTORY + 5


def test_resolving_a_run_folder_does_not_rescan_the_output_root_every_time(tmp_path, monkeypatch):
    """game_asset calls this once per asset: a page with thirty sprites against a few hundred game
    folders was thousands of stat calls, run synchronously inside async handlers - so it blocked
    the event loop and with it the live stream the progress view depends on."""
    from game_studio import server

    monkeypatch.setattr(server, "GAME_OUTPUT_ROOT", tmp_path)
    server._RUN_FOLDERS.clear()
    folder = tmp_path / "블록-강하_godot_3f0d505d3038"
    folder.mkdir()
    for n in range(30):
        (tmp_path / f"기타-게임-{n}_html5_{n:012x}").mkdir()

    scans = []
    real = server._scan_for_run
    monkeypatch.setattr(server, "_scan_for_run", lambda rid: scans.append(rid) or real(rid))

    assert [server.run_folder("3f0d505d3038") for _ in range(30)] == [folder.resolve()] * 30
    assert len(scans) == 1, "thirty assets must not mean thirty scans"

    # A folder that goes away corrects itself rather than serving a stale path forever.
    import shutil as _shutil
    _shutil.rmtree(folder)
    assert server.run_folder("3f0d505d3038") == (tmp_path / "3f0d505d3038").resolve()
    # And a miss is not cached, because the folder is usually about to be created by the run asking.
    folder.mkdir()
    assert server.run_folder("3f0d505d3038") == folder.resolve()


def test_an_unreadable_output_root_is_not_found_rather_than_a_crash(tmp_path, monkeypatch):
    """A permission problem or a broken junction in the output root used to turn every lookup into
    a 500. It is the same answer as "no such run"."""
    from game_studio import server

    monkeypatch.setattr(server, "GAME_OUTPUT_ROOT", tmp_path)
    server._RUN_FOLDERS.clear()
    monkeypatch.setattr(server.Path, "iterdir",
                        lambda self: (_ for _ in ()).throw(PermissionError("거부됨")))
    assert server.run_folder("3f0d505d3038") == (tmp_path / "3f0d505d3038").resolve()


def _graph_with(paused: set[str]):
    """A graph double whose state reports `next` only for the threads named as paused."""
    class Snapshot:
        def __init__(self, nxt): self.next = nxt

    class Graph:
        def get_state(self, config):
            thread = config["configurable"]["thread_id"]
            return Snapshot(("approval",) if thread in paused else ())
    return Graph()


def _seeded_saver(tmp_path, threads: dict[str, str]):
    """A real SqliteSaver holding one checkpoint per thread, stamped as given."""
    import sqlite3

    from langgraph.checkpoint.sqlite import SqliteSaver

    saver = SqliteSaver(sqlite3.connect(tmp_path / "cp.sqlite", check_same_thread=False))
    saver.setup()
    for thread_id, stamp in threads.items():
        config = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}
        saver.put(config, {"v": 1, "id": f"{thread_id}-1", "ts": stamp, "channel_values": {},
                           "channel_versions": {}, "versions_seen": {}}, {}, {})
    return saver


def test_finished_runs_stop_paying_for_checkpoints_and_paused_ones_never_do(tmp_path, monkeypatch):
    """The database exists for one thing: a pending approval has to survive a restart. It kept far
    more, because LangGraph snapshots the whole state at every super-step rather than a diff - a
    measured run wrote 229 checkpoints averaging 62KB, so one game costs 10-18MB and nothing ever
    removed any of it. One instance reached 246MB across 37 threads.

    What must never be deleted is a graph that stopped in the middle. `next` is empty only when the
    graph has nowhere left to go, so an approval waiting for a human and a run the process died
    during are both excluded by the same test - which is what makes this safe without having to
    know why a thread stopped.
    """
    from game_studio import server

    old = "2020-01-01T00:00:00+00:00"
    saver = _seeded_saver(tmp_path, {"done-1": old, "done-2": old, "waiting": old})
    monkeypatch.setattr(server, "CHECKPOINT_DB", str(tmp_path / "cp.sqlite"))

    finished = server._finished_threads(saver, _graph_with({"waiting"}))
    assert sorted(finished) == ["done-1", "done-2"]
    assert "waiting" not in finished, "a pending approval is the whole reason this file exists"

    removed, _freed = server.prune_checkpoints(saver, _graph_with({"waiting"}))
    assert removed == 2
    left = {row[0] for row in saver.conn.execute("SELECT DISTINCT thread_id FROM checkpoints")}
    assert left == {"waiting"}


def test_a_recent_thread_is_left_alone_whatever_its_state(tmp_path, monkeypatch):
    """The run may have finished seconds ago and still be on screen, and the space it holds is not
    worth the surprise."""
    from datetime import UTC, datetime

    from game_studio import server

    now = datetime.now(UTC).isoformat()
    saver = _seeded_saver(tmp_path, {"just-done": now})
    monkeypatch.setattr(server, "CHECKPOINT_DB", str(tmp_path / "cp.sqlite"))
    monkeypatch.setattr(server, "CHECKPOINT_RETENTION_DAYS", 3.0)
    assert server._finished_threads(saver, _graph_with(set())) == []

    # With retention off, the same thread is collected.
    monkeypatch.setattr(server, "CHECKPOINT_RETENTION_DAYS", 0.0)
    assert server._finished_threads(saver, _graph_with(set())) == ["just-done"]


def test_tidying_up_can_never_stop_the_dashboard_starting(tmp_path, monkeypatch):
    """Losing a checkpoint that should have been kept is far worse than keeping one that could have
    gone, so anything unclear is left alone - and a dashboard that cannot tidy its own database
    still has to start."""
    from game_studio import server

    monkeypatch.setattr(server, "CHECKPOINT_DB", str(tmp_path / "cp.sqlite"))
    saver = _seeded_saver(tmp_path, {"done": "2020-01-01T00:00:00+00:00"})

    class Exploding:
        def get_state(self, _config):
            raise RuntimeError("상태를 읽을 수 없습니다")

    assert server._finished_threads(saver, Exploding()) == [], "a thread that will not load stays"
    assert server.prune_checkpoints(saver, Exploding()) == (0, 0.0)

    # An in-memory saver has no threads to delete and must not be asked to.
    from langgraph.checkpoint.memory import InMemorySaver
    assert server.prune_checkpoints(InMemorySaver(), _graph_with(set())) == (0, 0.0)


def test_the_studios_own_state_lives_in_the_project_not_the_output_folder(tmp_path, monkeypatch):
    """The checkpoint database and the art memory were being written into GAME_OUTPUT_DIR, which
    put a 249MB SQLite file and a vector store in the directory a person browses to find their
    games. Neither is a deliverable: they are this installation's private state, nothing outside
    this process opens them, and they belong with the code that reads them.

    Created on first use, so a clone on another machine needs no setup step - and ignored by git,
    so they never travel.
    """
    import importlib

    from game_studio import art_memory
    from game_studio.models import project_data_dir

    root = project_data_dir()
    assert root.name == "data"
    assert root.parent == pathlib.Path(__file__).resolve().parents[1], "inside the project"

    monkeypatch.setattr(art_memory, "ART_MEMORY_DIR", "")
    assert art_memory.memory_dir() == root / "art-memory"

    from game_studio import server
    importlib.reload(server)
    try:
        assert pathlib.Path(server.CHECKPOINT_DB).parent == root
        assert server.GAME_OUTPUT_ROOT != root, "games stay where the games are"
    finally:
        importlib.reload(server)

    # A machine that has never run this finds nothing there, and both stores make themselves.
    fresh = tmp_path / "새-클론" / "data"
    monkeypatch.setenv("STUDIO_DATA_DIR", str(fresh))
    monkeypatch.setattr(art_memory, "ART_MEMORY_DIR", "")
    assert not fresh.exists()
    assert art_memory.remember(name="x.png", prompt="p", role="enemy", genre="g", run_id="r",
                               entry={"kind": "sprite", "width": 200, "height": 200,
                                      "removed_share": 0.5})
    assert (fresh / "art-memory").is_dir(), "the store creates its own directory"


def test_the_data_directory_is_never_committed():
    """A checkpoint file reached 249MB in normal use, and the vector store is machine-specific."""
    ignored = pathlib.Path(__file__).resolve().parents[1] / ".gitignore"
    assert "data/" in ignored.read_text(encoding="utf-8").splitlines()


def test_the_build_stops_on_its_call_budget_and_never_on_the_recursion_limit():
    """The two ceilings on one loop, and they are not interchangeable.

    ModelCallLimitMiddleware(exit_behavior="end") is the intended stop: it ends the loop cleanly and
    the run carries on into QA, repair and packaging with whatever is on disk. The graph's recursion
    limit is a backstop, and reaching it raises GraphRecursionError, which fails the run outright.

    They were written down separately and drifted. A flat recursion_limit of 120 against a budget of
    50 meant the clean stop could never fire - measured at 5.0 super-steps per model call, 120 steps
    is 24 calls - so every build that needed more than 24 died and one of them discarded a 25KB
    playable game. Deriving one from the other is what makes that impossible; this is the assertion
    that says so.
    """
    from game_studio.code_agent import (
        CODE_RECURSION_LIMIT,
        MODEL_CALL_LIMIT,
        RECURSION_HEADROOM,
        STEPS_PER_MODEL_CALL,
    )

    MEASURED_STEPS_PER_CALL = 5.0  # 120 super-steps / 24 model calls, run 471e72a794bf
    reachable = (CODE_RECURSION_LIMIT - RECURSION_HEADROOM) / MEASURED_STEPS_PER_CALL
    assert reachable > MODEL_CALL_LIMIT, (
        f"recursion limit {CODE_RECURSION_LIMIT} allows only {reachable:.0f} calls, "
        f"below the {MODEL_CALL_LIMIT} budget - the clean stop can never fire")

    # Padded above the measurement, never trimmed to it: the per-turn cost varies with the tools a
    # turn calls and with whether context editing runs, which an average understates.
    assert STEPS_PER_MODEL_CALL >= MEASURED_STEPS_PER_CALL


def test_one_run_cannot_spend_the_whole_afternoon_generating_pictures(monkeypatch):
    """The PNG count was the only image budget, and it does not measure what makes a run feel stuck.

    Measured across 64 real generations: a 512x512 sprite takes about 31s, a 1536x512 animation
    sheet about 58s, worst case 166s - so the same "14 images" is four minutes of sprites or a
    quarter of an hour of sheets, doubled again whenever an animation re-rolls. The agent is blocked
    on every one of them, writing nothing.

    Running out is not a failure. The art plan is Canvas-first with raster as an improvement, so the
    tool says so and the build carries on drawing shapes.
    """
    from game_studio import agent_tools

    monkeypatch.setattr(agent_tools, "_IMAGE_SECONDS", {})
    monkeypatch.setattr(agent_tools, "IMAGE_TIME_BUDGET", 100)

    assert agent_tools._out_of_image_time("/ws") == "", "a fresh run may generate"
    agent_tools._spend_image_time("/ws", 99)
    assert agent_tools._out_of_image_time("/ws") == "", "and may still afford one more"
    agent_tools._spend_image_time("/ws", 2)

    spent = agent_tools._out_of_image_time("/ws")
    assert "이미지 생성 시간" in spent and "Canvas" in spent, "say what to do instead"
    # Per workspace, not per process: one dashboard serves many runs, and a long afternoon must not
    # make the next run start already over budget.
    assert agent_tools._out_of_image_time("/other-run") == ""


def test_a_failed_generation_is_charged_for_the_time_it_took(monkeypatch):
    """A generation that timed out cost the run its wall clock just as surely as one that returned a
    picture - more, in fact. Charging only successes would let a stuck ComfyUI burn an unlimited
    amount of a run while the budget reads zero."""
    import re

    from game_studio import agent_tools

    body = pathlib.Path(agent_tools.__file__).read_text(encoding="utf-8")
    render = body[body.index("def _render_png("):body.index("def _generate_animation_frames(")]
    charge = re.search(r"finally:\n(.*\n)*?\s+_spend_image_time\(workspace", render)
    assert charge, "_render_png must charge its time from a finally block, not on the way out"


def test_a_revision_gets_its_own_image_clock(monkeypatch):
    """The image budget is keyed by workspace so one dashboard can serve many runs at once. A
    revision works in the SAME folder as the game it is reworking, so it inherited that game's spend
    - and a person asking for a change to a game that had used nine of its ten minutes would be told
    to draw the rest in Canvas shapes before it generated anything.

    The ceiling is per RUN, not per folder.
    """
    from game_studio import agent_tools

    monkeypatch.setattr(agent_tools, "_IMAGE_SECONDS", {})
    monkeypatch.setattr(agent_tools, "IMAGE_TIME_BUDGET", 100)

    agent_tools._spend_image_time("/games/블록-강하_html5_abc", 95)
    assert agent_tools._image_time_left("/games/블록-강하_html5_abc") == 5

    agent_tools.reset_image_time("/games/블록-강하_html5_abc")
    assert agent_tools._image_time_left("/games/블록-강하_html5_abc") == 100
    assert agent_tools._out_of_image_time("/games/블록-강하_html5_abc") == ""

    # Only that workspace. Another run in flight keeps whatever it has spent.
    agent_tools._spend_image_time("/games/other", 90)
    agent_tools.reset_image_time("/games/블록-강하_html5_abc")
    assert agent_tools._image_time_left("/games/other") == 10
    # And resetting a workspace that never spent anything is not an error.
    agent_tools.reset_image_time("/games/never-seen")
