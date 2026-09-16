"""What the pipeline stopped paying for, and the memory it gained.

Every item here was measured before it was changed. A measured run spent 85% of its tokens on
input, re-sending the same system prompt and the same ~2,700 tokens of tool definitions on each of
the code agent's twenty calls; the build loop spent a model call asking for a verdict that was
already knowable at write time; the code agent's whole transcript - whole games included, as
write_game_file arguments - was written into every checkpoint and read by nothing; and a pending
human approval, which the graph is designed to hold durably, died with the process.
"""

import json

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
    monkeypatch.setattr(gm, "run_deep_director", lambda *a, **kw: "plan")
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
