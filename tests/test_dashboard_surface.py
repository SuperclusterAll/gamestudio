import json

import pytest
from fastapi.testclient import TestClient

from game_studio import art_memory, server
from game_studio.server import app


@pytest.fixture(autouse=True)
def isolated_runs(monkeypatch, tmp_path_factory):
    """StudioService.runs is a module-level singleton that outlives a test.

    A test that starts or revises a run leaves it there, pointed at a tmp_path that the next test
    has already replaced - and the next test then resolves that id to the previous test's folder
    and fails for a reason that has nothing to do with what it is checking. restore() only drops
    the entries it restored itself, so a run started through the API survives it.
    """
    server.service.runs.clear()
    server._RUN_FOLDERS.clear()
    # TestClient(app) runs the lifespan, and the lifespan prunes checkpoints - against the real
    # database this process opened at import, not a temporary one. A test must not delete a
    # developer's paused approvals, and reading their 246MB file to decide not to took 90 seconds
    # per test. prune_checkpoints has its own tests, on its own database.
    monkeypatch.setattr(server, "prune_checkpoints", lambda *_a, **_kw: (0, 0.0))
    # And the art memory now defaults to the project's own data directory, so a test that judges a
    # sprite would write into the developer's real store. Same reason, same fix.
    monkeypatch.setattr(art_memory, "ART_MEMORY_DIR",
                        str(tmp_path_factory.mktemp("art-memory")))
    yield
    server.service.runs.clear()
    server._RUN_FOLDERS.clear()


def test_dashboard_serves_ui_assets():
    with TestClient(app) as client:
        page = client.get("/")
        script = client.get("/static/app.js")
    assert page.status_code == 200
    assert "Control Room" in page.text
    assert script.status_code == 200
    assert "WebSocket" in script.text


def test_completed_game_survives_dashboard_restart(tmp_path, monkeypatch):
    run_id = "3f0d505d3038"
    game = tmp_path / run_id / "index.html"
    game.parent.mkdir()
    game.write_text("<!doctype html><title>Saved game</title>", encoding="utf-8")
    monkeypatch.setattr(server, "GAME_OUTPUT_ROOT", tmp_path)

    with TestClient(app) as client:
        response = client.get(f"/games/{run_id}")

    assert response.status_code == 200
    assert "Saved game" in response.text


def test_missing_credentials_never_silently_runs_offline(monkeypatch):
    monkeypatch.setattr(server,'bedrock_credentials_configured',lambda:False)
    with TestClient(app) as client:
        response=client.post('/api/runs',json={'brief':'새로운 로그라이크'})
    assert response.status_code==503


def test_game_assets_resolve_and_cannot_escape_run(tmp_path, monkeypatch):
    root=tmp_path/'test-run'; root.mkdir(); (root/'index.html').write_text('<html></html>')
    (root/'game.js').write_text('window.gameReady=true')
    (tmp_path/'secret.js').write_text('secret')
    monkeypatch.setattr(server,'GAME_OUTPUT_ROOT',tmp_path)
    with TestClient(app) as client:
        assert client.get('/games/test-run').url.path=='/games/test-run/'
        assert client.get('/games/test-run/game.js').status_code==200
        assert client.get('/games/test-run/..%2Fsecret.js').status_code==404


def _manifest(root, run_id, **extra):
    folder = root / run_id
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "production-manifest.json").write_text(
        json.dumps({"engine": "html5", "concept": {"title": "블록 강하"},
                    "implementation_plan": {"genre": "퍼즐"}, **extra}, ensure_ascii=False),
        encoding="utf-8")
    return folder


def test_whether_a_game_was_worth_continuing_is_recorded_and_survives_a_restart(tmp_path, monkeypatch):
    """The service's headline metric, and the one judgement no check in this pipeline can make.
    Static QA proves a game runs and the design review proves it matches its contract; neither says
    whether anyone wants to build it. It was defined as a target and then never recorded, so the
    number the whole studio exists to move was the only one running on memory.

    Written into the manifest for the same reason the run list is rebuilt from manifests: an
    in-memory dict lasts as long as the process, and an adoption rate that resets on restart
    measures nothing.
    """
    monkeypatch.setattr(server, "GAME_OUTPUT_ROOT", tmp_path)
    folder = _manifest(tmp_path, "3f0d505d3038")

    with TestClient(app) as client:
        assert client.get("/api/adoption").json() == {
            "finished": 1, "decided": 0, "adopted": 0, "undecided": 1, "rate": None}, \
            "nothing judged yet is not a zero percent adoption rate"

        marked = client.post("/api/runs/3f0d505d3038/adopt",
                             json={"adopted": True, "note": "레벨 디자인만 손보면 됨"})
        assert marked.status_code == 200 and marked.json()["adopted"] is True
        assert marked.json()["decided_at"], "a decision without a date cannot be a trend"

        stats = client.get("/api/adoption").json()
        assert stats == {"finished": 1, "decided": 1, "adopted": 1, "undecided": 0, "rate": 1.0}

    # On disk, beside the game, so it comes back with the folder rather than with this process.
    manifest = json.loads((folder / "production-manifest.json").read_text(encoding="utf-8"))
    assert manifest["adoption"]["adopted"] is True
    assert manifest["adoption"]["note"] == "레벨 디자인만 손보면 됨"
    assert manifest["concept"]["title"] == "블록 강하", "the rest of the manifest is untouched"

    # A restarted dashboard offers the decision as already made rather than asking again.
    restored = server._restore_finished_runs()["3f0d505d3038"]
    assert restored.state["adoption"]["adopted"] is True

    # And it can be changed: a game adopted in haste is not adopted forever.
    with TestClient(app) as client:
        assert client.post("/api/runs/3f0d505d3038/adopt", json={"adopted": False}).status_code == 200
        assert client.get("/api/adoption").json()["rate"] == 0.0


def test_the_adoption_rate_counts_only_games_somebody_judged(tmp_path, monkeypatch):
    """A prototype that runs and is not worth continuing is a successful run of this pipeline and a
    failed idea. Counting undecided runs as rejections would make the rate fall every time the
    studio got busy, which is the opposite of what it is for."""
    monkeypatch.setattr(server, "GAME_OUTPUT_ROOT", tmp_path)
    _manifest(tmp_path, "aaaaaaaaaaaa", adoption={"adopted": True, "decided_at": "t"})
    _manifest(tmp_path, "bbbbbbbbbbbb", adoption={"adopted": False, "decided_at": "t"})
    _manifest(tmp_path, "cccccccccccc")
    (tmp_path / "ember-vault").mkdir()  # a hand-named folder is not a run

    with TestClient(app) as client:
        stats = client.get("/api/adoption").json()
    assert stats == {"finished": 3, "decided": 2, "adopted": 1, "undecided": 1, "rate": 0.5}


def test_adoption_cannot_be_recorded_against_anything_but_a_delivered_game(tmp_path, monkeypatch):
    """This endpoint writes to a file path built from the request, so the id has to be a plain
    token and the file has to be a manifest that already exists - a run that produced nothing has
    nothing to adopt."""
    monkeypatch.setattr(server, "GAME_OUTPUT_ROOT", tmp_path)
    (tmp_path / "3f0d505d3038").mkdir()  # a run folder with no manifest

    with TestClient(app) as client:
        assert client.post("/api/runs/3f0d505d3038/adopt", json={"adopted": True}).status_code == 404
        assert client.post("/api/runs/..%2F..%2Fetc/adopt", json={"adopted": True}).status_code == 404
        assert client.post("/api/runs/nope/adopt", json={"adopted": True}).status_code == 404
        # adopted is the whole payload and it is required: a blank mark is not a judgement.
        assert client.post("/api/runs/3f0d505d3038/adopt", json={}).status_code == 422


def test_a_finished_game_can_be_reworked_from_the_players_own_notes(tmp_path, monkeypatch):
    """The gap this closes: every check in this pipeline runs before anyone has played the result.
    Static QA proves the game runs, the design review proves it matches its contract, and neither
    can notice that the jump feels heavy. That feedback only exists after the game ships, and the
    only way back in was a new run from a brief - which produces a different game.
    """
    monkeypatch.setattr(server, "GAME_OUTPUT_ROOT", tmp_path)
    folder = _manifest(tmp_path, "3f0d505d3038", engine="html5",
                       implementation_plan={"genre": "퍼즐", "mechanics": ["떨어진다"]},
                       art={"palette": {"a": "#fff"}, "image_prompt": "p"},
                       code_model_id="global.anthropic.claude-sonnet-4-6")
    (folder / "index.html").write_text("<html>shipped</html>", encoding="utf-8")

    driven = []
    monkeypatch.setattr(server.StudioService, "_drive",
                        lambda self, run, payload: driven.append((run, payload)))
    monkeypatch.setattr(server, "bedrock_credentials_configured", lambda: True)

    with TestClient(app) as client:
        started = client.post("/api/runs/3f0d505d3038/revise",
                              json={"request": "점프가 너무 무겁습니다. 가볍게 해주세요."})
    assert started.status_code == 202

    # It has to run, and the payload is the whole point: the plan it already had, not a new one.
    assert len(driven) == 1
    run, payload = driven[0]
    assert payload["revision_request"] == "점프가 너무 무겁습니다. 가볍게 해주세요."
    assert payload["implementation_plan"]["mechanics"] == ["떨어진다"], "the contract carries over"
    assert payload["concept"]["title"] == "블록 강하"
    assert payload["approval"]["decision"] == "approved", "the gate was passed before the game shipped"
    # stage "art" is the supervisor's name for "art direction is settled", and its only edge out is
    # the code agent - so this is how the run enters at the build with no re-planning ahead of it.
    assert payload["stage"] == "art"
    assert payload["workspace_dir"] == str(folder), "it improves this game, in this folder"

    # The id stays the folder's, so /launch, /adopt and /games keep resolving - but the checkpointer
    # gets a fresh thread, or invoking it would resume the completed run instead of starting this.
    assert run.id == "3f0d505d3038"
    assert run.config["configurable"]["thread_id"].startswith("3f0d505d3038-rev-")
    assert run.config["configurable"]["thread_id"] != "3f0d505d3038"

    # The Canvas tools read and write draft.html; without the shipped game staged there the agent
    # would open a revision by finding no draft and writing a new game from the contract.
    assert (folder / "draft.html").read_text(encoding="utf-8") == "<html>shipped</html>"

    # And asking for improvements is continuing to develop it, so the adoption metric records
    # itself instead of through a button that only ever wrote the answer down.
    manifest = json.loads((folder / "production-manifest.json").read_text(encoding="utf-8"))
    assert manifest["adoption"]["adopted"] is True
    assert "점프가 너무 무겁" in manifest["adoption"]["note"]


def test_a_rework_needs_a_delivered_game_a_plan_and_something_to_say(tmp_path, monkeypatch):
    """This starts a real run against a folder named by the request, so every one of those has to
    be checked before anything is spent."""
    monkeypatch.setattr(server, "GAME_OUTPUT_ROOT", tmp_path)
    monkeypatch.setattr(server, "bedrock_credentials_configured", lambda: True)
    monkeypatch.setattr(server.StudioService, "_drive", lambda self, run, payload: None)
    (tmp_path / "3f0d505d3038").mkdir()                       # 산출물 없음
    _manifest(tmp_path, "aaaaaaaaaaaa", implementation_plan={})  # 계약 기록 없음

    with TestClient(app) as client:
        def revise(run_id, request="점프를 가볍게"):
            return client.post(f"/api/runs/{run_id}/revise", json={"request": request})

        assert revise("3f0d505d3038").status_code == 404, "no manifest, nothing to rework"
        assert revise("aaaaaaaaaaaa").status_code == 409, "a manifest with no contract is not one"
        assert revise("..%2F..%2Fetc").status_code == 404
        assert revise("nope").status_code == 404
        # An empty note is not feedback, and there is no default to fall back on.
        assert revise("aaaaaaaaaaaa", "").status_code == 422


def test_a_games_folder_says_what_the_game_is_and_which_engine_built_it(tmp_path, monkeypatch):
    """Folders were named by run id alone, so a directory of games was a directory of hex strings -
    you could not tell a Godot project from a Canvas page, or one game from another, without
    opening each manifest. The id stays, and stays last: it is what every lookup resolves by and
    the only part a request supplies, so keeping it a fixed token at a known position lets a folder
    be *found* by its id rather than built from a title.
    """
    from game_studio.models import workspace_name

    assert workspace_name("블록 강하", "godot", "74763df4098f") == "블록-강하_godot_74763df4098f"
    assert workspace_name("Neon Drift", "html5", "abc123def456") == "neon-drift_html5_abc123def456"
    # A title that slugs away to nothing still leaves a usable name, and a very long one is cut.
    assert workspace_name("!!!", "html5", "abc123def456") == "game_html5_abc123def456"
    long_title = workspace_name("가" * 80, "godot", "abc123def456")
    assert long_title.endswith("_godot_abc123def456") and len(long_title.split("_")[0]) == 40

    # And the id comes back out of every shape, including folders from before the rename.
    monkeypatch.setattr(server, "GAME_OUTPUT_ROOT", tmp_path)
    assert server.run_id_of("블록-강하_godot_74763df4098f") == "74763df4098f"
    assert server.run_id_of("74763df4098f") == "74763df4098f", "an old bare-id folder still resolves"
    assert server.run_id_of("ember-vault") == "", "a hand-named folder is not a run"

    named = tmp_path / "블록-강하_godot_74763df4098f"
    named.mkdir()
    assert server.run_folder("74763df4098f") == named.resolve()
    # Nothing matching resolves to the bare id, so the caller's own is_file() check still decides.
    assert server.run_folder("ffffffffffff") == (tmp_path / "ffffffffffff").resolve()
    # A token that could leave the output root never reaches the filesystem at all.
    for hostile in ("../etc", "a/b", "..", ""):
        assert not server.run_folder(hostile).is_relative_to(tmp_path) or \
            server.run_folder(hostile).name == "__invalid__", hostile


def test_every_lookup_finds_a_renamed_folder(tmp_path, monkeypatch):
    """The folder name changed; the id in every URL did not. Launch, adopt, revise, play and the
    asset route all address a run by id, so each has to resolve the new name or the rename would
    have unpublished every game it touched."""
    monkeypatch.setattr(server, "GAME_OUTPUT_ROOT", tmp_path)
    monkeypatch.setattr(server, "bedrock_credentials_configured", lambda: True)
    monkeypatch.setattr(server.StudioService, "_drive", lambda self, run, payload: None)
    folder = _manifest(tmp_path, "블록-강하_godot_3f0d505d3038", engine="godot")
    (folder / "index.html").write_text("<html>블록 강하</html>", encoding="utf-8")
    (folder / "game.js").write_text("window.ready=true", encoding="utf-8")

    with TestClient(app) as client:
        assert client.get("/games/3f0d505d3038/").status_code == 200
        assert client.get("/games/3f0d505d3038/game.js").status_code == 200
        assert client.post("/api/runs/3f0d505d3038/adopt", json={"adopted": True}).status_code == 200
        assert client.post("/api/runs/3f0d505d3038/revise",
                           json={"request": "점프를 가볍게"}).status_code == 202
        # And the restored run is keyed by the id, not by the folder name it now has.
        assert client.get("/api/adoption").json()["adopted"] == 1

    assert "3f0d505d3038" in server._restore_finished_runs()


def test_a_dollar_figure_always_says_what_it_is_measured_against(monkeypatch):
    """The two billing shapes are indistinguishable from inside the API - the same call on the same
    model returns the same token counts whether the account is metered or has bought capacity up
    front - so the basis has to be stated. Printing a figure with no basis attached is how
    "런당 $2~4" ended up in a service document written for an account that is not billed per token.
    """
    from game_studio import agents

    monkeypatch.setenv("BEDROCK_PRICING_MODE", "ondemand")
    metered = agents.pricing_basis()
    assert metered["mode"] == "ondemand" and metered["billed_per_token"] is True
    assert "종량제" in metered["label"] and "청구서가 아니며" in metered["note"]

    monkeypatch.setenv("BEDROCK_PRICING_MODE", "provisioned")
    committed = agents.pricing_basis()
    assert committed["billed_per_token"] is False, "a committed account is not billed per token"
    assert "정액제" in committed["label"]
    assert "환산" in committed["note"] and "호출 수" in committed["note"], \
        "it has to say what the real constraint is instead"

    # Anything unrecognised falls back to the conservative reading rather than guessing.
    monkeypatch.setenv("BEDROCK_PRICING_MODE", "무엇이든")
    assert agents.pricing_mode() == agents.DEFAULT_PRICING_MODE
    monkeypatch.delenv("BEDROCK_PRICING_MODE")
    assert agents.pricing_mode() == "ondemand"

    # And the dashboard is told, alongside the rates the figure was computed from.
    with TestClient(app) as client:
        pricing = client.get("/api/model-status").json()["pricing"]
    assert pricing["mode"] in agents.PRICING_MODES
    assert pricing["reference_model"] and pricing["input_per_mtok"] == 3.00
    assert pricing["output_per_mtok"] == 15.00


def test_generated_images_can_be_judged_and_only_within_their_own_run(tmp_path, monkeypatch):
    """The label the whole art memory turns on. An automatic verdict sees only geometry - it knows
    a sprite came back 109px wide and unusable, and it cannot tell a good mushroom from a bad one.

    The id addresses a row in a store shared by every run, and it arrives from the browser, so a
    verdict may only ever be filed against the run in the URL.
    """
    monkeypatch.setattr(server, "GAME_OUTPUT_ROOT", tmp_path)
    folder = _manifest(tmp_path, "블록-강하_html5_3f0d505d3038")
    assets = folder / "assets"
    assets.mkdir()
    (assets / "enemy.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    art_memory.remember(name="enemy.png", prompt="둥근 적, 굵은 외곽선",
                        role="enemy", genre="퍼즐", run_id=folder.name,
                        entry={"kind": "sprite", "width": 221, "height": 224,
                               "removed_share": 0.47})

    with TestClient(app) as client:
        listed = client.get("/api/runs/3f0d505d3038/sprites").json()
        assert [s["name"] for s in listed["sprites"]] == ["enemy.png"]
        assert listed["sprites"][0]["url"] == "/games/3f0d505d3038/assets/enemy.png"
        assert listed["sprites"][0]["prompt"] == "둥근 적, 굵은 외곽선"
        # The image itself has to load, or there is nothing to judge. A Godot run has no
        # index.html, which is why that is no longer what gates the assets route.
        assert client.get("/games/3f0d505d3038/assets/enemy.png").status_code == 200

        mine = f"{folder.name}:enemy.png"
        assert client.post("/api/runs/3f0d505d3038/sprites/verdict",
                           json={"sprite_id": mine, "label": "good"}).status_code == 200
        # A verdict against another run's image is refused however real that image is.
        assert client.post("/api/runs/3f0d505d3038/sprites/verdict",
                           json={"sprite_id": "다른-런_html5_ffffffffffff:enemy.png",
                                 "label": "good"}).status_code == 404
        assert client.post("/api/runs/3f0d505d3038/sprites/verdict",
                           json={"sprite_id": mine, "label": "훌륭"}).status_code == 422

    # And the mark is what the next game's art planning will actually see.
    recalled = art_memory.recall(visual_direction="굵은 외곽선", genre="퍼즐")
    assert recalled[0]["verdict"] == "good" and recalled[0]["verdict_by"] == "human"
