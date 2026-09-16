import json

from fastapi.testclient import TestClient

from game_studio import server
from game_studio.server import app


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
