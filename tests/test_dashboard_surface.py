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
