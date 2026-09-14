from game_studio.agents import static_qa
from game_studio.fallback_game import build_neon_drift_html


def test_fallback_game_is_shippable():
    html = build_neon_drift_html("Test Run")
    report = static_qa(html)
    assert report.status == "pass", report.findings
    assert "Test Run" in html
    assert "pointerdown" in html


def test_qa_blocks_network_dependent_game():
    report = static_qa('<canvas></canvas><script>fetch("https://example.com")</script>')
    assert report.status == "repair"
    assert any("external dependency" in item.lower() for item in report.findings)
