"""The eval harness itself, checked without spending anything.

Every other test in this suite stubs the model out, which is right for the code and is exactly why
the prompts went unmeasured: every prompt change in this project was validated by running it three
times and reading the titles. That found the "별똥별 사냥꾼 / 별똥별 사냥꾼 / 별똥별 낚시꾼" collapse
and would never have caught it coming back.

These pin the scoring, not the prompts - the harness has to tell a good generation from a regressed
one before its numbers mean anything. The generations here are fixtures; running the harness for
real costs money and is a separate, deliberate act.
"""

import json

import pytest
from pydantic import ValidationError

from game_studio import evaluate as ev
from game_studio.models import GameConcept, ImplementationPlan


def concept(title="블록 강하", refs=("Tetris — 테트로미노 낙하·회전",), loop=("낙하", "회전", "제거")):
    return GameConcept(title=title, elevator_pitch="p", player_goal="g", controls=["a", "b"],
                       core_loop=list(loop), difficulty_curve="d", visual_direction="v",
                       reference_games=list(refs))


def plan(genre="퍼즐", mechanics=3, tests=3):
    return ImplementationPlan(genre=genre, mechanics=[f"m{i}" for i in range(mechanics)],
                              win_condition="w", loss_condition="l",
                              state_transitions=["1", "2", "3"],
                              acceptance_tests=[f"t{i}" for i in range(tests)])


CLONE = {"id": "clone-tetris", "genre": "퍼즐", "brief": "테트리스랑 완전히 똑같은 게임을 만들어줘",
         "seed": "s1", "expect": {"genre": "퍼즐", "references": ["테트리스", "Tetris"]}}
AUTO = {"id": "auto-1", "genre": "auto", "brief": "", "seed": "eval-auto-0001",
        "expect": {"assigned_genre": True, "diverse_group": "auto"}}


def test_the_brief_is_written_exactly_as_the_dashboard_writes_it():
    """Measuring a prompt the pipeline never sends measures nothing."""
    explicit = ev.build_brief(CLONE)
    assert "Requested genre: 퍼즐" in explicit
    assert "Player brief: 테트리스랑 완전히 똑같은 게임을 만들어줘" in explicit
    assert "Run seed: s1" in explicit

    auto = ev.build_brief(AUTO)
    assert "Requested genre: 자동 기획" in auto
    # The exact sentence that player_requested() treats as "no request" - get this wrong and the
    # harness silently measures the explicit-request path instead.
    assert "사용자 경험 없이 독자적으로 기획하세요." in auto
    from game_studio.agents import player_requested
    assert player_requested(auto) == ""
    assert player_requested(explicit) == "테트리스랑 완전히 똑같은 게임을 만들어줘"


def test_the_scoring_separates_a_good_generation_from_a_regressed_one():
    """The failure this whole harness exists to catch: asked for Tetris, given a different game."""
    good, _ = ev._check(CLONE, ev.build_brief(CLONE), concept(), plan("퍼즐"))
    assert good["genre_kept"] and good["clone_named"]

    regressed, _ = ev._check(CLONE, ev.build_brief(CLONE),
                             concept("별똥별 사냥꾼", ["Missile Command — 요격"],
                                     ["출현", "조준", "발사"]),
                             plan("슈팅"))
    assert not regressed["genre_kept"], "the wrong genre has to score as wrong"
    assert not regressed["clone_named"], "and a clone that names something else has to fail"


def test_a_faithful_clone_is_verified_structurally_not_by_a_judge():
    """"Is this really Tetris" would need judgement, and a judge adds its own variance to a
    measurement whose whole purpose is separating a prompt regression from noise. The prompt
    requires a faithful clone to name what it reproduces, so the check is whether that name
    arrived - a string, not an opinion."""
    for refs in (["Tetris — 낙하·회전"], ["테트리스에서 줄 제거를 차용"], ["TETRIS"]):
        checks, _ = ev._check(CLONE, ev.build_brief(CLONE), concept(refs=refs), plan("퍼즐"))
        assert checks["clone_named"], refs
    checks, _ = ev._check(CLONE, ev.build_brief(CLONE), concept(refs=["2048"]), plan("퍼즐"))
    assert not checks["clone_named"]


def test_an_auto_case_is_judged_against_its_own_assignment(monkeypatch):
    """An auto brief has no fixed right answer - the genre comes from the run seed. Writing one
    into the eval set would make the harness disagree with the pipeline whenever either changed."""
    brief = ev.build_brief(AUTO)
    from game_studio.agents import resolve_auto_genre

    assigned = resolve_auto_genre(brief)
    assert assigned, "an auto brief must be assigned a genre at all"

    kept, expected = ev._check(AUTO, brief, concept(), plan(assigned))
    assert expected == assigned
    assert kept["genre_kept"] and kept["genre_assigned"]

    drifted, _ = ev._check(AUTO, brief, concept(), plan("레이싱" if assigned != "레이싱" else "퍼즐"))
    assert not drifted["genre_kept"]


def test_the_assigned_genre_has_to_be_one_the_reference_table_can_anchor():
    """Auto mode getting no exemplars is what collapsed four runs into the same game. A genre with
    no row in the table is the same failure wearing a different name."""
    brief = ev.build_brief(AUTO)
    checks, _ = ev._check(AUTO, brief, concept(), plan())
    assert checks["genre_assigned"] is True

    from game_studio import agents
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(agents, "resolve_auto_genre", lambda _brief: "닌텐도 독점")
    monkeypatch.setattr(ev, "resolve_auto_genre", lambda _brief: "닌텐도 독점")
    try:
        unanchored, _ = ev._check(AUTO, brief, concept(), plan("닌텐도 독점"))
    finally:
        monkeypatch.undo()
    assert not unanchored["genre_assigned"], "a genre with no exemplars is not an assignment"


def test_an_oversized_contract_cannot_be_produced_and_is_flagged_if_it_is():
    """The code agent has a finite call budget, and a contract at the old ceiling was the shape
    that spent it without finishing: the first measured Tetris plan came back at eight mechanics
    and eight acceptance tests, and there is no version of that build that fits in twenty calls.

    So the limit is enforced by the schema now rather than asked for in the prompt - a ceiling the
    model may quietly exceed is not a budget. The metric stays because the schema is not the only
    way a contract arrives (a restored manifest, a hand-edited plan), and because it is what tells
    us whether the prompt is straining against the limit or living comfortably inside it.
    """
    from game_studio.models import CONTRACT_MAX_ITEMS

    ok, _ = ev._check(CLONE, ev.build_brief(CLONE), concept(),
                      plan("퍼즐", mechanics=CONTRACT_MAX_ITEMS, tests=CONTRACT_MAX_ITEMS))
    assert ok["contract_sized"], "the limit itself has to pass"

    with pytest.raises(ValidationError):
        plan("퍼즐", mechanics=CONTRACT_MAX_ITEMS + 1, tests=3)
    with pytest.raises(ValidationError):
        plan("퍼즐", mechanics=3, tests=CONTRACT_MAX_ITEMS + 1)

    # And if one reaches the harness anyway, it still scores as oversized rather than silently.
    oversized = plan("퍼즐").model_copy(
        update={"mechanics": [f"m{i}" for i in range(CONTRACT_MAX_ITEMS + 2)]})
    big, _ = ev._check(CLONE, ev.build_brief(CLONE), concept(), oversized)
    assert not big["contract_sized"]


def test_diversity_looks_at_the_loop_not_only_the_title():
    """"별똥별 사냥꾼" and "별똥별 낚시꾼" are different strings and the same game."""
    cases = [{**AUTO, "id": f"auto-{n}"} for n in range(1, 4)]
    same = [ev.CaseResult(id=f"auto-{n}", brief="", title=t, genre="슈팅",
                          core_loop=["별똥별이 쏟아진다", "조준해서 맞힌다", "콤보가 오른다"])
            for n, t in enumerate(["별똥별 사냥꾼", "별똥별 낚시꾼", "별똥별 사냥꾼"], 1)]
    report = ev.diversity(same, cases)["auto"]
    assert report["distinct_titles"] == 2, "titles alone look almost fine"
    assert report["mean_loop_overlap"] > 0.9, "the loops give it away"
    assert report["distinct_genres"] == 1

    varied = [
        ev.CaseResult(id="auto-1", brief="", title="블록 강하", genre="퍼즐",
                      core_loop=["조각이 떨어진다", "회전시켜 쌓는다", "줄이 사라진다"]),
        ev.CaseResult(id="auto-2", brief="", title="불꽃 생존자", genre="액션 생존",
                      core_loop=["적이 몰려온다", "자동으로 공격한다", "강화 카드를 고른다"]),
        ev.CaseResult(id="auto-3", brief="", title="랩 기록", genre="레이싱",
                      core_loop=["코너를 감속한다", "추월한다", "기록을 단축한다"]),
    ]
    better = ev.diversity(varied, cases)["auto"]
    assert better["distinct_titles"] == 3 and better["distinct_genres"] == 3
    assert better["mean_loop_overlap"] < report["mean_loop_overlap"]


def test_a_failed_generation_is_recorded_rather_than_crashing_the_run(monkeypatch):
    """One bad case must not throw away the other nine, and must not quietly count as a pass."""
    monkeypatch.setattr(ev, "create_concept",
                        lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("Bedrock 거부")))
    result = ev.evaluate_case(CLONE, None)
    assert not result.ok and "Bedrock 거부" in result.error
    assert result.score == 0.0

    report = ev.run([CLONE], None)
    assert report["failed"] == [{"id": "clone-tetris", "error": result.error}]
    assert report["generated"] == 0
    assert "지표에서 제외됨" in ev.render(report, None)


def test_the_shipped_eval_set_is_coherent():
    """A set that disagrees with the pipeline measures the set, not the pipeline."""
    from game_studio.agents import genre_references

    cases = json.loads(ev.DEFAULT_SET.read_text(encoding="utf-8"))["cases"]
    assert len({case["id"] for case in cases}) == len(cases), "ids have to be unique"
    assert len({case["seed"] for case in cases}) == len(cases), "and so do seeds"

    for case in cases:
        expect = case.get("expect") or {}
        if genre := expect.get("genre"):
            assert genre_references(genre), f"{case['id']}: 표에 없는 장르는 앵커할 수 없습니다"
        if case.get("genre") != "auto":
            assert case.get("brief"), f"{case['id']}: 장르 지정 케이스에는 브리프가 필요합니다"
        elif case.get("brief"):
            # 자동 기획 is the default dropdown value, so a player who types a request and never
            # touches it lands here. The brief decides the genre, so the case has to say which.
            assert genre, f"{case['id']}: 브리프가 있는 auto 케이스는 기대 장르를 적어야 합니다"
            assert not expect.get("assigned_genre"), \
                f"{case['id']}: 브리프가 있으면 시드 배정이 아니라 브리프가 장르를 정합니다"
        else:
            assert expect.get("assigned_genre"), f"{case['id']}: 배정 여부를 확인해야 합니다"

    free = [c for c in cases if c.get("genre") == "auto" and not c.get("brief")]
    assert len(free) >= 3, "다양성은 표본이 셋은 있어야 의미가 있습니다"


def test_the_report_shows_movement_against_a_baseline():
    """A score with nothing to compare to cannot show a regression, which is the point."""
    report = {"cases": 1, "generated": 1, "failed": [], "score": 0.75, "seconds": 1.0,
              "metrics": {"genre_kept": 1.0, "clone_named": 0.5},
              "diversity": {}, "results": []}
    baseline = {"metrics": {"genre_kept": 1.0, "clone_named": 1.0}}
    text = ev.render(report, baseline)
    assert "-50.0%" in text, "a drop has to be visible"
    assert "—" in text, "and an unchanged metric has to read as unchanged"


def test_an_acceptance_test_nobody_can_run_is_scored_as_one():
    """A measured contract asked six times for something to be confirmed "프레임 단위 로그로" or
    "좌표 로그로". Nothing in this pipeline reads a frame log, so those get scored by a model
    reading source and come back disputed however good the game is - and the run then spends repair
    budget on a disagreement that has no resolution."""
    watchable = plan("플랫포머").model_copy(update={"acceptance_tests": [
        "시작하면 바로 좌우로 움직일 수 있다",
        "적을 밟으면 적이 사라지고 점수가 오른다",
        "구덩이에 빠지면 게임 오버 화면이 뜨고 재시작된다",
    ]})
    ok, _ = ev._check(CLONE, ev.build_brief(CLONE), concept(), watchable)
    assert ok["tests_observable"]

    instrumented = plan("플랫포머").model_copy(update={"acceptance_tests": [
        "시작하면 바로 좌우로 움직일 수 있다",
        "방향키를 누르면 10프레임 이내에 최고 속도에 도달함을 확인한다",
        "구덩이에 빠지면 게임 오버 화면이 뜬다",
    ]})
    bad, _ = ev._check(CLONE, ev.build_brief(CLONE), concept(), instrumented)
    assert not bad["tests_observable"], "one unrunnable test is enough to fail the contract"
