"""How hard verification is allowed to push back, and what it costs to run.

The pipeline was ending runs with dozens of open findings against a game that was actually
playable, and the audit that produced them was the slowest single call in the build. These pin the
policy that fixed that: judge the approved contract and nothing else, on the fast model, and
publish rather than grind when the disagreement is small.
"""

import pytest
from langchain_core.messages import AIMessageChunk
from pydantic import ValidationError

from game_studio.models import (
    DesignReview,
    GameConcept,
    ImplementationPlan,
    RequirementCheck,
)

PLAYABLE = ('<html><canvas>score restart keydown KeyW KeyA KeyS KeyD '
            'requestAnimationFrame</canvas></html>')


def concept() -> GameConcept:
    return GameConcept(title='테스트 레이서', elevator_pitch='짧은 랩 경쟁', player_goal='기록 단축',
                       controls=['방향키', 'WASD'], core_loop=['가속', '코너', '완주'],
                       difficulty_curve='랩마다 상승', visual_direction='네온')


def contract() -> ImplementationPlan:
    return ImplementationPlan(genre='레이싱', mechanics=['가속', '드리프트', '충돌'],
                              win_condition='완주', loss_condition='시간 초과',
                              state_transitions=['시작', '주행', '결과'],
                              acceptance_tests=['랩 기록', '부스트', '리트라이'])


def requirements(plan: ImplementationPlan) -> list[str]:
    return plan.mechanics + [plan.win_condition, plan.loss_condition] + plan.acceptance_tests


def test_verification_runs_on_the_fast_judgement_model(monkeypatch):
    """Auditing reads a finished game and answers with a verdict, so it does not need the authoring
    model - and it sits directly between a finished build and a shipped game, which is where
    latency is felt. The audit and the decision about the audit have to agree on the model."""
    import game_studio.graph as gm
    from game_studio.agents import DEFAULT_QA_MODEL_ID, price_per_mtok, qa_model_id
    from game_studio.server import BEDROCK_MODELS

    monkeypatch.delenv('BEDROCK_QA_MODEL_ID', raising=False)
    assert 'haiku' in DEFAULT_QA_MODEL_ID
    assert DEFAULT_QA_MODEL_ID in BEDROCK_MODELS, 'the QA default has to be a selectable model'
    # One unpriced call flips the whole run's cost estimate to "not priced", so moving a stage onto
    # a new model without pricing it silently breaks the dashboard's spend report.
    assert all(price_per_mtok(model) is not None for model in BEDROCK_MODELS), \
        'every offered model must be priced'
    # The run's own planning model never overrides the judgement default; only the env var does.
    assert qa_model_id('global.anthropic.claude-sonnet-4-6') == DEFAULT_QA_MODEL_ID
    monkeypatch.setenv('BEDROCK_QA_MODEL_ID', 'global.amazon.nova-2-lite-v1:0')
    assert qa_model_id(None) == 'global.amazon.nova-2-lite-v1:0'
    monkeypatch.delenv('BEDROCK_QA_MODEL_ID')

    plan = contract()
    used = []

    def structured(schema, system, user, model_id=None, **kw):
        used.append(model_id)
        return DesignReview(checks=[RequirementCheck(requirement=r, passed=True)
                                    for r in requirements(plan)])

    monkeypatch.setattr(gm, '_structured', structured)
    gm.qa_node({'concept': concept().model_dump(), 'implementation_plan': plan.model_dump(),
                'model_id': 'global.anthropic.claude-sonnet-4-6',
                'code_model_id': 'global.anthropic.claude-sonnet-4-6', 'game_html': PLAYABLE})
    assert used == [DEFAULT_QA_MODEL_ID]


def test_a_reviewer_inventing_its_own_requirements_cannot_block_the_release():
    """The rejection ratio counted everything the reviewer rejected, including strings matching no
    requirement we asked about. A reviewer that paraphrased the contract and then failed its own
    wording blocked a build whose real contract was met - which is how a playable game ended up
    published with dozens of open items against it."""
    from game_studio.graph import _verdict
    reqs = requirements(contract())
    invented = DesignReview(
        checks=[RequirementCheck(requirement=r, passed=True) for r in reqs]
        + [RequirementCheck(requirement=f'파생 요구사항 {i}', passed=False, evidence='없음')
           for i in range(12)])
    verdict = _verdict(invented, reqs)
    assert verdict.unmet == [], 'a requirement we never asked about is not an unmet requirement'
    assert not verdict.blocking

    # What the contract does say is still enforced: reject most of it and the build goes back.
    assert _verdict(DesignReview(checks=[
        RequirementCheck(requirement=r, passed=False, evidence='구현되지 않았습니다')
        for r in reqs]), reqs).blocking


def test_a_minority_of_disputed_mechanics_ships_with_its_findings_recorded():
    """Below the tolerance the build is published with the disagreement on record, instead of
    paying for another repair cycle over a couple of mechanics."""
    import game_studio.graph as gm
    from game_studio.graph import _verdict
    reqs = requirements(contract())
    disputed = reqs[:2]
    verdict = _verdict(DesignReview(checks=[
        RequirementCheck(requirement=r, passed=r not in disputed,
                         evidence='' if r not in disputed else '해당 함수가 없습니다')
        for r in reqs]), reqs)
    assert len(verdict.unmet) == len(disputed)
    assert not verdict.blocking, 'two disputed items out of eight is feedback, not a broken game'
    assert all(any(item in finding for finding in verdict.findings) for item in disputed), \
        'but every one of them still has to be reported'
    assert len(disputed) / len(reqs) <= gm.QA_REJECT_TOLERANCE


def test_advisory_notes_are_recorded_but_never_hold_a_release():
    """Other bugs the reviewer noticed are opinions about a game that may well be playable, not
    unmet terms of the approved contract. An unbounded list of them became the repair
    instructions - one run shipped with 31 open items, nearly all of them style opinions."""
    import game_studio.graph as gm
    from game_studio.graph import _verdict
    reqs = requirements(contract())
    verdict = _verdict(DesignReview(
        checks=[RequirementCheck(requirement=r, passed=True) for r in reqs],
        findings=[f'스타일 지적 {i}' for i in range(40)]), reqs)
    assert not verdict.blocking, 'advisories alone must never block a release'
    assert verdict.unmet == []
    assert len(verdict.advisories) == gm.ADVISORY_FINDING_LIMIT, 'and they have to be capped'
    assert len(verdict.findings) == gm.ADVISORY_FINDING_LIMIT, 'while still reaching the report'


def test_the_repair_prompt_carries_the_unmet_contract_not_the_opinions(monkeypatch):
    """Only the unmet terms are worth paying a repair for. Advisories stay visible in the report
    but must not steer the rewrite."""
    import game_studio.graph as gm
    plan = contract()
    reqs = requirements(plan)
    monkeypatch.setattr(gm, '_structured', lambda *a, **kw: DesignReview(
        checks=[RequirementCheck(requirement=r, passed=r != '가속',
                                 evidence='' if r != '가속' else 'accelerate()가 없습니다')
                for r in reqs],
        findings=['변수명이 마음에 들지 않습니다']))
    qa = gm.qa_node({'concept': concept().model_dump(), 'implementation_plan': plan.model_dump(),
                     'game_html': PLAYABLE})['qa']
    assert '가속' in qa['repair_instructions']
    assert '변수명' not in qa['repair_instructions'], 'a style note must not drive a rewrite'
    assert any('변수명' in finding for finding in qa['findings']), 'but it is still on record'


def test_a_passing_check_does_not_have_to_justify_itself():
    """Demanding written evidence for every requirement made the audit's output grow with the
    contract until it ran out of budget mid-array. Only a rejection has to say what is missing."""
    review = DesignReview.model_validate({'checks': [{'requirement': '가속', 'passed': True}]})
    assert review.checks[0].evidence == ''


def test_a_truncated_answer_is_retried_with_more_room_not_the_same_budget(monkeypatch):
    """A truncated answer is not a schema violation the model can fix by being told about it: it
    ran out of room. Re-asking on the same budget failed in exactly the same place, which is what
    made running out of tokens look like the model ignoring the schema."""
    from game_studio import agents
    budgets, prompts = [], []

    class Model:
        def __init__(self, max_tokens):
            self.max_tokens = max_tokens

        def bind_tools(self, tools, tool_choice=None):
            return self

        def stream(self, messages):
            prompts.append(messages[1][1])
            budgets.append(self.max_tokens)
            # Cut off before the first field, exactly as a real over-budget answer arrives.
            yield AIMessageChunk(
                content='',
                tool_call_chunks=[{'name': 'GameConcept', 'args': '{', 'id': 'c1', 'index': 0}],
                response_metadata={'stopReason': 'max_tokens'})

    monkeypatch.setattr(agents, '_model', lambda *a, max_tokens=0, **kw: Model(max_tokens))
    monkeypatch.setattr(agents, '_note', lambda *a: None)
    monkeypatch.setattr(agents, 'STRUCTURED_MAX_ATTEMPTS', 2)
    with pytest.raises(ValidationError):
        agents._structured(GameConcept, 'sys', 'original request', 'm',
                           on_chunk=lambda _: None, max_tokens=4000)

    assert budgets == [4000, 8000], 'the retry after a truncation has to get more room'
    assert '짧게' in prompts[1], 'and be told to be shorter, not to obey the schema it already knew'
    assert 'original request' in prompts[1], 'without losing what was asked'
    assert agents.STRUCTURED_MAX_RETRY_TOKENS >= agents.STRUCTURED_MAX_TOKENS, \
        'the ceiling still has to end a runaway answer'


def test_a_schema_violation_is_still_retried_on_the_same_budget(monkeypatch):
    """Growing the budget is the answer to truncation only. A model that genuinely omitted a field
    has to be told what the validator rejected, on the budget it already had."""
    from game_studio import agents
    budgets, prompts = [], []

    class Model:
        def __init__(self, max_tokens):
            self.max_tokens = max_tokens

        def bind_tools(self, tools, tool_choice=None):
            return self

        def stream(self, messages):
            prompts.append(messages[1][1])
            budgets.append(self.max_tokens)
            yield AIMessageChunk(
                content='',
                tool_call_chunks=[{'name': 'GameConcept', 'args': '{"title": "t"}',
                                   'id': 'c1', 'index': 0}],
                response_metadata={'stopReason': 'end_turn'})

    monkeypatch.setattr(agents, '_model', lambda *a, max_tokens=0, **kw: Model(max_tokens))
    monkeypatch.setattr(agents, '_note', lambda *a: None)
    monkeypatch.setattr(agents, 'STRUCTURED_MAX_ATTEMPTS', 2)
    with pytest.raises(ValidationError):
        agents._structured(GameConcept, 'sys', 'original request', 'm',
                           on_chunk=lambda _: None, max_tokens=4000)

    assert budgets == [4000, 4000], 'a real schema violation does not buy more room'
    assert 'elevator_pitch' in prompts[1], 'it is told which field it missed'


def test_a_long_answer_is_accumulated_without_re_parsing_what_came_before():
    """`gathered = gathered + piece` re-parses the whole accumulated tool argument on every chunk,
    so accumulating a streamed answer was quadratic in the answer's own length - and the code agent
    writes an entire game as one tool argument. Measured on a 24KB game: 1.112s of pure CPU to
    accumulate the old way against 0.0007s this way, after the model had already finished."""
    import json
    import time

    from langchain_core.messages import AIMessageChunk

    from game_studio.agents import StreamAccumulator

    payload = json.dumps({'html': '<html>' + 'a' * 24000 + '</html>'})
    pieces = [AIMessageChunk(
        content='', tool_call_chunks=[{'name': 'write_game_file', 'args': payload[i:i + 40],
                                       'id': 'c1', 'index': 0}])
        for i in range(0, len(payload), 40)]

    accumulator = StreamAccumulator()
    started = time.monotonic()
    for piece in pieces:
        accumulator.add(piece)
    answer = accumulator.finish()
    elapsed = time.monotonic() - started

    assert elapsed < 0.25, f'accumulating one game must not cost seconds of CPU, took {elapsed:.3f}s'
    # And it still has to be the same answer: one tool call, arguments parsed once, intact.
    assert [call['name'] for call in answer.tool_calls] == ['write_game_file']
    assert answer.tool_calls[0]['args']['html'].endswith('</html>')
    assert len(answer.tool_calls[0]['args']['html']) == len('<html>') + 24000 + len('</html>')


def test_a_truncated_tool_argument_is_still_repaired_the_way_langchain_does_it():
    """The partial-JSON repair is relied on to report a cut-off answer honestly rather than as an
    unexplained schema violation, so assembling the turn in one go must not lose it."""
    from langchain_core.messages import AIMessageChunk

    from game_studio.agents import StreamAccumulator

    accumulator = StreamAccumulator()
    for fragment in ('{"html": "<html>', 'unfinished'):
        accumulator.add(AIMessageChunk(content='', tool_call_chunks=[
            {'name': 'write_game_file', 'args': fragment, 'id': 'c1', 'index': 0}]))
    accumulator.add(AIMessageChunk(content='', response_metadata={'stopReason': 'max_tokens'}))

    assert accumulator.truncated(), 'running out of room has to stay visible'
    assert accumulator.finish().tool_calls[0]['args']['html'] == '<html>unfinished'


def test_a_live_preview_is_built_on_a_timer_not_on_every_chunk(monkeypatch):
    """Composing a preview re-reads everything generated so far, so doing it per chunk is quadratic
    in the answer's length - and the longest calls here emit thousands of chunks."""
    from langchain_core.messages import AIMessageChunk

    from game_studio import agents

    class Model:
        def stream(self, _messages):
            for _ in range(400):
                yield AIMessageChunk(content='token ')

    shown = []
    agents.stream_turn(Model(), [], on_preview=lambda turn: shown.append(turn.text))
    # One throttled tick at most, plus the final flush that guarantees the finished answer is shown.
    assert len(shown) <= 3, f'previews must be rate-limited at the source, got {len(shown)}'
    assert shown and shown[-1].endswith('token '), 'the finished answer still has to be shown'


def test_the_contract_is_sized_and_ordered_so_a_half_finished_build_is_still_playable():
    """The measured failure this is all for. A Mario-like run produced a contract of eight
    mechanics averaging 190 characters - per-frame acceleration, friction and boost tables, ? block
    item tiers, coin 1-ups, flagpole scoring bands - and eight acceptance tests that wanted frame
    logs and coordinate logs. QA passed it with five unmet, because five out of eleven is under the
    rejection tolerance. The unmet five included running and jumping.

    Three rules come out of that, and all three are enforced rather than requested: the contract is
    short, each item is a sentence instead of a specification, and the order is the build order
    with the playable core first.
    """
    import game_studio.graph as gm
    from game_studio.models import (
        CONTRACT_ITEM_CHARS,
        CONTRACT_MAX_ITEMS,
        GameConcept,
        ImplementationPlan,
    )

    # The rendered prompt, not its source: the limits reach the model through an f-string, so
    # reading the source would pass while the numbers were never interpolated.
    captured = {}
    patch = pytest.MonkeyPatch()
    patch.setattr(gm, "_structured",
                  lambda schema, system, user, *a, **kw: captured.update(system=system, user=user))
    try:
        gm._implementation_plan(
            GameConcept(title="t", elevator_pitch="p", player_goal="g", controls=["a", "b"],
                        core_loop=["1", "2", "3"], difficulty_curve="d", visual_direction="v"),
            "brief", "", None, None)
    finally:
        patch.undo()
    prompt = captured["system"]
    assert str(CONTRACT_MAX_ITEMS) in prompt and str(CONTRACT_ITEM_CHARS) in prompt, \
        "the limits have to reach the model, not only the validator"
    assert "build order" in prompt, "the order is the build order and it has to say so"
    assert "sixty seconds" in prompt, "an acceptance test nobody can watch cannot be verified here"
    assert "no frame counts" in prompt.lower()

    # A per-frame tuning paragraph is refused outright - it is a specification, not a contract item.
    spec = ("【달리기 & 가속】플레이어가 방향키를 누르면 수평 속도가 0에서 최대 4px/프레임까지 "
            "0.4px/프레임²로 선형 가속된다. 키를 떼면 0.3px/프레임²로 감속한다. 최대 속도에 "
            "도달하면 달리기 상태 플래그가 켜지고 점프 시 수평 관성이 유지된다. 공중에서는 "
            "가속도가 절반으로 줄고 마찰은 적용되지 않는다.")
    assert len(spec) > CONTRACT_ITEM_CHARS
    with pytest.raises(ValidationError):
        ImplementationPlan(genre="플랫포머", mechanics=[spec, "점프한다", "적을 밟는다"],
                           win_condition="w", loss_condition="l",
                           state_transitions=["1", "2", "3"],
                           acceptance_tests=["t1", "t2", "t3"])

    # The same mechanic said as a mechanic passes.
    ImplementationPlan(genre="플랫포머",
                       mechanics=["방향키로 좌우 이동하고 점프한다", "적을 밟아 처치한다", "깃발에 닿으면 클리어"],
                       win_condition="깃발 도달", loss_condition="구덩이 낙하",
                       state_transitions=["시작", "진행", "결과"],
                       acceptance_tests=["시작하면 바로 조작할 수 있다", "적을 밟으면 사라진다", "죽으면 재시작된다"])


def test_the_code_agent_is_told_to_reach_playable_before_complete():
    """A finite call budget makes build order a correctness question, not a style one: whatever is
    on disk when the turns run out is what ships. One measured run spent its budget on ? blocks and
    flagpole scoring and delivered a game that never started."""
    from game_studio.prompts import CODE_SYSTEM, GODOT_CODE_SYSTEM

    for system in (CODE_SYSTEM, GODOT_CODE_SYSTEM):
        assert "PLAYABLE FIRST" in system
        assert "runs out of them" in system, "the reason has to be there, not just the rule"
    # Godot's most common breakage is a scene referenced a turn before it is written.
    assert "Never reference a file you have not written yet" in GODOT_CODE_SYSTEM


def test_the_art_brief_does_not_carry_the_other_engines_half(tmp_path):
    """This message opens the agent's history and is re-sent on every one of its turns, so anything
    unusable in it is paid for twenty times over. canvas_effects is the art director's largest
    field - a measured run put 1,310 characters of ctx.fillRect recipes in it - and on a Godot run
    it describes an API the agent cannot call."""
    from game_studio.graph import _art_brief
    from game_studio.models import ArtDirection

    art = ArtDirection(palette={"sky": "#5C94FC"}, image_prompt="16x16 픽셀 배경",
                       asset_plan=["player: 빨간 모자"],
                       canvas_effects=["하늘 배경: ctx.fillRect으로 단색 채우기"] * 12)
    html5, godot = _art_brief(art, "html5"), _art_brief(art, "godot")

    assert "ctx.fillRect" in html5, "the canvas path still needs its canvas recipes"
    assert "ctx.fillRect" not in godot
    assert len(godot) < len(html5) / 2
    # What both engines build from survives either way.
    for brief in (html5, godot):
        assert "#5C94FC" in brief and "빨간 모자" in brief and "픽셀 배경" in brief


def test_qa_blocks_a_network_dependent_game():
    """G1: a generated game may not depend on anything outside the file it ships as. Moved here
    when the fallback-game module it used to sit beside was deleted - the rule outlived the
    template, and this is the only test that holds the Canvas side of it."""
    from game_studio.agents import static_qa

    report = static_qa('<canvas></canvas><script>fetch("https://example.com")</script>')
    assert report.status == "repair"
    assert any("external dependency" in item.lower() for item in report.findings)


def test_qa_does_not_demand_art_the_run_was_unable_to_make(tmp_path, monkeypatch):
    """A mandate the build cannot satisfy is not a finding, it is a deadlock.

    Measured on two revisions: sprites a person had rejected were retired and made mandatory, the
    agent could not generate replacements, QA blocked on their absence, and the run spent both
    repair attempts and both rethink cycles before dying - with a game that was otherwise finished
    sitting on disk.

    The pipeline already excuses this when raster generation is switched off, because a run with no
    image tool can never satisfy such a mandate. Running out of image time is the same inability
    arriving later, and it deserves the same answer: the game is finished and the art is reported as
    missing rather than held against it.
    """
    from game_studio import agent_tools
    from game_studio.required_art import missing_required

    monkeypatch.setattr(agent_tools, "_IMAGE_SECONDS", {})
    monkeypatch.setattr(agent_tools, "IMAGE_TIME_BUDGET", 100)
    workspace = str(tmp_path / "게임_html5_abc123")

    assert not agent_tools.out_of_image_time(workspace), "a fresh run can still be asked"
    agent_tools._spend_image_time(workspace, 100)
    assert agent_tools.out_of_image_time(workspace)

    # The absence itself is unchanged - what changes is whether it blocks.
    assets = tmp_path / "게임_html5_abc123" / "assets"
    assets.mkdir(parents=True)
    assert missing_required(["bird-1", "bird-2"], assets) == ["bird-1", "bird-2"]


def test_the_block_still_applies_while_the_run_can_still_generate(tmp_path, monkeypatch):
    """The excuse is inability, not inconvenience. An agent that simply did not bother is exactly
    what this backstop was built for: a re-planned art direction exists because verification said
    art was missing, and an answer nobody is obliged to act on is how the same finding came back a
    cycle later."""
    from game_studio import agent_tools

    monkeypatch.setattr(agent_tools, "_IMAGE_SECONDS", {})
    monkeypatch.setattr(agent_tools, "IMAGE_TIME_BUDGET", 600)
    workspace = str(tmp_path / "게임_html5_abc123")
    agent_tools._spend_image_time(workspace, 300)
    assert not agent_tools.out_of_image_time(workspace), "half spent is not spent"
