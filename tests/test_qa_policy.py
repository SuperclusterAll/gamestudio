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
