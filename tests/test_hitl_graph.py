from pathlib import Path
import pytest
from langchain_core.messages import AIMessage, AIMessageChunk
from pydantic import ValidationError

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from game_studio.graph import build_graph
from game_studio.models import GameConcept, ArtDirection
from game_studio.models import ImplementationPlan, DesignReview, RequirementCheck


def default_concept(brief):
    # Test fixture only; production has no prewritten concept fallback.
    return GameConcept(title='테스트 퍼즐',elevator_pitch=brief,player_goal='출구 개방',
                       controls=['방향키','클릭'],core_loop=['회전','연결','출구'],
                       difficulty_curve='퍼즐 단계 증가',visual_direction='격자')


def default_art(concept):
    return ArtDirection(palette={},image_prompt=concept.title,asset_plan=[],canvas_effects=[])


def text_chunks(text, size=8):
    """Split text into AIMessageChunks the way a real streamed answer arrives."""
    parts = [text[i:i + size] for i in range(0, len(text), size)] or ['']
    return [AIMessageChunk(content=part) for part in parts]


class FakeModel:
    """Fake chat model for the streaming call path the code and repair agents use.

    They stream rather than invoke so their progress is visible while the answer is still being
    written, so a usable double has to serve .stream(); .invoke() is kept for any caller that
    still blocks on a single response.
    """

    def __init__(self, chunks, capture=None):
        self._chunks = chunks
        self._capture = capture if capture is not None else {}

    def bind_tools(self, tools):
        self._capture['tools'] = [tool.name for tool in tools]
        return self

    def stream(self, messages):
        self._capture['messages'] = messages
        yield from self._chunks

    def invoke(self, messages):
        self._capture['messages'] = messages
        return AIMessage(content=''.join(str(chunk.content) for chunk in self._chunks))


class StubAgent:
    """Stands in for the compiled create_agent graph: code_node only streams it for updates."""

    def __init__(self, messages):
        self._messages = messages

    def stream(self, payload, config=None, stream_mode=None):
        self.payload = payload
        yield ("updates", {"model": {"messages": self._messages}})


def test_design_approval_pauses_then_resumes(tmp_path, monkeypatch):
    concept = default_concept('Requested genre: 퍼즐\nPlayer brief: 퍼즐 테스트')
    plan = ImplementationPlan(genre='퍼즐', mechanics=['회전', '연결', '출구'], win_condition='연결 완료', loss_condition='시간 초과', state_transitions=['시작', '진행', '결과'], acceptance_tests=['블록 회전', '회로 연결', '출구 개방'])
    requirements = plan.mechanics + [plan.win_condition, plan.loss_condition] + plan.acceptance_tests
    monkeypatch.setattr('game_studio.graph.run_director', lambda *a,**kw: 'Model production plan')
    monkeypatch.setattr('game_studio.graph.create_concept', lambda *a,**kw: concept)
    monkeypatch.setattr('game_studio.graph.create_art', lambda *a, **kw: default_art(concept))
    monkeypatch.setattr('game_studio.graph._structured', lambda schema,*a,**kw: plan if schema is ImplementationPlan else DesignReview(checks=[RequirementCheck(requirement=r, passed=True, evidence='Test reviewer model response for this requirement') for r in requirements], findings=[]))
    monkeypatch.setattr('game_studio.graph.build_code_agent',
                        lambda *a, **kw: StubAgent([]))
    graph = build_graph(InMemorySaver())
    config = {"configurable": {"thread_id": "hitl-test"}}
    initial = {
        "brief": "Build a short original keyboard-controlled arcade survival game.",
        "output_dir": str(tmp_path),
        "use_llm": True,
        "workspace_dir": str(tmp_path),
        "generate_images": False,
        "repair_attempts": 0,
        "trace_notes": [],
    }
    graph.invoke(initial, config)
    paused = graph.get_state(config)
    assert paused.values["implementation_plan"]["genre"] == '퍼즐'
    assert paused.next == ("approval",)
    assert not (tmp_path / 'index.html').exists()
    (tmp_path / 'draft.html').write_text('<!doctype html><html><canvas></canvas><script>requestAnimationFrame(()=>{});addEventListener("keydown",e=>({KeyW:1,KeyA:1,KeyS:1,KeyD:1})[e.code]);const score=0;function restart(){}</script></html>', encoding='utf-8')

    completed = graph.invoke(Command(resume={"decision": "approve", "comment": "Ship it"}), config)
    assert completed["approval"]["decision"] == "approved"
    assert completed["qa"]["status"] == "pass"
    assert completed["game_path"]
    assert Path(completed["game_path"]).exists()


def test_code_model_disabled_and_missing_draft_fail_instead_of_shipping_template(tmp_path):
    from game_studio.graph import code_node, qa_node, package_node
    with pytest.raises(RuntimeError, match='코드 모델'):
        code_node({'use_llm':False})
    with pytest.raises(RuntimeError, match='draft.html'):
        qa_node({'concept':default_concept('test').model_dump(), 'output_dir':str(tmp_path)})
    with pytest.raises(RuntimeError, match='검증 실패'):
        package_node({'qa':{'status':'repair','findings':['wrong genre']}})


def test_empty_design_review_is_not_a_pass(monkeypatch):
    """An audit that checked nothing must not read as a clean bill of health."""
    from game_studio.graph import qa_node
    monkeypatch.setattr('game_studio.graph._structured', lambda *a, **kw: DesignReview(checks=[], findings=[]))
    plan = ImplementationPlan(genre='로그라이크', mechanics=['공격','강화','탐험'], win_condition='보스 처치',
                              loss_condition='사망', state_transitions=['시작','전투','결과'],
                              acceptance_tests=['공격 피해','강화 반영','보스 승리'])
    state = {'concept': default_concept('test').model_dump(), 'implementation_plan': plan.model_dump(),
             'game_html': '<html><canvas>score restart keydown KeyW KeyA KeyS KeyD requestAnimationFrame</canvas></html>'}
    result = qa_node(state)
    assert result['qa']['status'] == 'repair'
    assert any('비어 있습니다' in f for f in result['qa']['findings'])


def test_qa_does_not_fail_over_the_reviewers_wording(monkeypatch):
    """Requirements were matched by exact string, so a paraphrase or a requirement the reviewer
    simply did not mention counted as unmet and sent a working build back into repair."""
    from game_studio.graph import qa_node
    plan = ImplementationPlan(genre='로그라이크', mechanics=['공격','강화','탐험'], win_condition='보스 처치',
                              loss_condition='사망', state_transitions=['시작','전투','결과'],
                              acceptance_tests=['공격 피해','강화 반영','보스 승리'])
    # One check is reworded, one requirement is skipped entirely, none is rejected.
    checks = [RequirementCheck(requirement='공격', passed=True, evidence='attack() 에서 피해 적용됨'),
              RequirementCheck(requirement=' 강 화 ', passed=True, evidence='upgrade() 에서 반영됨')]
    monkeypatch.setattr('game_studio.graph._structured', lambda *a, **kw: DesignReview(checks=checks, findings=[]))
    state = {'concept': default_concept('test').model_dump(), 'implementation_plan': plan.model_dump(),
             'game_html': '<html><canvas>score restart keydown KeyW KeyA KeyS KeyD requestAnimationFrame</canvas></html>'}
    assert qa_node(state)['qa']['status'] == 'pass'

    # One rejected item out of eight is recorded but does not hold the release.
    monkeypatch.setattr('game_studio.graph._structured', lambda *a, **kw: DesignReview(
        checks=[RequirementCheck(requirement='공격', passed=False, evidence='공격 처리가 없습니다')], findings=[]))
    single = qa_node(state)
    assert single['qa']['status'] == 'pass'
    assert any('공격' in f for f in single['qa']['findings']), 'it still has to be reported'

    # Rejecting most of the contract does block it.
    reqs = plan.mechanics + [plan.win_condition, plan.loss_condition] + plan.acceptance_tests
    monkeypatch.setattr('game_studio.graph._structured', lambda *a, **kw: DesignReview(
        checks=[RequirementCheck(requirement=r, passed=False, evidence='구현되지 않았습니다') for r in reqs],
        findings=[]))
    assert qa_node(state)['qa']['status'] == 'repair'


def test_code_node_hands_the_agent_the_selected_model_and_the_approved_design(tmp_path, monkeypatch):
    """The loop moved inside create_agent, so what matters here is what code_node gives it."""
    import game_studio.graph as gm
    captured = {}

    def factory(model_id, tools, system_prompt, retry_on):
        captured.update(model_id=model_id, tools=[t.name for t in tools],
                        system=system_prompt, retry_on=retry_on)
        return StubAgent([AIMessage(content='done')])

    monkeypatch.setattr(gm, 'build_code_agent', factory)
    concept = default_concept('test')
    (tmp_path / 'draft.html').write_text('<html><canvas></canvas></html>', encoding='utf-8')
    result = gm.code_node({'brief': 'Implement jumping and gravity', 'concept': concept.model_dump(),
                           'art': default_art(concept).model_dump(),
                           'implementation_plan': {'mechanics': ['gravity']},
                           'output_dir': str(tmp_path), 'workspace_dir': str(tmp_path),
                           'code_model_id': 'selected-coding-model', 'model_id': 'planning-model'})
    assert captured['model_id'] == 'selected-coding-model'
    assert 'write_game_file' in captured['tools'] and 'run_static_qa' in captured['tools']
    assert captured['retry_on'] is gm._is_transient
    # The approved design reaches the agent as its task.
    task = gm._code_task({'brief': 'Implement jumping and gravity', 'concept': concept.model_dump(),
                          'art': default_art(concept).model_dump(),
                          'implementation_plan': {'mechanics': ['gravity']}})
    assert 'Implement jumping and gravity' in task and 'gravity' in task
    # The transcript deliberately does not come back into graph state: the agent rebuilds its own
    # payload on every entry, nothing reads it, and writing it carried whole games - as
    # write_game_file arguments - into every checkpoint, growing with each rethink.
    assert 'messages' not in result
    assert result['stage'] == 'code' 


def test_code_agent_middleware_carries_the_budgets_the_loop_used_to_hand_roll():
    from langchain.agents.middleware import (ContextEditingMiddleware, ModelCallLimitMiddleware,
                                             ModelRetryMiddleware, TodoListMiddleware)
    from game_studio.code_agent import (MODEL_CALL_LIMIT, StudioObservability, build_code_agent)
    import inspect
    source = inspect.getsource(build_code_agent)
    for cls in (StudioObservability, ModelCallLimitMiddleware, ModelRetryMiddleware,
                ContextEditingMiddleware, TodoListMiddleware):
        assert cls.__name__ in source, f"{cls.__name__} is not in the middleware stack"
    assert 'clear_tool_inputs=True' in source, 'the game source must be cleared from old turns'
    assert MODEL_CALL_LIMIT >= 12, 'the budget has to cover coding plus sprite generation'


def test_the_supervisor_is_the_only_node_that_routes():
    """Every stage reports back to the supervisor and the supervisor alone chooses the next edge,
    so there is no second place - a router beside QA, a rethink node of its own - that can decide
    where a run goes. Image generation and the write/inspect/repair loop belong to the code agent,
    so neither is an outer-graph node either."""
    from game_studio.graph import DESTINATIONS, WORKERS
    graph = build_graph().get_graph()
    for gone in ("image_agent", "image_tools", "tools", "director", "rethink"):
        assert gone not in graph.nodes, f"{gone} should no longer be an outer-graph node"
    edges = {(e.source, e.target) for e in graph.edges}
    assert ("__start__", "supervisor") in edges, "the run opens on the supervisor"
    for worker in WORKERS:
        assert (worker, "supervisor") in edges, f"{worker} must report back to the supervisor"
        assert not [t for s, t in edges if s == worker and t != "supervisor"], \
            f"{worker} routes somewhere other than the supervisor"
    for destination in DESTINATIONS:
        assert ("supervisor", destination) in edges, f"the supervisor cannot reach {destination}"
    # Every path still ends the run rather than throwing.
    for terminal in ("package", "abandoned", "rejected"):
        assert (terminal, "__end__") in edges


def _code_prompt(generate_images):
    """The system prompt is pure a function of state, so assert it directly."""
    from game_studio.graph import _code_system_prompt
    concept = default_concept('test')
    art = default_art(concept).model_dump()
    art['asset_plan'] = ['플레이어 차량: 빨간 스포츠카 16x32', '라이벌 차량: 청록 세단 16x32']
    return _code_system_prompt({'concept': concept.model_dump(), 'art': art,
                                'generate_images': generate_images})


def test_code_agent_is_required_to_render_the_planned_sprites():
    """The art agent planned a sprite per object and the run generated none, shipping a Canvas
    rectangle for the player. With images enabled the plan has to reach the code agent as a
    requirement, not as an optional menu it can quietly skip."""
    system = _code_prompt(generate_images=True)
    assert 'generate_comfyui_image' in system
    assert 'asset_name' in system
    # The planned objects themselves must be in the prompt, not just a pointer to them.
    assert '플레이어 차량: 빨간 스포츠카 16x32' in system
    assert 'primary opponent or obstacle' in system
    assert 'Canvas-only rectangle' in system


def test_code_agent_is_told_canvas_only_when_images_are_disabled():
    system = _code_prompt(generate_images=False)
    assert 'Canvas-only art' in system
    assert 'asset_name' not in system


def test_a_clean_design_review_validates():
    """A reviewer with nothing extra to report omits the key; that must not be an error."""
    review = DesignReview.model_validate(
        {'checks': [{'requirement': '가속', 'passed': True, 'evidence': 'accelerate() 에서 속도 증가'}]})
    assert review.findings == []
    # What the audit genuinely needs is still mandatory.
    with pytest.raises(Exception):
        DesignReview.model_validate({'findings': []})


def test_structured_output_retries_with_the_validator_complaint(monkeypatch):
    """One schema violation used to end the node; now the model is told what was wrong and asked
    again, which is what keeps a single omitted field from discarding a whole run."""
    from game_studio import agents
    concept = default_concept('test')
    prompts, attempts = [], {'n': 0}

    class Structured:
        def invoke(self, messages):
            prompts.append(messages[1][1])
            attempts['n'] += 1
            if attempts['n'] == 1:
                raise ValidationError.from_exception_data(
                    'GameConcept',
                    [{'type': 'missing', 'loc': ('core_loop',), 'input': {}}])
            return concept

    class Model:
        def with_structured_output(self, schema):
            return Structured()

    monkeypatch.setattr(agents, '_model', lambda *a, **kw: Model())
    result = agents._structured(GameConcept, 'sys', 'original request', 'm')

    assert result is concept
    assert attempts['n'] == 2, 'the failed attempt must be retried'
    assert 'original request' in prompts[1], 'the retry keeps the original request'
    assert 'core_loop' in prompts[1], 'and tells the model which field it missed'


def test_a_truncated_structured_answer_says_so_instead_of_looking_like_a_schema_violation(monkeypatch):
    """LangChain's partial-JSON parser closes off a half-written tool argument and hands back what
    looks like a perfectly good call, just missing every field the model never reached. That
    surfaced as a bare "checks: Field required" and read like the schema had been ignored."""
    from game_studio import agents
    from game_studio.models import DesignReview

    class Truncated:
        def bind_tools(self, tools, tool_choice=None):
            return self

        def stream(self, _messages):
            yield AIMessageChunk(
                content='',
                tool_call_chunks=[{'name': 'DesignReview', 'args': '{"findings": []}',
                                   'id': 'c1', 'index': 0}],
                response_metadata={'stopReason': 'max_tokens'})

    notes = []
    monkeypatch.setattr(agents, '_model', lambda *a, **kw: Truncated())
    monkeypatch.setattr(agents, '_note', lambda agent, text: notes.append(text))
    monkeypatch.setattr(agents, 'STRUCTURED_MAX_ATTEMPTS', 1)
    with pytest.raises(ValidationError):
        agents._structured(DesignReview, 'sys', 'user', 'm', on_chunk=lambda _: None)
    assert any('끊겼습니다' in note for note in notes), 'the real cause has to be reported'


def test_the_design_review_asks_for_more_output_budget_than_the_flat_schemas(monkeypatch):
    """It returns one check per requirement with its own evidence, so its answer grows with the
    contract - measured at 4.6-8.4KB of JSON for 18 requirements. Sharing the budget sized for the
    flat concept/art schemas left no headroom and truncated the audit mid-array."""
    import game_studio.graph as gm
    from game_studio.agents import STRUCTURED_MAX_TOKENS
    assert gm.DESIGN_REVIEW_MAX_TOKENS > STRUCTURED_MAX_TOKENS

    plan = ImplementationPlan(genre='퍼즐', mechanics=['a', 'b', 'c'], win_condition='w',
                              loss_condition='l', state_transitions=['1', '2', '3'],
                              acceptance_tests=['t1', 't2', 't3'])
    reqs = plan.mechanics + [plan.win_condition, plan.loss_condition] + plan.acceptance_tests
    captured = {}

    def structured(schema, system, user, model_id=None, **kw):
        captured.update(kw)
        return DesignReview(checks=[RequirementCheck(requirement=r, passed=True,
                                                     evidence='코드 근거가 여기에 있습니다')
                                    for r in reqs])

    monkeypatch.setattr(gm, '_structured', structured)
    state = {'concept': default_concept('t').model_dump(), 'implementation_plan': plan.model_dump(),
             'game_html': '<html><canvas>score restart keydown KeyW KeyA KeyS KeyD '
                          'requestAnimationFrame</canvas></html>'}
    assert gm.qa_node(state)['qa']['status'] == 'pass'
    assert captured['max_tokens'] == gm.DESIGN_REVIEW_MAX_TOKENS


def test_transient_bedrock_timeout_is_retried_instead_of_losing_the_run(monkeypatch):
    """A read timeout mid-generation used to unwind the whole graph and discard an approved run."""
    from botocore.exceptions import ReadTimeoutError
    import game_studio.graph as gm
    from game_studio.graph import MODEL_RETRY, _is_transient

    assert _is_transient(ReadTimeoutError(endpoint_url="https://bedrock-runtime.us-east-1.amazonaws.com"))
    assert MODEL_RETRY.retry_on is _is_transient and MODEL_RETRY.max_attempts >= 2
    # Every model-calling node must carry the policy; the tools node must not, since its tools
    # write files and generate images and a retry there would duplicate real side effects.
    nodes = gm.build_graph().nodes
    for name in ("supervisor", "idea", "design_document", "art", "code", "qa", "repair"):
        assert nodes[name].retry_policy, f"{name} has no retry policy"
    # Inside the code agent the same distinction holds: model calls are retried, tool calls are
    # not, because the tools write files and generate images and a retry would repeat that.
    import inspect
    from game_studio.code_agent import build_code_agent
    stack = inspect.getsource(build_code_agent)
    assert "ModelRetryMiddleware" in stack
    assert "ToolRetryMiddleware" not in stack, "retrying tool side effects is not safe"


def test_hitting_the_token_cap_is_reported(monkeypatch):
    """A truncated answer is otherwise invisible: LangChain's partial-JSON parser closes off a
    half-streamed tool argument, so the call looks fine and only the file inside is cut short."""
    import game_studio.graph as gm
    logged = []
    monkeypatch.setattr(gm, '_log', lambda kind, **kw: logged.append((kind, kw)))
    monkeypatch.setattr(gm, '_chunk_sink', lambda step: None)

    class Truncated:
        max_tokens = 16000

        def stream(self, _messages):
            yield AIMessageChunk(
                content='',
                tool_call_chunks=[{'name': 'write_game_file',
                                   'args': '{"html": "<!doctype html><html>',
                                   'id': 'c1', 'index': 0}],
                response_metadata={'stopReason': 'max_tokens'})

    result = gm._stream_call(Truncated(), [], '코드 Agent', 'code')
    # The parser does produce a usable-looking call, which is exactly why the cap must be reported.
    assert result.tool_calls and result.tool_calls[0]['name'] == 'write_game_file'
    assert '</html>' not in result.tool_calls[0]['args']['html'], 'the html really is truncated'
    assert any('끊겼습니다' in kw.get('text', '') for _, kw in logged), 'the cap must be logged'


def test_failing_qa_escalates_through_the_supervisor_and_still_finishes(tmp_path, monkeypatch):
    """A build QA keeps rejecting must not kill the graph: the supervisor spends its rethink cycles
    on fresh code loops, then its repair, then ends the run with a verdict rather than throwing."""
    import game_studio.graph as gm
    from game_studio.graph import build_graph as build
    from game_studio.models import SupervisorDecision

    concept = default_concept('test')
    plan = ImplementationPlan(genre='퍼즐', mechanics=['회전','연결','출구'], win_condition='완료',
                              loss_condition='초과', state_transitions=['s','p','r'],
                              acceptance_tests=['t1','t2','t3'])
    monkeypatch.setattr(gm, 'run_director', lambda *a, **kw: 'plan')
    monkeypatch.setattr(gm, 'create_concept', lambda *a, **kw: concept)
    monkeypatch.setattr(gm, 'create_art', lambda *a, **kw: default_art(concept))
    reqs = plan.mechanics + [plan.win_condition, plan.loss_condition] + plan.acceptance_tests

    def structured(schema, *a, **kw):
        if schema is ImplementationPlan:
            return plan
        if schema is SupervisorDecision:
            return SupervisorDecision(action='code', reason='계약이 구현되지 않았습니다',
                                      instructions='1. 계약의 메커닉을 구현하세요')
        return DesignReview(
            checks=[RequirementCheck(requirement=r, passed=False, evidence='구현되지 않았습니다')
                    for r in reqs],
            findings=['게임이 계약을 충족하지 않습니다'])

    monkeypatch.setattr(gm, '_structured', structured)
    # Every draft the agents produce is missing 'score', so static QA rejects it forever.
    monkeypatch.setattr(gm, '_model', lambda *a, **kw: FakeModel(text_chunks('no fix')))
    # The loop lives inside create_agent now; the draft it leaves behind is what QA judges.
    draft = ('<!doctype html><html><canvas></canvas><script>requestAnimationFrame(()=>{});'
             'addEventListener("keydown",e=>({KeyW:1,KeyA:1,KeyS:1,KeyD:1})[e.code]);'
             'function restart(){}</script></html>')

    def stub_agent(*a, **kw):
        (tmp_path / 'draft.html').write_text(draft, encoding='utf-8')
        return StubAgent([])

    monkeypatch.setattr(gm, 'build_code_agent', stub_agent)

    graph = build(InMemorySaver())
    config = {"configurable": {"thread_id": "escalate"}, "recursion_limit": 200}
    graph.invoke({"brief": "b", "output_dir": str(tmp_path), "workspace_dir": str(tmp_path),
                  "use_llm": True, "generate_images": False, "repair_attempts": 0,
                  "trace_notes": []}, config)
    (tmp_path / 'draft.html').write_text('<html><canvas></canvas></html>', encoding='utf-8')
    final = graph.invoke(Command(resume={"decision": "approve", "comment": ""}), config)

    assert final['qa']['status'] != 'pass'
    # Every rethink cycle the budget allows is spent before the run gives up - and no more. The
    # budget is what it is set to, not a number frozen into this test: a rethink is a whole code
    # agent loop plus another audit, so it is deliberately tunable.
    assert final.get('rethink_cycles') == gm.MAX_RETHINK_CYCLES, 'every rethink cycle must be spent'
    assert final.get('qa_guidance'), 'the rethink has to produce fix instructions'
    assert final.get('qa_report_path'), 'the run must end with an inspectable verdict'
    assert (tmp_path / 'qa-report.json').is_file()
    # Publish the best draft rather than producing nothing, but never claim it passed.
    assert (tmp_path / 'index.html').is_file(), 'the run must still deliver a playable game'
    assert final.get('game_path')
    import json as _json
    manifest = _json.loads((tmp_path / 'production-manifest.json').read_text(encoding='utf-8'))
    assert manifest['generation_mode'] == 'model_generated_qa_failed'
    assert manifest['qa_outstanding'], 'the open findings have to travel with the game'


class RecordingModel:
    """A chat model that answers once and remembers what it was asked."""

    def __init__(self, answer="한 문장 루프: 블록을 쌓는다.\n이번에 만들지 않는 것: 멀티플레이."):
        self.answer, self.seen = answer, []

    def invoke(self, messages):
        self.seen.append(messages)
        return AIMessage(content=self.answer)


def test_the_director_decides_scope_in_one_call_instead_of_planning_the_game_again(monkeypatch):
    """It used to be a deepagents supervisor whose four subagents ran on IDEA_SYSTEM, ART_SYSTEM,
    CODE_SYSTEM and QA_SYSTEM - the same prompts the real stages run on. Delegating to its "idea"
    subagent therefore ran the idea agent, and then idea_node ran it again: the planning happened
    twice, up to twelve steps of it, and the first copy was thrown away except for one paragraph.

    What no later stage can do for itself is say what to leave out, and that is worth one call.
    """
    from game_studio import agents

    # Switched on explicitly: the pass is off by default in .env, and the dashboard test's
    # lifespan loads that file into this process. A test that only passes while a config file
    # says so is testing the config file.
    monkeypatch.setattr(agents, "DIRECTOR_TIMEOUT_SECONDS", 60)
    model = RecordingModel()
    monkeypatch.setattr(agents, "_model", lambda *a, **kw: model)
    brief = agents.run_director("슈퍼마리오와 똑같은 게임", True, "m")

    assert len(model.seen) == 1, "one call, not a fan-out of subagent loops"
    assert "블록을 쌓는다" in brief

    system = "".join(str(part) for role, part in model.seen[0] if role == "system")
    assert "only job is scope" in system
    assert "이번에 만들지 않는 것" in system, "naming what is cut is the whole point of the pass"
    assert "Do not design the game" in system


def test_the_director_is_told_which_engine_the_run_delivers(monkeypatch):
    """The old director only ever knew the Canvas path - its code subagent ran on CODE_SYSTEM and
    never on GODOT_CODE_SYSTEM - so a Godot run was coordinated as "a shippable standalone HTML
    game", and the planners were then told to align their concept with that description."""
    from game_studio import agents

    monkeypatch.setattr(agents, "DIRECTOR_TIMEOUT_SECONDS", 60)
    model = RecordingModel()
    monkeypatch.setattr(agents, "_model", lambda *a, **kw: model)

    agents.run_director("점프 게임", True, "m", engine="godot")
    agents.run_director("점프 게임", True, "m", engine="html5")
    godot, html5 = (str(seen[-1][1]) for seen in model.seen)

    assert "Godot 4" in godot and "GDScript" in godot
    assert "HTML" not in godot.replace("HTML5", ""), "a Godot run must not be briefed as a web page"
    assert "HTML Canvas" in html5 and "Godot" not in html5


def test_the_director_pass_can_be_switched_off(monkeypatch):
    from game_studio import agents

    called = []
    monkeypatch.setattr(agents, "_model", lambda *a, **kw: called.append(1))
    monkeypatch.setattr(agents, "DIRECTOR_TIMEOUT_SECONDS", 0)  # .env 기본값과 같은 값

    assert agents.run_director("brief", True, "m") == ""
    assert not called, "a zero budget must skip the call entirely"
    # Offline is the same non-event, and so is a brief that is skipped: "" rather than a sentence
    # explaining itself, because this value is handed to the planners as their production brief.
    monkeypatch.setattr(agents, "DIRECTOR_TIMEOUT_SECONDS", 60)
    assert agents.run_director("brief", False, "m") == ""
    assert not called


def test_a_director_brief_is_bounded(monkeypatch):
    """Past a paragraph the director is designing the game again, which is what it stopped doing."""
    from game_studio import agents

    monkeypatch.setattr(agents, "DIRECTOR_TIMEOUT_SECONDS", 60)
    monkeypatch.setattr(agents, "_model", lambda *a, **kw: RecordingModel("가" * 5000))
    assert len(agents.run_director("brief", True, "m")) == agents.DIRECTOR_BRIEF_CHARS


def test_director_brief_reaches_the_planning_agents(monkeypatch):
    """The expensive supervisor pass has to shape the game, not just land in the manifest."""
    from game_studio.graph import idea_node, design_document_node
    seen = {}
    concept = default_concept('test')
    plan = ImplementationPlan(genre='퍼즐', mechanics=['a','b','c'], win_condition='w',
                              loss_condition='l', state_transitions=['1','2','3'],
                              acceptance_tests=['t1','t2','t3'])

    def structured(schema, system, user, model_id=None, **kw):
        seen.setdefault(schema.__name__, user)
        return plan if schema is ImplementationPlan else concept

    monkeypatch.setattr('game_studio.agents._structured', structured)
    monkeypatch.setattr('game_studio.graph._structured', structured)
    state = {'brief': 'b', 'production_brief': 'DIRECTOR-PLAN-MARKER', 'concept': concept.model_dump()}
    idea_node(state)
    design_document_node(state)
    assert 'DIRECTOR-PLAN-MARKER' in seen['GameConcept']
    assert 'DIRECTOR-PLAN-MARKER' in seen['ImplementationPlan']


def test_blank_player_experience_calls_model_instead_of_selecting_a_template(monkeypatch):
    from game_studio.agents import create_concept
    calls=[]
    def response(*args, **kwargs):
        calls.append(args)
        return default_concept('model result')
    monkeypatch.setattr('game_studio.agents._structured',response)
    create_concept('',True,'planning-model')
    create_concept('',True,'planning-model')
    assert len(calls)==2
    assert calls[0][-1]=='planning-model'


def test_identical_source_is_not_re_audited(monkeypatch):
    """The design review ships the whole game source, so repair cycles were paying for the same
    audit again. Byte-identical HTML reuses the previous verdict."""
    import game_studio.graph as gm
    calls = []
    monkeypatch.setattr(gm, '_structured', lambda *a, **kw: calls.append(1))
    html = '<!doctype html><html><canvas></canvas><script>requestAnimationFrame(()=>{});' \
           'addEventListener("keydown",e=>({KeyW:1,KeyA:1,KeyS:1,KeyD:1})[e.code]);' \
           'const score=0;function restart(){}</script></html>'
    import hashlib
    plan = ImplementationPlan(genre='퍼즐', mechanics=['a', 'b', 'c'], win_condition='w',
                              loss_condition='l', state_transitions=['1', '2', '3'],
                              acceptance_tests=['t1', 't2', 't3'])
    state = {'concept': default_concept('t').model_dump(), 'implementation_plan': plan.model_dump(),
             'game_html': html, 'qa': {'status': 'pass', 'findings': []},
             'design_review': {'checks': [], 'findings': []},
             'design_review_hash': hashlib.sha256(html.encode()).hexdigest()}
    result = gm.qa_node(state)
    assert not calls, 'an unchanged source must not be sent to the reviewer again'
    assert result['design_review'] is state['design_review']


def test_code_node_retries_when_the_agent_writes_nothing(tmp_path, monkeypatch):
    """create_agent stops as soon as a turn has no tool calls, so one chatty answer could end the
    build with no file at all and leave QA to fail on a draft that never existed."""
    import game_studio.graph as gm
    tasks = []

    class ChattyThenWrites:
        def stream(self, payload, config=None, stream_mode=None):
            tasks.append(payload["messages"][0][1])
            if len(tasks) == 2:   # second attempt actually writes
                (tmp_path / 'draft.html').write_text('<html><canvas></canvas></html>', encoding='utf-8')
            yield ("updates", {"model": {"messages": [AIMessage(content=f'turn {len(tasks)}')]}})

    monkeypatch.setattr(gm, 'build_code_agent', lambda *a, **kw: ChattyThenWrites())
    concept = default_concept('test')
    result = gm.code_node({'brief': 'b', 'concept': concept.model_dump(),
                           'art': default_art(concept).model_dump(),
                           'implementation_plan': {}, 'output_dir': str(tmp_path),
                           'workspace_dir': str(tmp_path), 'code_model_id': 'm'})
    assert len(tasks) == 2, 'a build that produced no draft must be retried'
    assert 'write_game_file' in tasks[1], 'the retry has to demand the tool explicitly'
    assert 'messages' not in result, 'the transcript stays out of graph state'
    # Once a draft exists there is no further retry.
    tasks.clear()
    gm.code_node({'brief': 'b', 'concept': concept.model_dump(),
                  'art': default_art(concept).model_dump(), 'implementation_plan': {},
                  'output_dir': str(tmp_path), 'workspace_dir': str(tmp_path), 'code_model_id': 'm'})
    assert len(tasks) == 1


def test_generated_sprites_must_actually_be_drawn(tmp_path):
    """A run generated six sprites and referenced none of them, so the art was paid for and thrown
    away. It is reported every time - it is a fact about files on disk, not a keyword guess - but it
    does not fail a release: the game runs, and blocking on it spent a whole rethink budget on "you
    did not use art you paid for" and then shipped with the finding open anyway. The place it is
    worth acting on is the write result, where the agent still has calls left."""
    from game_studio.agents import static_qa
    from game_studio.agent_tools import write_game_file
    game = ('<!doctype html><html><canvas></canvas><script>requestAnimationFrame(()=>{});'
            'addEventListener("keydown",e=>({KeyW:1,KeyA:1,KeyS:1,KeyD:1})[e.code]);'
            'let score=0;function restart(){}</script></html>')

    assert static_qa(game, []).status == 'pass'
    unused = static_qa(game, ['ship-small.png', 'reef.png'])
    assert unused.status == 'pass', 'waste is not breakage'
    assert any('ship-small.png' in f and 'reef.png' in f for f in unused.findings)

    used = game.replace('let score=0;', "let score=0;const s=new Image();s.src='assets/ship-small.png';")
    assert static_qa(used, ['ship-small.png']).status == 'pass'

    # The write tool reports it without waiting for a QA round trip.
    (tmp_path / 'assets').mkdir()
    (tmp_path / 'assets' / 'reef.png').write_bytes(b'PNG')
    concept = default_concept('t').model_dump()
    state = {'concept': concept, 'output_dir': str(tmp_path), 'workspace_dir': str(tmp_path)}
    result = write_game_file.invoke({'html': game, 'state': state})
    assert 'reef.png' in result and 'drawImage' in result


def test_advisory_naming_does_not_block_a_playable_game():
    """"Missing restart" over a working reset() button used to cost a repair and a rethink cycle."""
    from game_studio.agents import static_qa
    game = ('<!doctype html><html><canvas></canvas><script>requestAnimationFrame(()=>{});'
            'addEventListener("keydown",e=>({KeyW:1,KeyA:1,KeyS:1,KeyD:1})[e.code]);'
            'let combo=0;function reset(){}</script></html>')
    assert static_qa(game).status == 'pass'
    # Something that genuinely stops the game working still blocks.
    assert static_qa(game.replace('requestAnimationFrame(()=>{});', '')).status == 'repair'


def test_concept_records_the_real_games_it_borrows_from():
    concept = GameConcept(title='t', elevator_pitch='p', player_goal='g', controls=['좌', '우'],
                          core_loop=['이동', '회피', '점수'], difficulty_curve='d',
                          visual_direction='v',
                          reference_games=['Flappy Bird: 한 버튼 상승, 중력 하강'])
    assert concept.reference_games[0].startswith('Flappy Bird')
    # Optional, so an existing concept without references still validates.
    assert GameConcept.model_validate({k: v for k, v in concept.model_dump().items()
                                       if k != 'reference_games'}).reference_games == []


def test_genre_picks_its_own_reference_games(monkeypatch):
    """A 퍼즐 request should anchor on puzzle loops, not on whatever the model finds interesting."""
    from game_studio import agents
    from game_studio.agents import genre_references
    from game_studio.prompts import GENRE_REFERENCES

    def rows(text):
        return [line.removeprefix('- ') for line in text.splitlines()]

    puzzle = rows(genre_references('Requested genre: 퍼즐\nPlayer brief: x'))
    assert puzzle and set(puzzle) <= set(GENRE_REFERENCES['퍼즐'])
    racing = rows(genre_references('Requested genre: 레이싱\nPlayer brief: x'))
    assert racing and set(racing) <= set(GENRE_REFERENCES['레이싱'])
    assert not set(puzzle) & set(racing), 'one genre must not leak exemplars into another'
    # 물리 퍼즐 contains 퍼즐 as a substring; the longer label has to win the lookup.
    physics = rows(genre_references('Requested genre: 물리 퍼즐\nPlayer brief: x'))
    assert set(physics) <= set(GENRE_REFERENCES['물리 퍼즐'])
    # 자동 기획 and 커스텀 have no single genre to anchor on, and "x" describes nothing, so
    # nothing is injected rather than something arbitrary.
    assert genre_references('Requested genre: 자동 기획\nPlayer brief: x') == ''

    prompts = []
    monkeypatch.setattr(agents, '_structured',
                        lambda schema, system, user, *a, **kw: prompts.append(user) or default_concept('c'))
    agents.create_concept('Requested genre: 슈팅\nPlayer brief: x', True, 'm')
    shown = [entry for entry in GENRE_REFERENCES['슈팅'] if entry in prompts[0]]
    assert len(shown) == agents.GENRE_REFERENCE_SAMPLE, 'the chosen genre has to reach the prompt'
    assert '장르를 섞거나' in prompts[0], 'it must also be told not to drift out of the genre'


def test_the_run_seed_picks_which_exemplars_not_only_which_genre():
    """Six rows of three fixed games meant auto planning had six outcomes, and inside one genre it
    opened from the same three exemplars every time. The seed now reaches one level further down."""
    from game_studio.agents import GENRE_REFERENCE_SAMPLE, sample_references
    from game_studio.prompts import GENRE_REFERENCES

    trios = {tuple(sample_references('퍼즐', f'Run seed: {n}')) for n in range(20)}
    assert len(trios) > 1, 'a different run seed has to bring different exemplars'
    assert all(len(trio) == GENRE_REFERENCE_SAMPLE for trio in trios)
    assert all(set(trio) <= set(GENRE_REFERENCES['퍼즐']) for trio in trios)

    # Same seed, same trio - a run has to be reproducible, which is the whole point of seeding it.
    assert sample_references('퍼즐', 'Run seed: fixed') == sample_references('퍼즐', 'Run seed: fixed')

    # A game the player named survives the shuffle. Asked for a faithful Tetris, the one thing that
    # must not happen is Tetris being shuffled out and Bejeweled offered in its place.
    for seed in range(20):
        picked = sample_references('퍼즐', f'Run seed: {seed}', '테트리스랑 똑같이 만들어줘')
        assert any(entry.startswith('Tetris') for entry in picked), seed


def test_broken_javascript_is_caught_when_node_is_available():
    """Without a parser, static QA passed games that could not even run: the only other check was
    the paid design review, which reads for requirements rather than parsing."""
    import shutil
    from game_studio.agents import static_qa
    if not shutil.which('node'):
        pytest.skip('node is not installed in this environment')
    broken = ('<!doctype html><html><canvas></canvas><script>requestAnimationFrame(()=>{});'
              'addEventListener("keydown",e=>({KeyW:1,KeyA:1,KeyS:1,KeyD:1})[e.code]);'
              'let score=0;function restart(){score=0;function update(d){score+=d;}</script></html>')
    report = static_qa(broken)
    assert report.status == 'repair'
    assert any('syntax error' in f for f in report.findings)

    # A <script type="module"> legitimately uses export; reporting that would be a false failure.
    module = broken.replace('function restart(){score=0;function update(d){score+=d;}',
                            'function restart(){}</script><script type="module">export const x=1;')
    assert not any('syntax error' in f for f in static_qa(module).findings)


def test_static_qa_is_cached_per_content():
    """The code agent calls run_static_qa repeatedly inside its loop, often on an unchanged draft,
    and each of those was paying a ~300ms node spawn - the whole cost of the check."""
    import time
    from game_studio.agents import static_qa
    game = ('<!doctype html><html><canvas></canvas><script>requestAnimationFrame(()=>{});'
            'addEventListener("keydown",e=>({KeyW:1,KeyA:1,KeyS:1,KeyD:1})[e.code]);'
            'let score=0;function restart(){}</script></html>')
    first = static_qa(game)
    started = time.monotonic()
    again = static_qa(game)
    assert (time.monotonic() - started) < 0.05, 'a repeat check must not respawn node'
    assert again.status == first.status and again.findings == first.findings
    # Callers mutate the report (qa_node rewrites status/findings), so the cache must hand out copies.
    again.findings.append('mutated')
    assert 'mutated' not in static_qa(game).findings
    # Different sprite sets are different questions, so they get different answers from the cache.
    assert any('ship.png' in f for f in static_qa(game, ['ship.png']).findings)
    assert static_qa(game, []).findings == []


def test_every_stage_defaults_to_sonnet_4_6(monkeypatch):
    """One default, used by the dashboard, the CLI and every pipeline stage."""
    from game_studio.agents import DEFAULT_MODEL_ID
    from game_studio.server import BEDROCK_MODELS, CreateRun
    expected = 'global.anthropic.claude-sonnet-4-6'
    assert DEFAULT_MODEL_ID == expected
    run = CreateRun()
    assert run.model_id == expected and run.code_model_id == expected
    assert run.model_id in BEDROCK_MODELS, 'the default must survive request validation'
    # The dashboard preselects the first entry, so the recommended model has to lead the list.
    assert next(iter(BEDROCK_MODELS)) == expected
    # And it has to be priced, or the cost estimate silently under-reports.
    from game_studio.agents import price_per_mtok
    assert price_per_mtok(expected) is not None

    # No env override leaves a stage behind on an older model.
    for var in ('BEDROCK_MODEL_ID', 'BEDROCK_DIRECTOR_MODEL_ID', 'BEDROCK_QA_MODEL_ID'):
        monkeypatch.delenv(var, raising=False)
    from game_studio.agents import _model
    assert _model(None).model_id == expected


def test_the_supervisor_walks_the_production_ladder_without_paying_for_it():
    """Only two visits cost a model call - opening the run and deciding what a failed verification
    costs next. Every other hop through the hub is a lookup, or the graph would pay a supervisor
    round trip between every pair of stages."""
    import game_studio.graph as gm
    for stage, expected in (('idea', 'design_document'), ('design_document', 'approval'),
                            ('art', 'code'), ('code', 'qa'), ('repair', 'qa')):
        assert gm.supervisor_node({'stage': stage})['next_step'] == expected

    approved = {'stage': 'approval', 'approval': {'decision': 'approved'}}
    assert gm.supervisor_node(approved)['next_step'] == 'art'
    assert gm.supervisor_node({**approved, 'approval': {'decision': 'rejected'}})['next_step'] == 'rejected'
    assert gm.supervisor_node({'stage': 'qa', 'qa': {'status': 'pass'}})['next_step'] == 'package'


def test_the_supervisor_cannot_spend_a_budget_the_run_does_not_have():
    """Its choice is clamped in code, not trusted to the prompt: a repair only while one is
    unspent, a fresh code loop only while rethink cycles remain, art re-planning only once."""
    from game_studio.graph import (MAX_REPAIR_ATTEMPTS, MAX_RETHINK_CYCLES, _affordable_actions,
                                   _fallback_action)
    fresh = {'repair_attempts': 0, 'rethink_cycles': 0}
    assert set(_affordable_actions(fresh)) == {'code', 'art', 'repair'}
    assert 'art' not in _affordable_actions({**fresh, 'art_revised': True})
    # Against the configured budget, not a number written in here: the constants are env-tunable
    # and were raised once already, and a test that pins the old value tests the old value.
    assert 'repair' in _affordable_actions({**fresh, 'repair_attempts': MAX_REPAIR_ATTEMPTS - 1})
    assert 'repair' not in _affordable_actions({**fresh, 'repair_attempts': MAX_REPAIR_ATTEMPTS})
    assert 'code' in _affordable_actions({**fresh, 'rethink_cycles': MAX_RETHINK_CYCLES - 1})
    assert 'code' not in _affordable_actions({**fresh, 'rethink_cycles': MAX_RETHINK_CYCLES})
    assert _affordable_actions({'repair_attempts': 99, 'rethink_cycles': 99}) == []

    # repair is a single model call with no tools: it can rewrite HTML and nothing else. So when the
    # supervisor names a move it cannot afford, a sprite that was never drawn still falls back to
    # the only worker that can generate and wire one up, and a text-only fault to the cheap rewrite.
    art = ['Generated sprites are never drawn: ship.png, reef.png.']
    text = ['Missing keyboard controls.']
    assert _fallback_action(art, ['code', 'repair']) == 'code'
    assert _fallback_action(text, ['code', 'repair']) == 'repair'
    assert _fallback_action(art, ['repair']) == 'repair', 'it still takes whatever is left'


def _supervise(monkeypatch, state, decision):
    """Run the supervisor over a failed QA with a canned decision. Returns (update, prompt)."""
    import game_studio.graph as gm
    captured = {}

    def structured(schema, system, user, model_id=None, **kw):
        captured.update(schema=schema, system=system, user=user)
        return decision

    monkeypatch.setattr(gm, '_structured', structured)
    return gm.supervisor_node(state), captured


def _failed_qa_state(tmp_path, concept, **overrides):
    art = default_art(concept).model_dump()
    art['asset_plan'] = ['player: 우주선']
    return {'stage': 'qa', 'concept': concept.model_dump(), 'art': art, 'implementation_plan': {},
            'qa': {'status': 'repair', 'findings': ['Generated sprites are never drawn: ship.png']},
            'output_dir': str(tmp_path), 'workspace_dir': str(tmp_path), 'generate_images': True,
            'game_html': '<html></html>', 'trace_notes': [], **overrides}


def test_the_supervisor_decides_on_the_findings_and_the_art_that_already_exists(tmp_path, monkeypatch):
    """So it can say "draw the sprite you already have" instead of paying to generate another."""
    from game_studio.models import SupervisorDecision
    (tmp_path / 'assets').mkdir()
    (tmp_path / 'assets' / 'ship.png').write_bytes(b'PNG')
    out, captured = _supervise(
        monkeypatch, _failed_qa_state(tmp_path, default_concept('t')),
        SupervisorDecision(action='code', reason='생성된 스프라이트가 그려지지 않습니다',
                           instructions='1. ship.png 를 플레이어에 그리세요'))
    assert captured['schema'] is SupervisorDecision
    assert 'generate_comfyui_image' in captured['system'], 'art has to stay one of its moves'
    assert 'ship.png' in captured['user'], 'and it must know what has already been generated'
    assert 'Raster generation enabled: True' in captured['user']
    assert 'Actions still available to you' in captured['user'], 'the budget is part of the decision'
    # The code agent gets a fresh budget and a cleared html so QA re-reads what it writes next.
    assert out['next_step'] == 'code' and out['rethink_cycles'] == 1
    assert out['tool_iterations'] == 0 and out['game_html'] == ''
    assert out['qa_guidance'] == '1. ship.png 를 플레이어에 그리세요'


def test_art_replanning_is_offered_once_and_then_taken_off_the_menu(tmp_path, monkeypatch):
    """art_node used to run exactly once per run with no way back, so a plan that was missing an
    object the game needed stayed wrong for the rest of the run. The supervisor can send it back -
    but only once, because re-planning every cycle pays a model call to hear the same plan again."""
    from game_studio.models import SupervisorDecision
    concept = default_concept('t')
    replan = SupervisorDecision(action='art', reason='에셋 계획에 빠진 객체가 있습니다',
                                instructions='1. 플레이어 스프라이트를 계획에 추가하세요')

    first, _ = _supervise(monkeypatch, _failed_qa_state(tmp_path, concept), replan)
    assert first['next_step'] == 'art' and first['art_revision_needed'] is True

    # Already re-planned once, so the same answer is no longer affordable and coding gets the work.
    second, captured = _supervise(
        monkeypatch, _failed_qa_state(tmp_path, concept, art_revised=True), replan)
    assert second['next_step'] == 'code' and second['art_revision_needed'] is False
    assert 'Actions still available to you: ["code", "repair"]' in captured['user']

    # With nothing left to spend the run ends with a verdict, and without a model call to say so.
    spent = _failed_qa_state(tmp_path, concept, repair_attempts=9, rethink_cycles=9)
    assert _supervise(monkeypatch, spent, replan)[0] == {'next_step': 'abandoned'}


def test_an_unanswerable_decision_falls_back_instead_of_losing_an_approved_run(tmp_path, monkeypatch):
    """The escalation is the last thing that happens to an approved run, so a model that never
    manages to fill the schema must not be what ends it."""
    import game_studio.graph as gm
    monkeypatch.setattr(gm, '_structured',
                        lambda *a, **kw: (_ for _ in ()).throw(RuntimeError('no structured output')))
    out = gm.supervisor_node(_failed_qa_state(tmp_path, default_concept('t')))
    assert out['next_step'] == 'code', 'an art finding still reaches the tool-capable worker'
    assert 'ship.png' in out['qa_guidance'], 'and the findings themselves become the instructions'


def test_replanned_art_is_told_what_to_fix(tmp_path, monkeypatch):
    import game_studio.graph as gm
    (tmp_path / 'assets').mkdir()
    (tmp_path / 'assets' / 'ship.png').write_bytes(b'PNG')
    seen = {}
    concept = default_concept('t')
    monkeypatch.setattr(gm, 'create_art',
                        lambda c, use_llm, model_id=None, findings=None, existing_sprites=None:
                        seen.update(findings=findings, sprites=existing_sprites) or default_art(c))
    out = gm.art_node({'concept': concept.model_dump(), 'output_dir': str(tmp_path),
                       'workspace_dir': str(tmp_path), 'art_revision_needed': True,
                       'qa': {'findings': ['Generated sprites are never drawn: ship.png']},
                       'trace_notes': []})
    assert seen['findings'] == ['Generated sprites are never drawn: ship.png']
    assert seen['sprites'] == ['ship.png'], 'it must know what already exists'
    # The flag is consumed so the run cannot loop through art planning forever.
    assert out['art_revision_needed'] is False and out['art_revised'] is True


def test_an_auto_genre_run_is_assigned_a_genre_instead_of_being_left_open(monkeypatch):
    """"자동 기획" used to mean the idea agent got no genre - and, because the reference table is
    keyed by genre, no exemplars either. It was the one mode with nothing to anchor on, which
    sounds like freedom and is the opposite: left with only the standing constraints (one Canvas
    file, a 60-120 second session, a score to compete against, passive play must lose, borrow a
    loop that is known to work), the model converges on the one design that satisfies all of them.
    Three consecutive auto runs came back as the same falling-object shooter, two with the same
    title.
    """
    from game_studio.agents import genre_references, resolve_auto_genre
    from game_studio.prompts import GENRE_REFERENCES

    def auto(seed):
        return ('Requested genre: 자동 기획\n'
                'Player brief: 사용자 경험 없이 독자적으로 기획하세요.\n'
                f'Run seed: {seed}')

    # Every assignment is a genre the reference table can actually anchor.
    picked = {resolve_auto_genre(auto(f'seed{n}')) for n in range(40)}
    assert picked <= set(GENRE_REFERENCES)
    assert len(picked) >= 4, f'the seed has to spread across genres, got {picked}'
    assert all(genre_references(g) for g in picked), 'an assigned genre must bring its exemplars'

    # The run seed was always meant to make runs differ and could not - it is a hex string in a
    # prompt, not a sampling seed. Now it selects, so it reproduces too.
    assert resolve_auto_genre(auto('fixed')) == resolve_auto_genre(auto('fixed'))

    # A player who chose a genre keeps it; nothing is assigned over the top.
    assert resolve_auto_genre('Requested genre: 퍼즐\nRun seed: x') == ''
    assert resolve_auto_genre('Requested genre: 로그라이크') == ''


def test_a_written_brief_chooses_the_genre_and_the_seed_does_not_overrule_it(monkeypatch):
    """자동 기획 is the default dropdown value, so it is also what a player leaves selected while
    typing the game they want. The seed used to assign a genre over the top of that description:
    a brief reading "블록을 회전시켜 빈틈없이 쌓는 게임" was planned as a 플랫포머 and handed Super
    Mario Bros as the loop to borrow. A description is a genre choice; it gets read, not overwritten.
    """
    from game_studio import agents

    brief = ('Requested genre: 자동 기획\n'
             'Player brief: 블록을 회전시켜 빈틈없이 쌓는 게임을 만들어줘\nRun seed: abc123')
    assert agents.resolve_auto_genre(brief) == '', 'the seed must not assign over a description'
    assert agents.genre_label(brief) == '퍼즐', 'and the description has to be read instead'

    captured = {}
    monkeypatch.setattr(agents, '_structured',
                        lambda schema, system, user, *a, **kw: captured.update(user=user)
                        or default_concept('x'))
    agents.create_concept(brief, True, 'm')
    assert 'Super Mario Bros' not in captured['user'], 'the wrong genre must not reach the prompt'
    assert '이번 실행에 배정된 장르' not in captured['user']
    # The list is offered as background here, not as a menu: told to pick from one, a model asked
    # for a faithful Tetris picks the nearest listed game instead of building what was described.
    assert '어긋나는 것은 무시하세요' in captured['user']
    assert '이 중에서 골라' not in captured['user']


def test_a_description_that_names_no_genre_attaches_no_exemplars():
    """Guessing wrong is worse than not guessing - a wrong row is exactly the failure above. Two
    loop words are a description; one is a coincidence, and none is silence."""
    from game_studio.agents import infer_genre

    assert infer_genre('재미있는 게임 하나 만들어줘') == ''
    assert infer_genre('') == ''
    # One stray word is not enough to pick a row on.
    assert infer_genre('점프해서 넘어가는 뭔가') == ''
    # Two together are.
    assert infer_genre('발판을 점프로 넘어가는 게임') == '플랫포머'
    # A named game decides outright, in either spelling, without needing a second word.
    assert infer_genre('팩맨 같은 거') == '미로 추격'
    assert infer_genre('make it like Flappy Bird') == '무한 러너'


def test_the_assigned_genre_reaches_the_idea_prompt_with_its_exemplars(monkeypatch):
    """Assigning a genre is only worth anything if the idea agent is told about it, and told not to
    wander off it - the whole point is to replace an empty anchor with a real one."""
    from game_studio import agents

    captured = {}
    monkeypatch.setattr(agents, '_structured',
                        lambda schema, system, user, *a, **kw: captured.update(user=user)
                        or default_concept('x'))
    # The exact sentence server.py writes for an empty brief box - a fixture that only paraphrases
    # it reads as a player request and takes the other branch entirely.
    brief = ('Requested genre: 자동 기획\n'
             'Player brief: 사용자 경험 없이 독자적으로 기획하세요.\nRun seed: fixed')
    agents.create_concept(brief, True, 'm')

    assigned = agents.resolve_auto_genre(brief)
    assert f'이번 실행에 배정된 장르: {assigned}' in captured['user']
    assert '다른 장르로 바꾸지 마세요' in captured['user']
    from game_studio.prompts import GENRE_REFERENCES
    shown = [entry for entry in GENRE_REFERENCES[assigned] if entry in captured['user']]
    assert len(shown) == agents.GENRE_REFERENCE_SAMPLE, \
        'the assigned genre has to pull in its reference games'


def test_a_tool_call_the_token_cap_cut_in_half_is_not_retried_identically():
    """The loop this ends, seen on a real run: four turns in a row spent writing the same oversized
    game.

        코드 Agent 모델 호출 · 토큰 예산을 모두 써서 답변이 끊겼습니다
        코드 Agent 도구 결과 write_game_file — html: Field required. Please fix the error and try again
        코드 Agent 도구 호출 write_game_file()          ← 인자가 비어 있음
        ... 같은 순서로 3번 더, 20호출 예산이 소진될 때까지

    When a turn hits the output ceiling part-way through a tool argument the JSON is cut off and the
    framework parses what is left as {}. The tool then answers with its own validation error, which
    describes the symptom and says nothing about the cause - so the model does the only thing that
    error suggests, and writes the same thing again.

    Raising CODE_MAX_TOKENS makes this rarer and cannot make it impossible: the ceiling is a limit
    and games have no upper bound. What ends the loop is saying what actually happened.
    """
    from game_studio.code_agent import CODE_MAX_TOKENS, StudioObservability

    observer = StudioObservability()

    class Request:
        tool_call = {'name': 'write_game_file', 'args': {}, 'id': 'c1'}

    def unreachable(_request):
        raise AssertionError('a call with no arguments must not reach the tool at all')

    # Not truncated: an empty call is a real mistake and the tool's own error is the right answer.
    observer.truncated = False
    called = []
    assert observer.wrap_tool_call(Request(), lambda r: called.append(r) or 'ok') == 'ok'
    assert called, 'without a truncation this has to behave exactly as before'

    observer.truncated = True
    answer = observer.wrap_tool_call(Request(), unreachable)
    assert answer.tool_call_id == 'c1' and answer.status == 'error'
    # The cause, not the symptom - and the number, so "shorter" means something.
    assert str(CODE_MAX_TOKENS) in answer.content
    assert '잘렸습니다' in answer.content
    assert '그대로 다시 보내면 똑같이 잘립니다' in answer.content
    # And a way out that is not "try again": write less, or repair a section instead of rewriting.
    assert 'repair_html' in answer.content and 'read_game_file' in answer.content

    # A truncated turn that still delivered its arguments is a normal call: the argument is short
    # enough to have arrived, and only the prose after it was cut.
    class Complete:
        tool_call = {'name': 'write_game_file', 'args': {'html': '<!doctype html>'}, 'id': 'c2'}

    assert observer.wrap_tool_call(Complete(), lambda r: 'ok') == 'ok'


def test_a_named_game_replaces_the_sample_instead_of_being_padded_out_to_three():
    """The failure, from a real run. Asked for "슈퍼마리오와 똑같은 게임", the concept came back with

        슈퍼 마리오 브라더스 — 횡스크롤 이동·점프·적 밟기·파워업·깃발 클리어 루프 전체를 그대로 차용
        동키콩 주니어 — 타이머 압박과 목숨 시스템 구조 참고
        버블보블 — 스테이지 단위 진행과 코인·아이템 수집 점수 구조 참고

    and each one arrived carrying mechanics. The contract then owed a timer-and-lives system and a
    stage-and-collectible scoring structure nobody asked for, on top of Mario - which is how these
    builds ran out of budget with the game still not starting. Padding a named request up to the
    sample size is the studio adding scope the player did not request.
    """
    from game_studio.agents import GENRE_REFERENCE_SAMPLE, sample_references

    named = sample_references('플랫포머', 'Run seed: abc', '슈퍼마리오와 똑같은 게임을 만들어줘')
    assert len(named) == 1 and named[0].startswith('Super Mario Bros')

    # With nothing named there is a real choice to make, and the seeded sample still makes it.
    free = sample_references('플랫포머', 'Run seed: abc', '재밌는 점프 게임')
    assert len(free) == GENRE_REFERENCE_SAMPLE

    # Two named games are two references - the rule is "what you named", not "exactly one".
    both = sample_references('퍼즐', 'Run seed: abc', '테트리스와 2048을 섞은 것 같은 게임')
    assert len(both) == 2


def test_a_game_is_recognised_however_the_player_spells_it():
    """Every spelling difference used to drop the named game out of the references entirely, and
    the run was handed three arbitrary same-genre games in its place. The table says "슈퍼 마리오"
    and the brief said "슈퍼마리오"; the table says "Plants vs. Zombies" and nobody types the stop."""
    from game_studio.agents import named_in_brief

    def only(brief):
        found = named_in_brief(brief)
        assert len(found) == 1, f'{brief} → {found}'
        return found[0].split(':')[0]

    assert only('슈퍼마리오와 똑같은 게임') == 'Super Mario Bros(슈퍼 마리오)'   # 띄어쓰기 없음
    assert only('슈퍼 마리오 브라더스처럼') == 'Super Mario Bros(슈퍼 마리오)'
    assert only('super mario 같은 거') == 'Super Mario Bros(슈퍼 마리오)'      # 줄여 부름
    assert only('pac man 처럼') == 'Pac-Man(팩맨)'                          # 하이픈 없음
    assert only('plants vs zombies 처럼') == 'Plants vs. Zombies(식물 대 좀비)'  # 마침표 없음
    assert only('2048 똑같이') == '2048'

    # And none of that may start matching things that are not games. A short name is matched as a
    # whole word for exactly this reason - "N++" normalises to "n", "uno" hides inside "unorthodox".
    for innocent in ('world of warcraft 같은 거', 'unorthodox한 게임', 'n개의 스테이지가 있는 게임',
                     '동전을 모으는 게임', '재밌는 플랫포머 하나'):
        assert named_in_brief(innocent) == [], innocent


def test_a_revision_tells_the_code_agent_to_fix_the_game_not_rewrite_it():
    """A revision runs against a game that already ships, so the player has actually played it -
    which is more than any check in this pipeline has done. That outranks the contract: the
    contract is what they were promised, this is what they think of it.

    And the instruction that matters most is what NOT to touch. Handed a request and a contract,
    an agent will happily rebuild the whole game from the contract and call it a revision.
    """
    from game_studio.graph import _code_task

    plan = ImplementationPlan(genre='플랫포머', mechanics=['달린다', '점프한다', '적을 밟는다'],
                              win_condition='깃발', loss_condition='낙하',
                              state_transitions=['시작', '진행', '결과'],
                              acceptance_tests=['t1', 't2', 't3'])
    concept = default_concept('b')
    base = {'brief': 'b', 'concept': concept.model_dump(), 'art': default_art(concept).model_dump(),
            'implementation_plan': plan.model_dump(), 'approval': {'comment': ''}}

    plain = _code_task(base)
    assert '플레이어가 완성된 게임을 직접 해 보고' not in plain, 'a first build has no revision'

    revised = _code_task({**base, 'revision_request': '점프가 너무 무겁습니다. 가볍게 해주세요.'})
    assert '점프가 너무 무겁습니다' in revised
    assert '이 요청이 위의 계약보다 우선합니다' in revised
    assert '처음부터 다시 쓰지 말고 읽어서' in revised
    assert '요청과 무관한 부분은 그대로' in revised
    # The contract still travels - a revision is a change to this game, not a replacement for it.
    assert '적을 밟는다' in revised
