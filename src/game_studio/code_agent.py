"""The code agent, built with create_agent instead of a hand-rolled tool loop.

What used to be `code_node` + `ToolNode` + `route_after_code_agent` in graph.py is now one
`create_agent` graph. The behaviours that loop was carrying by hand map onto middleware:

    tool_iterations budget          -> ModelCallLimitMiddleware
    transient-failure retry         -> ModelRetryMiddleware
    history compaction              -> ContextEditingMiddleware(ClearToolUsesEdit)
    step attribution / model log    -> StudioObservability.wrap_model_call
    tool call + result log          -> StudioObservability.wrap_tool_call
    planning across a long build    -> TodoListMiddleware

The outer graph still owns orchestration - the supervisor node routes approval, QA, repair and its
own rethink cycles; this module owns only the write/inspect/repair loop.
"""

from __future__ import annotations

import os
from typing import Any

from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import (
    AgentMiddleware,
    ContextEditingMiddleware,
    ModelCallLimitMiddleware,
    ModelFallbackMiddleware,
    ModelRetryMiddleware,
    TodoListMiddleware,
)
from langchain.agents.middleware.context_editing import ClearToolUsesEdit
from langchain_core.messages import ToolMessage
from langsmith import traceable

from .agents import (
    CURRENT_STEP,
    StreamAccumulator,
    _content_text,
    _model,
    cache_control_for,
    daily_cap_fallback,
    model_fallbacks,
    stream_turn,
)

# One model call per tool round trip, and the hardest ceiling in the pipeline: whatever is on disk
# when these run out is what ships. Image generation spends from the same budget as writing and
# repairing the game, so a tight limit quietly starves the art too - a measured run generated 4
# sprites and then had no calls left to wire them in.
#
# Raised 20 -> 50 when the per-run budget grew, then 50 -> 70. At 20 the agent finished neither
# half: a Mario-like build spent everything on ? blocks and flagpole scoring and shipped a game that
# never started. The build order now puts the playable core first (see CODE_SYSTEM), so extra calls
# go into finishing and verifying mechanics rather than into starting more of them.
#
# What this number is NOT is a cost control. On a committed-capacity account there is no per-token
# billing, so the scarce resource is the account's shared DAILY TOKEN QUOTA - the one that killed
# three runs in a single morning. This ceiling exists to stop a runaway repair loop from spending
# that quota and locking out every other run for the day. Nothing else.
#
# So it is set far above real work rather than near it. Measured across every delivered game, the
# code agent used 9-21 calls, median 14 - not one came close to 50. What actually killed builds was
# the recursion limit below, which worked out to 24 calls and sat just above that working range;
# see CODE_RECURSION_LIMIT for why the two must be derived from each other rather than written down
# separately.
#
# Running out is not a failure. exit_behavior="end" stops the loop cleanly and the run carries on
# into QA, repair and packaging with whatever is on disk.
MODEL_CALL_LIMIT = int(os.getenv("CODE_AGENT_MODEL_CALLS", "70"))
# Super-steps the inner agent graph spends per model call - before_model, the model itself, the
# after_model hooks and the tool node.
#
# Measured exactly, on the run that exposed this: its `code` namespace stopped at super-step 120
# with 24 model calls recorded, so 5.0. The constant is deliberately DOUBLE that, and the asymmetry
# is the point. This number does not decide when the build stops - ModelCallLimitMiddleware does,
# at MODEL_CALL_LIMIT, cleanly, leaving a playable game. The recursion limit is only a backstop for
# the case where that middleware never fires at all.
#
# So the two errors are not the same size. Too high costs nothing: the budget still stops the loop
# first, hundreds of steps earlier. Too low is the bug this whole constant exists to prevent - at a
# flat 120 against a budget of 50, the clean stop could NEVER fire, every long build died of
# GraphRecursionError instead, and one of them discarded a 25KB game that was already on disk.
# A per-turn cost that varies with the tools called and with whether context editing runs is
# exactly the kind of thing a measured average understates, so it is padded rather than trusted.
STEPS_PER_MODEL_CALL = 10
# Room for the closing turn after the budget's own stop, which still has to run its hooks.
RECURSION_HEADROOM = 20
CODE_RECURSION_LIMIT = MODEL_CALL_LIMIT * STEPS_PER_MODEL_CALL + RECURSION_HEADROOM


# The draft game runs past 20,000 characters. Left alone it sits in the history as a
# write_game_file argument and again as a read_game_file result, and gets re-sent on every later
# turn; one measured run spent 265k input tokens in this loop alone. Clearing tool inputs and old
# results is what keeps that flat - the file is on disk, so the agent re-reads what it needs.
#
# The trigger is deliberately below one game's worth of tokens. A measured run of a 16.5KB game
# (~5.5k tokens) averaged 10.8k input tokens per call with the trigger at 20k, because the file was
# allowed to accumulate two or three times over before anything was cleared.
CONTEXT_EDIT_TRIGGER_TOKENS = 8000
KEEP_RECENT_TOOL_RESULTS = 2


class CodeAgentState(AgentState):
    """Agent state extended with the studio fields the game tools read via InjectedState."""

    brief: str
    concept: dict[str, Any]
    art: dict[str, Any]
    implementation_plan: dict[str, Any]
    approval: dict[str, Any]
    qa_guidance: str
    output_dir: str
    workspace_dir: str
    generate_images: bool
    required_assets: list[str]
    engine: str
    model_id: str
    code_model_id: str


def _emit(payload: dict[str, Any]) -> None:
    """Best-effort write onto the run's custom stream, which the dashboard reads."""
    try:
        from langgraph.config import get_stream_writer

        get_stream_writer()(payload)
    except Exception:
        return


def _trim(value: object, limit: int = 300) -> str:
    text = value if isinstance(value, str) else str(value)
    return text if len(text) <= limit else f"{text[:limit]}…({len(text)}자)"


# The code agent is the most expensive stage in the pipeline and, until these, the most opaque: the
# outer graph traces it as one "autonomous-code-agent" span covering up to twenty model calls and
# every tool round trip inside them. A slow build looked like a slow node with nothing to point at.
#
# @traceable costs about 37us per call even with tracing switched off (measured: 20k calls in 752ms
# against 2ms plain), so it belongs on per-turn and per-tool boundaries and must stay out of the
# per-chunk paths - StreamAccumulator.add runs thousands of times for one game.
def _trace_model_inputs(inputs: dict) -> dict:
    """What is worth recording about a turn. The middleware object and the full message history are
    not: one does not serialise and the other is the whole game, repeatedly."""
    request = inputs.get("request")
    messages = list(getattr(request, "messages", []) or [])
    return {
        "model": getattr(getattr(request, "model", None), "model_id", ""),
        "tools": [getattr(t, "name", str(t)) for t in (getattr(request, "tools", None) or [])],
        "message_count": len(messages),
    }


def _trace_model_output(answer) -> dict:
    usage = getattr(answer, "usage_metadata", None) or {}
    return {
        "tool_calls": [call.get("name") for call in (getattr(answer, "tool_calls", None) or [])],
        "input_tokens": usage.get("input_tokens"),
        "output_tokens": usage.get("output_tokens"),
        "cache_read": (usage.get("input_token_details") or {}).get("cache_read"),
        "stop_reason": (getattr(answer, "response_metadata", None) or {}).get("stopReason"),
    }


def _trace_tool_inputs(inputs: dict) -> dict:
    call = getattr(inputs.get("request"), "tool_call", None) or {}
    return {"tool": call.get("name", "tool"),
            "args": {key: _trim(value, 200) for key, value in (call.get("args") or {}).items()}}


class StudioObservability(AgentMiddleware):
    """Reports what the agent is doing, using hooks instead of call-site instrumentation.

    wrap_model_call runs the model itself in streaming mode so a long generation shows progress
    while it is still being written, and tags token usage with the step that is really executing.
    wrap_tool_call reports each tool call and its result, which the dashboard previously had to
    reconstruct from graph update events.
    """

    state_schema = CodeAgentState

    def __init__(self, agent_name: str = "코드 Agent", step: str = "code") -> None:
        super().__init__()
        self.agent_name = agent_name
        self.step = step
        # Whether the turn that produced the pending tool calls ran out of output budget. Carried
        # across the two hooks because only wrap_model_call can see it and only wrap_tool_call can
        # act on it - see there for why the difference matters.
        self.truncated = False

    @traceable(name="code-agent-turn", run_type="llm",
               process_inputs=_trace_model_inputs, process_outputs=_trace_model_output)
    def wrap_model_call(self, request, handler):
        CURRENT_STEP.set(self.step)
        model = getattr(request, "model", None)
        _emit({"kind": "model_call", "agent": self.agent_name,
               "model": getattr(model, "model_id", "") or ""})
        messages = list(getattr(request, "messages", []) or [])
        system = getattr(request, "system_message", None)
        if system is not None:
            messages = [system, *messages]
        tools = getattr(request, "tools", None)
        try:
            # The loop's whole point is many calls over one stable prefix, so this is where the
            # cache pays most: the system prompt and every tool definition are byte-identical on
            # each of the turns after the first.
            caching = cache_control_for(getattr(model, "model_id", ""))
            bound = model.bind_tools(tools, **caching) if tools else model
            # This is the longest generation in the pipeline - an entire game arrives as one tool
            # argument - so how the stream is accumulated is not a detail here. See
            # StreamAccumulator: merging chunk by chunk re-parsed the whole half-written game on
            # every token and cost minutes of CPU per build on its own.
            turn = stream_turn(bound, messages, on_preview=self._show)
            if turn.empty:
                raise RuntimeError("빈 스트림")
        except Exception:
            # Streaming is only there for visibility. If it cannot be done, fall back to the
            # agent's own call rather than failing the build over a progress feature.
            return handler(request)
        answer = turn.finish()
        self.truncated = turn.truncated()
        if self.truncated:
            _emit({"kind": "model_text", "agent": self.agent_name,
                   "text": f"출력 예산({CODE_MAX_TOKENS} 토큰)을 모두 써서 답변이 끊겼습니다. "
                           "도구 인자가 잘렸을 수 있습니다."})
        if not answer.tool_calls and (text := _content_text(answer).strip()):
            _emit({"kind": "model_text", "agent": self.agent_name, "text": _trim(text, 400)})
        return answer

    def _truncated_call(self, call: dict, name: str) -> ToolMessage | None:
        """Answer a tool call whose arguments never arrived, instead of letting it fail blind.

        When a turn hits the output ceiling part-way through a tool argument, the JSON is cut off
        and the framework parses what is left as {}. The tool then answers with its own validation
        error - "html: Field required. Please fix the error and try again" - which describes the
        symptom and says nothing about the cause, so the model does the only thing that error
        suggests: it writes the same oversized game again. One run spent four of its twenty calls
        on identical truncated writes and ended with nothing on disk.

        Raising CODE_MAX_TOKENS makes this rarer; it cannot make it impossible, because the ceiling
        is a limit and games have no upper bound. What ends the loop is saying what happened.
        """
        if call.get("args") or not self.truncated:
            return None
        _emit({"kind": "model_text", "agent": self.agent_name,
               "text": f"{name} 호출이 잘려 인자가 비었습니다. 더 짧게 쓰도록 되돌려보냅니다."})
        return ToolMessage(
            tool_call_id=call.get("id", ""),
            name=name,
            status="error",
            content=(
                f"{name} 호출이 실행되지 않았습니다. 인자가 비어 있습니다 — 내용이 틀린 게 아니라 "
                f"이번 답변이 출력 한도({CODE_MAX_TOKENS} 토큰)에 걸려 중간에 잘렸습니다.\n"
                "같은 내용을 그대로 다시 보내면 똑같이 잘립니다. 더 짧게 만들어 다시 호출하세요:\n"
                "1) 없어도 게임이 성립하는 것부터 빼세요 (장식용 파티클, 여러 단계 레벨 데이터, "
                "긴 주석, 중복된 헬퍼).\n"
                "2) 좌표·맵·패턴을 긴 리터럴로 적지 말고 코드로 생성하세요.\n"
                "3) 이미 저장된 초안을 고치는 중이라면 전체를 다시 쓰지 말고, "
                "read_game_file(outline=True)로 위치를 찾은 뒤 repair_html로 그 구간만 고치세요."
            ),
        )

    def _show(self, turn: StreamAccumulator) -> None:
        """Forward a snapshot of the turn being written: prose so far, plus the tool call being
        composed. Rate-limited by stream_turn, not called per token."""
        if preview := turn.preview():
            _emit({"step": self.step, "text": preview})

    @traceable(name="code-agent-tool", run_type="tool", process_inputs=_trace_tool_inputs)
    def wrap_tool_call(self, request, handler):
        call = getattr(request, "tool_call", None) or {}
        name = call.get("name", "tool")
        args = {key: _trim(value) for key, value in (call.get("args") or {}).items()}
        _emit({"kind": "tool_call", "agent": self.agent_name, "name": name, "args": args})
        if (answer := self._truncated_call(call, name)) is not None:
            return answer
        result = handler(request)
        content = getattr(result, "content", "")
        if isinstance(content, list):
            content = " ".join(str(part) for part in content)
        _emit({"kind": "tool_result", "agent": self.agent_name, "name": name,
               "text": _trim(content, 400)})
        return result


# The output ceiling for one turn, and the largest in the pipeline: a whole game arrives as a
# single write_game_file argument, so this is not "how long may an answer be" but "how big may the
# game be". At 16k a run hit the ceiling mid-argument four times in a row - the JSON was cut off,
# the framework parsed the truncated call as {}, and every retry wrote the same oversized game
# again until the call budget was gone.
#
# Raising it costs nothing on a turn that does not use it - max_tokens is a limit, not a
# reservation - so the only real argument for keeping it low is discouraging sprawl, and a build
# that cannot finish its own file is a worse outcome than a long one. 32k is roughly 128KB of HTML,
# several times the largest game this pipeline has produced. Tunable, and Sonnet accepts 64000 if a
# run genuinely needs it.
CODE_MAX_TOKENS = int(os.getenv("CODE_MAX_TOKENS", "32000"))


def _fallback_middleware(model_id: str | None):
    """The same model through another inference profile, for when this one's quota is gone.

    The longest-running stage in the pipeline is also the one with the most to lose: fifty calls
    into a build, a throttled profile means the game on disk is half written. The profiles share
    weights and not quota pools, so asking through the other door is often the difference between
    a finished game and an abandoned one.

    The last entry is the exception, and a deliberate one. When the daily pool is gone no profile
    has room and the choice is not "a consistent game or a mixed one", it is "a mixed game or no
    game until tomorrow". Which models actually ran is recorded per call on the run's usage, so the
    manifest says so rather than leaving it to be noticed.
    """
    names = list(model_fallbacks(model_id))
    # Last, and only reached once every profile has refused: a different model, for the day the
    # account's Sonnet pool is spent. The profiles share that pool, so by the time the chain gets
    # here the earlier entries have each cost one fast failure - the price of not having to tell a
    # per-minute spike from a daily cap inside middleware that cannot see the error.
    if cap_model := daily_cap_fallback(model_id):
        names.append(cap_model)
    alternates = [_model(name, max_tokens=CODE_MAX_TOKENS) for name in names]
    return [ModelFallbackMiddleware(*alternates)] if alternates else []


def build_code_agent(model_id: str | None, tools, system_prompt: str, retry_on):
    """Assemble the code agent. Middleware order is outermost first."""
    return create_agent(
        model=_model(model_id, max_tokens=CODE_MAX_TOKENS),
        tools=tools,
        system_prompt=system_prompt,
        state_schema=CodeAgentState,
        middleware=[
            StudioObservability(),
            ModelCallLimitMiddleware(run_limit=MODEL_CALL_LIMIT, exit_behavior="end"),
            ModelRetryMiddleware(max_retries=2, retry_on=retry_on, on_failure="continue"),
            *_fallback_middleware(model_id),
            ContextEditingMiddleware(edits=[ClearToolUsesEdit(
                trigger=CONTEXT_EDIT_TRIGGER_TOKENS,
                keep=KEEP_RECENT_TOOL_RESULTS,
                clear_tool_inputs=True,
                placeholder="[생략 · read_game_file로 다시 읽으세요]",
            )]),
            TodoListMiddleware(),
        ],
        name="code_agent",
    )
