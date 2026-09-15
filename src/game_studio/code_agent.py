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
    ModelRetryMiddleware,
    TodoListMiddleware,
)
from langchain.agents.middleware.context_editing import ClearToolUsesEdit

from .agents import CURRENT_STEP, StreamAccumulator, _content_text, _model, stream_turn

# One model call per tool round trip. Image generation spends from the same budget as writing and
# repairing the game, so a tight limit quietly starves the art: a measured run generated 4 sprites
# and then had no calls left to wire them in. Raising it costs tokens, so it is tunable.
MODEL_CALL_LIMIT = int(os.getenv("CODE_AGENT_MODEL_CALLS", "20"))
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
            bound = model.bind_tools(tools) if tools else model
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
        if turn.truncated():
            _emit({"kind": "model_text", "agent": self.agent_name,
                   "text": "토큰 예산을 모두 써서 답변이 끊겼습니다. 도구 인자가 잘렸을 수 있습니다."})
        if not answer.tool_calls and (text := _content_text(answer).strip()):
            _emit({"kind": "model_text", "agent": self.agent_name, "text": _trim(text, 400)})
        return answer

    def _show(self, turn: StreamAccumulator) -> None:
        """Forward a snapshot of the turn being written: prose so far, plus the tool call being
        composed. Rate-limited by stream_turn, not called per token."""
        if preview := turn.preview():
            _emit({"step": self.step, "text": preview})

    def wrap_tool_call(self, request, handler):
        call = getattr(request, "tool_call", None) or {}
        name = call.get("name", "tool")
        args = {key: _trim(value) for key, value in (call.get("args") or {}).items()}
        _emit({"kind": "tool_call", "agent": self.agent_name, "name": name, "args": args})
        result = handler(request)
        content = getattr(result, "content", "")
        if isinstance(content, list):
            content = " ".join(str(part) for part in content)
        _emit({"kind": "tool_result", "agent": self.agent_name, "name": name,
               "text": _trim(content, 400)})
        return result


CODE_MAX_TOKENS = 16000


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
