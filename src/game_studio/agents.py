"""Model-backed specialist adapters and deterministic syntax checks."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import random
import re
from pathlib import Path
import shutil
import subprocess
import time
from functools import lru_cache
from html.parser import HTMLParser
from collections.abc import Callable
from contextvars import ContextVar
from typing import TypeVar

import boto3
from botocore.config import Config as BotocoreConfig
from langchain_aws import ChatBedrockConverse
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.messages.tool import tool_call_chunk
from langsmith import traceable
from pydantic import ValidationError

from .models import ArtDirection, GameConcept, QAReport
from .required_art import unused_sprites
from .prompts import (
    ART_SYSTEM,
    DIRECTOR_SYSTEM,
    GENRE_KEYWORDS,
    GENRE_REFERENCES,
    IDEA_SYSTEM,
)

T = TypeVar("T")
# Inference-profile ID verified callable in this account. The creative stages default to it.
DEFAULT_MODEL_ID = "global.anthropic.claude-sonnet-4-6"
# Verification and escalation are judgement calls, not authoring: they read a finished game and
# answer with a short verdict. Sonnet spent minutes and thousands of output tokens writing essays
# about code it was only asked to check, and that audit sits directly between a finished build and
# a shipped game - so it is the one stage where latency is felt end to end. Haiku 4.5 answers the
# same question several times faster and at a fraction of the cost. BEDROCK_QA_MODEL_ID still wins.
DEFAULT_QA_MODEL_ID = "global.anthropic.claude-haiku-4-5-20251001-v1:0"


def qa_model_id(fallback: str | None = None) -> str:
    """The model that verifies and escalates, resolved from one place.

    The design review and the supervisor's escalation decision both use it, so a run cannot end up
    auditing on one model and deciding what to do about the audit on another.
    """
    return os.getenv("BEDROCK_QA_MODEL_ID", "").strip() or DEFAULT_QA_MODEL_ID or fallback or ""


# botocore defaults to a 60 second read timeout, which is far too short for this pipeline: the code
# agent writes a whole game as one tool argument with a 16k token budget, and a single long
# generation that goes quiet for a minute raised ReadTimeoutError and destroyed the entire approved
# run. Legacy retry mode also does not retry read timeouts.
BEDROCK_READ_TIMEOUT_SECONDS = int(os.getenv("BEDROCK_READ_TIMEOUT_SECONDS", "600"))
BEDROCK_CONNECT_TIMEOUT_SECONDS = int(os.getenv("BEDROCK_CONNECT_TIMEOUT_SECONDS", "15"))


@lru_cache(maxsize=8)
def _bedrock_client(service: str, region: str):
    """One shared boto3 client per (service, region). Building a client costs about two seconds
    here - botocore service data, credential and endpoint resolution - and ChatBedrockConverse
    builds two of them (bedrock-runtime for inference, bedrock for the control plane) unless both
    are handed in, so the pipeline was paying that on every model object it created."""
    return boto3.client(
        service,
        region_name=region,
        config=BotocoreConfig(
            read_timeout=BEDROCK_READ_TIMEOUT_SECONDS,
            connect_timeout=BEDROCK_CONNECT_TIMEOUT_SECONDS,
            retries={"max_attempts": 3, "mode": "standard"},
        ),
    )


# Estimated Bedrock on-demand prices in USD per 1M tokens as (input, output). These drive the
# dashboard's cost estimate only; they are not billing data, they go stale when AWS changes prices,
# and a private rate card will differ. Override with BEDROCK_PRICE_OVERRIDES, e.g.
# BEDROCK_PRICE_OVERRIDES='{"global.anthropic.claude-sonnet-4-6": [3.0, 15.0]}'
# Keyed by the part of the id that survives the us./global. inference-profile prefix. Every model
# the dashboard offers has to appear here: one unpriced call flips the whole run's estimate to
# "priced": False, and moving verification onto Haiku would otherwise have silently done exactly
# that to every run.
MODEL_PRICES_PER_MTOK: dict[str, tuple[float, float]] = {
    "anthropic.claude-sonnet-4-6": (3.00, 15.00),
    "anthropic.claude-sonnet-4-5": (3.00, 15.00),
    "anthropic.claude-haiku-4-5": (1.00, 5.00),
    "amazon.nova-pro": (0.80, 3.20),
    "amazon.nova-2-lite": (0.06, 0.24),
    "amazon.nova-lite": (0.06, 0.24),
}


# What a cached input token costs relative to a fresh one. Anthropic's published multipliers:
# reading from cache is a tenth of the input rate, writing to it is 1.25x. These drive the
# dashboard estimate only - like the price table itself, they are not billing data.
CACHE_READ_RATE = 0.1
CACHE_WRITE_RATE = 1.25


def price_per_mtok(model_id: str) -> tuple[float, float] | None:
    """Look up (input, output) price for a model id, ignoring the us./global. profile prefix."""
    try:
        overrides = json.loads(os.getenv("BEDROCK_PRICE_OVERRIDES", "") or "{}")
    except ValueError:
        overrides = {}
    if model_id in overrides:
        pair = overrides[model_id]
        return (float(pair[0]), float(pair[1]))
    for key, pair in MODEL_PRICES_PER_MTOK.items():
        if key in model_id:
            return pair
    return None


# Which pipeline step is executing right now. The dashboard used to attribute token usage to
# run.current_step, but that only updates once a node has finished, so every code-agent call was
# billed to the previous node - the tools node appeared to spend 267k tokens without making a
# single model call. Nodes set this on entry and the usage callback reports it.
CURRENT_STEP: ContextVar[str] = ContextVar("current_step", default="")


class _UsageTracker(BaseCallbackHandler):
    """Captures token usage for every model call from one place.

    Attached to the model itself, so it covers all the call shapes this pipeline uses - invoke,
    stream, with_structured_output and bind_tools - without
    each call site having to report anything. Usage is forwarded on the run's stream, where the
    dashboard totals it; outside a graph run this is a no-op.
    """

    def on_llm_end(self, response, **kwargs) -> None:
        usage, model_id = None, ""
        for generation in (response.generations or [[]])[0]:
            message = getattr(generation, "message", None)
            usage = usage or getattr(message, "usage_metadata", None)
            model_id = model_id or (getattr(message, "response_metadata", {}) or {}).get("model_name", "")
        if not usage:
            usage = (response.llm_output or {}).get("usage") if response.llm_output else None
        if not usage:
            return
        model_id = model_id or (response.llm_output or {}).get("model_id", "") or ""
        prompt_tokens = int(usage.get("input_tokens") or usage.get("inputTokens") or 0)
        output_tokens = int(usage.get("output_tokens") or usage.get("outputTokens") or 0)
        if not (prompt_tokens or output_tokens):
            return
        # Cached input is billed at a different rate, so counting it at the full input price makes
        # the estimate wrong in exactly the situation caching is there to create - and hides
        # whether caching is working at all. input_tokens already includes these.
        details = usage.get("input_token_details") or {}
        cache_read = int(details.get("cache_read") or 0)
        cache_write = int(details.get("cache_creation") or 0)
        prices = price_per_mtok(model_id)
        cost = None
        if prices:
            fresh = max(0, prompt_tokens - cache_read - cache_write)
            cost = (fresh * prices[0]
                    + cache_read * prices[0] * CACHE_READ_RATE
                    + cache_write * prices[0] * CACHE_WRITE_RATE
                    + output_tokens * prices[1]) / 1_000_000
        try:
            from langgraph.config import get_stream_writer

            get_stream_writer()({
                "kind": "usage", "model": model_id, "step": CURRENT_STEP.get(),
                "input_tokens": prompt_tokens, "output_tokens": output_tokens,
                "cache_read_tokens": cache_read, "cache_write_tokens": cache_write,
                "cost_usd": cost,
            })
        except Exception:
            return


_USAGE_TRACKER = _UsageTracker()


@lru_cache(maxsize=16)
def _build_model(model_id: str, region: str, max_tokens: int) -> ChatBedrockConverse:
    """Cached per (model, region, token budget). The planning, coding and director passes each use
    a different token budget, so without this the pipeline paid client setup again on every node -
    worst case a dozen times over during the code agent's tool loop alone."""
    return ChatBedrockConverse(
        model_id=model_id,
        region_name=region,
        temperature=0.45,
        max_tokens=max_tokens,
        client=_bedrock_client("bedrock-runtime", region),
        bedrock_client=_bedrock_client("bedrock", region),
        callbacks=[_USAGE_TRACKER],
    )


def _model(model_id: str | None = None, *, max_tokens: int = 4096) -> ChatBedrockConverse:
    """Return the LangChain Bedrock chat model used by every text specialist."""
    return _build_model(
        model_id or os.getenv("BEDROCK_MODEL_ID", DEFAULT_MODEL_ID),
        os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
        max_tokens,
    )


# Bedrock prompt caching. A cached prefix is billed at roughly a tenth of its normal input rate,
# and this pipeline is dominated by input: a measured run spent 85% of its tokens there, because
# every call in the code agent's loop re-sends the same system prompt and the same ~2,700 tokens of
# tool definitions. Measured here on that prefix: the second call read 6,520 of its 6,523 input
# tokens from cache.
#
# Only the Anthropic models are given a cache point. The Nova entries in the model list have their
# own rules for where a cachePoint may sit, and an unsupported combination is rejected by the API
# rather than ignored - a cost optimisation must not be able to fail a run.
PROMPT_CACHE_TTL = os.getenv("BEDROCK_PROMPT_CACHE_TTL", "5m").strip()


def cache_control_for(model_id: str | None) -> dict:
    """The cache_control kwarg for this model, or nothing when caching does not apply."""
    if not PROMPT_CACHE_TTL or PROMPT_CACHE_TTL.lower() in {"0", "off", "none"}:
        return {}
    return {"cache_control": {"ttl": PROMPT_CACHE_TTL}} if "anthropic" in (model_id or "") else {}


def _note(agent: str, text: str) -> None:
    """Best-effort progress line into the dashboard's agent log, using the same envelope the graph
    nodes use. Silently does nothing outside an active graph run."""
    try:
        from langgraph.config import get_stream_writer

        get_stream_writer()({"kind": "model_text", "agent": agent, "text": text})
    except Exception:
        return


# Composing a live preview costs O(answer-so-far): the accumulated content blocks are re-joined and
# the half-written tool argument re-copied. Doing that on every chunk makes a long generation
# quadratic in its own length, and the longest calls here (writing a 16k-token game, auditing it)
# emit thousands of chunks. The dashboard repaints a few times a second regardless, so previews are
# built on a timer rather than per chunk - the token stream itself is never slowed down by it.
PREVIEW_INTERVAL_SECONDS = float(os.getenv("PREVIEW_INTERVAL_SECONDS", "0.25"))
# What Bedrock reports when an answer stopped because it ran out of room rather than because the
# model had finished saying what it meant to say.
TRUNCATION_STOP_REASONS = {"max_tokens", "length", "model_length_exceeded"}


class PreviewTicker:
    """Allows one preview per interval, plus a final flush so the last state is never lost."""

    __slots__ = ("_interval", "_last")

    def __init__(self, interval: float | None = None) -> None:
        self._interval = PREVIEW_INTERVAL_SECONDS if interval is None else interval
        self._last = 0.0

    def ready(self) -> bool:
        now = time.monotonic()
        if now - self._last < self._interval:
            return False
        self._last = now
        return True


class StreamAccumulator:
    """Collect one streamed model turn without re-merging chunks on every token.

    `gathered = gathered + piece` is the shape every LangChain streaming example uses, and it is a
    trap for answers this long. Each `+` constructs a new AIMessageChunk, and constructing one
    re-parses the whole accumulated tool-call argument (AIMessageChunk.init_tool_calls runs
    parse_partial_json on it) and re-merges every content block, so the cost is quadratic in the
    answer's own length. Measured here on a write_game_file call carrying a 24KB game: 1.112s of
    pure CPU to accumulate, against 0.0007s the way this class does it. The code agent writes a
    whole game as a single tool argument, once per build and again on every repair, and that time
    was spent in our own loop after the model had already finished speaking.

    Fragments are appended to lists here and the finished turn is assembled exactly once, so
    accumulation is linear in the number of characters streamed. The final assembly still goes
    through AIMessageChunk, so a truncated argument is repaired by the same partial-JSON parser as
    before - that behaviour is relied on to report a cut-off answer honestly.
    """

    __slots__ = ("_blocks", "_metadata", "_seen", "_text", "_tool_args", "_tool_meta", "_usage")

    def __init__(self) -> None:
        self._text: list[str] = []
        self._blocks: list = []
        self._tool_args: dict[int, list[str]] = {}
        self._tool_meta: dict[int, dict] = {}
        self._metadata: dict = {}
        self._usage = None
        self._seen = False

    def add(self, piece) -> None:
        self._seen = True
        content = getattr(piece, "content", "")
        if isinstance(content, str):
            if content:
                self._text.append(content)
        else:
            for block in content or []:
                if not isinstance(block, dict):
                    self._blocks.append(block)
                elif block.get("type", "text") == "text":
                    if block.get("text"):
                        self._text.append(block["text"])
                elif block.get("type") != "tool_use":
                    # A tool_use content block duplicates what tool_call_chunks already carries,
                    # and Bedrock rebuilds it from tool_calls when the turn is sent back. Anything
                    # else is kept verbatim rather than silently dropped.
                    self._blocks.append(block)
        for chunk in getattr(piece, "tool_call_chunks", None) or []:
            index = chunk.get("index") or 0
            meta = self._tool_meta.setdefault(index, {"name": None, "id": None})
            if chunk.get("name"):
                meta["name"] = chunk["name"]
            if chunk.get("id"):
                meta["id"] = chunk["id"]
            if chunk.get("args"):
                self._tool_args.setdefault(index, []).append(chunk["args"])
        if metadata := getattr(piece, "response_metadata", None):
            self._metadata.update(metadata)
        # Bedrock reports usage once, as run totals, in its metadata event - so the last report is
        # the whole turn's usage rather than one increment of it.
        if usage := getattr(piece, "usage_metadata", None):
            self._usage = usage

    @property
    def empty(self) -> bool:
        return not self._seen

    @property
    def text(self) -> str:
        return "".join(self._text)

    @property
    def metadata(self) -> dict:
        return self._metadata

    def tool_arguments(self) -> list[tuple[str, str]]:
        """(tool name, raw argument JSON so far) per tool call, in the order they were opened."""
        return [(self._tool_meta[index].get("name") or "tool", "".join(fragments))
                for index, fragments in sorted(self._tool_args.items())]

    def preview(self) -> str:
        """Readable snapshot of a half-finished turn: prose so far plus the tool call being
        composed. This is the one place that shape is defined, for every streaming call site."""
        parts = [text] if (text := self.text.strip()) else []
        parts += [f"[{name}] {args}" for name, args in self.tool_arguments()]
        return "\n".join(parts)

    def truncated(self) -> bool:
        """Whether the model stopped because it ran out of output budget rather than because it
        had finished. The only honest signal that a complete-looking answer is not complete."""
        stop = self._metadata.get("stopReason") or self._metadata.get("finish_reason") or ""
        return str(stop).lower() in TRUNCATION_STOP_REASONS

    def finish(self) -> AIMessage:
        """Assemble the turn. One partial-JSON parse for the whole answer, not one per chunk."""
        merged = AIMessageChunk(content="", tool_call_chunks=[
            tool_call_chunk(name=meta.get("name"), args="".join(self._tool_args.get(index, [])),
                            id=meta.get("id"), index=index)
            for index, meta in sorted(self._tool_meta.items())
        ])
        text = self.text
        content: str | list = text
        if self._blocks:
            content = ([{"type": "text", "text": text}] if text else []) + self._blocks
        return AIMessage(
            content=content,
            tool_calls=list(merged.tool_calls),
            invalid_tool_calls=list(merged.invalid_tool_calls),
            response_metadata=dict(self._metadata),
            usage_metadata=self._usage,
        )


def stream_turn(
    model, messages, on_preview: Callable[[StreamAccumulator], None] | None = None
) -> StreamAccumulator:
    """Run one streamed model turn, offering a rate-limited live view of it while it is written.

    Every streaming call in this pipeline goes through here: the structured planners, the repair
    rewrite and the code agent's own loop each used to carry their own copy of this loop, and each
    copy independently paid for per-chunk accumulation and per-chunk preview composition.

    on_preview is handed the accumulator rather than a rendered string, because what is worth
    showing differs by call site - a structured answer is its tool argument, a code-agent turn is
    its prose and the tool call it is composing. It is called at most once per PREVIEW_INTERVAL,
    plus once at the end so the finished answer is never left half-drawn on screen.
    """
    accumulator, ticker = StreamAccumulator(), PreviewTicker()
    for piece in model.stream(messages):
        accumulator.add(piece)
        if on_preview is not None and ticker.ready():
            on_preview(accumulator)
    if on_preview is not None and not accumulator.empty:
        on_preview(accumulator)
    return accumulator


STRUCTURED_MAX_ATTEMPTS = int(os.getenv("STRUCTURED_MAX_ATTEMPTS", "3"))
# Output budget for a structured answer. Meant for the flat concept/art/plan schemas, but even those
# grow with the prompt: ImplementationPlan asks for up to 8 mechanics plus up to 8 acceptance tests,
# each with real numbers, and a long answer got cut off before its last field (acceptance_tests),
# which then failed validation as a bare "Field required" rather than an explained truncation. A
# caller whose answer grows with its input more sharply than that - the design review returns one
# check per requirement - still has to ask for more via its own max_tokens (see
# DESIGN_REVIEW_MAX_TOKENS in graph.py), or it is truncated mid-array.
STRUCTURED_MAX_TOKENS = int(os.getenv("STRUCTURED_MAX_TOKENS", "8000"))


# How far one call's output budget may grow across its retries. A truncated answer is not a schema
# violation the model can fix by being told about it - it ran out of room - so asking again on the
# same budget fails in exactly the same place. Each retry after a truncation gets more room, capped
# so a runaway answer still ends.
STRUCTURED_BUDGET_GROWTH = 2.0
STRUCTURED_MAX_RETRY_TOKENS = int(os.getenv("STRUCTURED_MAX_RETRY_TOKENS", "32000"))


def _structured_once(
    schema: type[T],
    system: str,
    user: str,
    model_id: str | None,
    on_chunk: Callable[[str], None] | None,
    max_tokens: int,
    outcome: dict[str, bool] | None = None,
) -> T:
    """One structured call. Records whether the answer was cut short in `outcome`, because that
    changes what a retry should do - see _structured."""
    messages = [("system", system), ("human", user)]
    if on_chunk is None:
        return _model(model_id, max_tokens=max_tokens).with_structured_output(schema).invoke(messages)
    base = _model(model_id, max_tokens=max_tokens)
    # Caching pays here across the retries: a truncated or rejected answer is re-asked with the
    # same system prompt and the same (often very large) source attached.
    model = base.bind_tools([schema], tool_choice=schema.__name__,
                            **cache_control_for(getattr(base, "model_id", "")))

    def show(accumulator: StreamAccumulator) -> None:
        # A structured answer is its tool argument; the prose around it is of no interest here.
        if arguments := accumulator.tool_arguments():
            on_chunk(arguments[0][1])

    answer = stream_turn(model, messages, on_preview=show)
    if answer.empty:
        raise RuntimeError(f"{schema.__name__} 모델이 빈 스트림을 반환했습니다.")
    # Running out of budget mid-answer is silent otherwise: LangChain's partial-JSON parser closes
    # off the half-written tool argument and hands back what looks like a perfectly good call, just
    # missing every field it never reached. That surfaces as a bare "<field>: Field required" on a
    # required key, which reads like the model ignored the schema rather than ran out of room.
    if answer.truncated():
        if outcome is not None:
            outcome["truncated"] = True
        _note(schema.__name__,
              f"출력 예산({max_tokens} 토큰)을 모두 써서 응답이 중간에 끊겼습니다. "
              "누락된 필드는 모델이 거기까지 쓰지 못한 것입니다.")
    if tool_calls := answer.finish().tool_calls:
        return schema.model_validate(tool_calls[0]["args"])
    raise RuntimeError(f"{schema.__name__} 스트리밍 응답에서 구조화된 결과를 얻지 못했습니다.")


def _trace_structured_inputs(inputs: dict) -> dict:
    """The schema and the model, not the prompt. A design-review prompt carries the whole game."""
    schema = inputs.get("schema")
    return {"schema": getattr(schema, "__name__", str(schema)),
            "model": inputs.get("model_id") or "",
            "max_tokens": inputs.get("max_tokens"),
            "prompt_chars": len(str(inputs.get("user", "")))}


@traceable(name="structured-output", run_type="chain", process_inputs=_trace_structured_inputs)
def _structured(
    schema: type[T],
    system: str,
    user: str,
    model_id: str | None = None,
    on_chunk: Callable[[str], None] | None = None,
    max_tokens: int = STRUCTURED_MAX_TOKENS,
) -> T:
    """Ask the model for one structured field, retrying with the validator's complaint attached.

    Without on_chunk this blocks until the whole answer is ready. With on_chunk, it streams the
    model's tool-call JSON as it is generated and calls on_chunk with the growing raw text after
    every chunk, so the dashboard can show live progress.

    A schema violation used to end the node outright - one omitted field (a reviewer with no extra
    findings leaving the key out) took down the whole run. Telling the model exactly what the
    validator rejected and asking again fixes almost all of these, so the retry happens here and
    covers every structured call rather than each caller handling it.

    Two different failures arrive as the same ValidationError, and they need opposite responses. A
    genuine schema violation is fixed by telling the model what the validator rejected. A truncated
    answer is not: the model was writing the right thing and ran out of room, so the missing field
    comes back missing however clearly it is asked for. Re-asking on the same budget is what made
    this look like the model refusing to obey the schema; the retry after a truncation gets more
    room and a different instruction - be shorter - instead.
    """
    prompt, budget = user, max_tokens
    for attempt in range(1, STRUCTURED_MAX_ATTEMPTS + 1):
        outcome: dict[str, bool] = {}
        try:
            return _structured_once(schema, system, prompt, model_id, on_chunk, budget, outcome)
        except ValidationError as error:
            if attempt == STRUCTURED_MAX_ATTEMPTS:
                raise
            complaints = "; ".join(
                f"{'.'.join(str(p) for p in issue['loc'])}: {issue['msg']}"
                for issue in error.errors()
            )
            _note(schema.__name__, f"구조화 응답 검증 실패({attempt}/{STRUCTURED_MAX_ATTEMPTS}): {complaints[:200]} — 다시 요청합니다.")
            if outcome.get("truncated"):
                grown = min(int(budget * STRUCTURED_BUDGET_GROWTH), STRUCTURED_MAX_RETRY_TOKENS)
                if grown > budget:
                    _note(schema.__name__, f"출력 예산을 {budget} → {grown} 토큰으로 늘려 다시 시도합니다.")
                    budget = grown
                prompt = (
                    f"{user}\n\n[이전 시도가 출력 예산을 초과해 {complaints} 에서 잘렸습니다]\n"
                    f"같은 내용을 훨씬 짧게 답하세요. 설명과 근거 문장을 최소한으로 줄이고, "
                    f"{schema.__name__}의 모든 필수 필드를 끝까지 채우는 것을 최우선으로 하세요."
                )
                continue
            prompt = (
                f"{user}\n\n[이전 시도가 스키마 검증에 실패했습니다]\n{complaints}\n"
                f"{schema.__name__}의 모든 필수 필드를 채워 다시 반환하세요. "
                "내용이 없는 목록은 생략하지 말고 빈 배열로 명시하세요."
            )
    raise RuntimeError(f"{schema.__name__} 구조화 응답을 받지 못했습니다.")


def _content_text(message: object) -> str:
    """Normalize Bedrock Converse content blocks to plain text for HTML generation."""
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
        )
    return str(content)


def create_concept(
    brief: str,
    use_llm: bool,
    model_id: str | None = None,
    on_chunk: Callable[[str], None] | None = None,
    production_brief: str = "",
    output_root: str | Path | None = None,
) -> GameConcept:
    if not use_llm:
        raise RuntimeError("기획 모델 연결이 필요합니다. 오프라인 템플릿은 제작에 사용하지 않습니다.")
    user = f"Player brief:\n{brief}"
    # A player who said what they want outranks everything this module adds on its own. Asked for
    # a faithful Tetris, the pipeline answered with a different game: the standing prompt wants one
    # twist of the borrowed loop and an original work, and the recent-productions nudge was telling
    # it to avoid the puzzle game it shipped last week. All three are right in the absence of a
    # request and wrong against one.
    requested = player_requested(brief)
    if requested:
        user += (
            "\n\n플레이어가 원하는 것을 직접 지정했습니다. 위 요청이 다른 모든 지침보다 우선합니다. "
            "요청이 특정 게임을 그대로 만들어 달라는 것이면 그 게임의 규칙·조작·승패 조건을 "
            "알아볼 수 있게 그대로 재현하세요. 이 경우 독창적인 변형을 더하지 말고, 요청과 다른 "
            "방향으로 바꾸지도 마세요. 제목·캐릭터·아트만 직접 가져오지 않으면 됩니다."
        )
    # An auto request is given a genre here rather than left open - see resolve_auto_genre for why
    # "the agent picks freely" produced the same game every time.
    auto = resolve_auto_genre(brief)
    if auto:
        user += (
            f"\n\n이번 실행에 배정된 장르: {auto}\n"
            "장르가 지정되지 않은 요청이라 이 실행의 시드로 장르를 배정했습니다. "
            "이 장르로 기획하세요. 다른 장르로 바꾸지 마세요."
        )
    # Seeded per run, so two auto runs in the same genre no longer open from the same three games.
    references = genre_references(f"{brief}\n{auto}", seed=brief)
    if references and not requested:
        user += (
            f"\n\n이 장르의 대표작과 빌릴 메커닉:\n{references}\n"
            "이 중에서 골라 reference_games에 적고, 그 루프를 알아볼 수 있게 유지하세요. "
            "자기 변형은 하나까지만 더하고, 장르를 섞거나 낯선 조작을 만들지 마세요."
        )
    elif named_in_brief(requested):
        # The player named the game. There is nothing to choose between, so this is not a list -
        # and saying it as a list is how "슈퍼마리오와 똑같은 게임" came back citing Super Mario
        # Bros, Donkey Kong Jr and Bubble Bobble, each contributing mechanics to the contract.
        user += (
            f"\n\n플레이어가 이 게임을 지목했습니다:\n{references}\n"
            "reference_games에는 **이 게임 하나만** 적으세요. 다른 게임을 덧붙이지 마세요 — "
            "타이머·목숨·스테이지 진행 같은 구조를 다른 게임에서 빌려 오면 요청에 없던 분량이 "
            "계약에 들어가고, 그만큼 완성되지 않은 게임이 나옵니다. 이 게임의 루프만 재현하세요."
        )
    elif references:
        # A menu is the right framing when the studio is choosing and the wrong one when the player
        # already has: told to pick from a list, a model asked for a faithful Tetris will pick the
        # nearest listed game instead. Here the list is background, and the request outranks it.
        user += (
            f"\n\n같은 계열의 대표작과 빌릴 메커닉입니다:\n{references}\n"
            "플레이어 요청에 맞는 것만 참고하고 어긋나는 것은 무시하세요. 이 목록 때문에 요청과 "
            "다른 게임이 되면 안 됩니다. 장르를 섞거나 낯선 조작을 만들지 마세요. "
            "reference_games는 최대 2개입니다 — 빌린 루프 하나에 변형 하나면 충분합니다."
        )
    # Borrowing a famous game is encouraged and reproducing one is not the same thing. A Mario-like
    # request came back as Mario's feature list - running acceleration, ? blocks, coin 1-ups, timer
    # bonuses, flagpole scoring bands, damage states - and the build spent its whole budget on the
    # list and shipped a game that never started. The loop is what transfers; the feature list is
    # what the original had years to add.
    if references:
        user += (
            "\n\n대표작에서 가져올 것은 **루프 하나**입니다. 그 게임의 기능 목록을 재현하려 하지 "
            "마세요 — 원작이 몇 년에 걸쳐 쌓은 것이고, 여기서 만드는 것은 한 판 60~120초짜리 "
            "게임입니다. core_loop는 플레이어가 그 시간 동안 반복하는 행동만 적으세요."
        )
    # The one thing this agent cannot work out for itself: what it already built. A genre assigned
    # from the run seed spreads runs apart, but inside one genre the model still reaches for the
    # same design, and nothing in the prompt has ever told it otherwise.
    # Only when the player did not say what they wanted. Variety is what to optimise for in the
    # absence of a request, never against one: a player who asks for a faithful Tetris and is told
    # "make something distinctly different from the puzzle game you shipped last week" gets neither.
    if not requested and (recent := recent_productions(output_root)):
        user += (
            "\n\n이 스튜디오가 최근에 만든 게임들입니다:\n"
            + "\n".join(f"- {entry}" for entry in recent)
            + "\n이것들과 뚜렷하게 다른 게임을 기획하세요. 핵심 루프, 플레이어가 조작하는 대상, "
              "승리 조건 중 최소 두 가지가 위 어느 것과도 겹치지 않아야 합니다."
        )
    if production_brief:
        user += f"\n\nProduction director's brief - align your concept with it:\n{production_brief}"
    return _structured(GameConcept, IDEA_SYSTEM, user, model_id, on_chunk=on_chunk)


# How many exemplars reach the idea prompt. Three is what a row used to hold outright; rows now
# hold five to seven and the run seed picks which three, so one genre stops meaning one fixed
# answer. More than three crowds the prompt without adding choice - the model borrows one loop.
GENRE_REFERENCE_SAMPLE = int(os.getenv("GENRE_REFERENCE_SAMPLE", "3"))

# Longest label first: "퍼즐" is a substring of "물리 퍼즐", and a brief that asked for 물리 퍼즐 must
# not match the plain puzzle row on the way past.
_GENRE_LABELS = sorted(GENRE_REFERENCES, key=len, reverse=True)

# Two loop words, not one. "점프" alone turns up in a shooter brief and "카드" in a roguelike one;
# two of a genre's words together are a description rather than a coincidence. A named game needs
# no threshold - naming Tetris is not an accident.
GENRE_KEYWORD_THRESHOLD = 2


def _entry_names(entry: str) -> list[str]:
    """Every name a reference game goes by: "Tetris(테트리스): ..." is both Tetris and 테트리스."""
    latin, _, korean = entry.split(":", 1)[0].partition("(")
    return [name for name in (latin.strip(), korean.rstrip(")").strip()) if name]


def _despace(text: str) -> str:
    """Lowercased, with spaces and punctuation removed.

    Game titles are written however the player feels like writing them, and every difference used
    to be a miss: the table says "슈퍼 마리오" and a real brief said "슈퍼마리오와 똑같은 게임"; the
    table says "Plants vs. Zombies" and nobody types the full stop; "Pac-Man" is written "pac man".
    Each miss dropped the named game out of the references entirely, and the run was handed three
    arbitrary same-genre games instead - Super Mario Bros, Donkey Kong Jr and Bubble Bobble, all
    three contributing mechanics to a contract that only ever asked for Mario.
    """
    return "".join(char for char in text.lower() if char.isalnum())


# Below this many characters a name is too generic to look for inside a sentence: "N++" normalises
# to "n" and would match every brief with the letter n in it, and "uno" appears inside "unorthodox".
# Short names are matched as whole words instead, which is how anyone writes them anyway.
_SHORT_NAME_CHARS = 5


def _name_present(text: str, name: str) -> bool:
    if len(normalised := _despace(name)) >= _SHORT_NAME_CHARS:
        return normalised in _despace(text)
    return re.search(rf"(?<![0-9a-z]){re.escape(name.lower())}(?![0-9a-z])", text.lower()) is not None


# Words that never end a name anyone would shorten to. Without them "world of" matches World of Goo
# on a brief about World of Warcraft, and "binding of" matches on anything.
_TRAILING_STOPWORDS = {"of", "the", "vs", "vs.", "and", "a", "an", "to"}


def _shortened(spoken: str, full: str) -> bool:
    """Whether `spoken` is how people shorten `full` - "Super Mario" for "Super Mario Bros".

    Two words at least, and not ending on a joining word, so a genuine short name matches and a
    fragment does not. One word would match "Super" against every game starting with it.
    """
    words = full.lower().split()
    for count in range(2, len(words)):
        if words[count - 1] in _TRAILING_STOPWORDS:
            continue
        if _despace(" ".join(words[:count])) == _despace(spoken):
            return True
    return False


def _names(text: str, entry: str) -> bool:
    words = text.lower().split()
    for name in _entry_names(entry):
        if _name_present(text, name):
            return True
        # The other direction: the brief says less of the name than the table does.
        if any(_shortened(" ".join(words[start:start + size]), name)
               for size in (2, 3) for start in range(max(0, len(words) - size + 1))):
            return True
    return False


def named_in_brief(text: str) -> list[str]:
    """Reference games the player actually named, across every genre.

    Separate from genre inference because the two answer different questions: which row to draw
    from, and whether the player already decided. Naming a game is the strongest signal this
    module gets, and it is the one that has to survive into reference_games unpadded.
    """
    if not (text := (text or "").strip()):
        return []
    return [entry for entries in GENRE_REFERENCES.values() for entry in entries
            if _names(text, entry)]


def _leader(scores: dict[str, int], minimum: int) -> str:
    """The highest scorer, ties going to table order, or "" if nothing clears the bar."""
    label = max(scores, key=lambda key: scores[key]) if scores else ""
    return label if label and scores[label] >= minimum else ""


def infer_genre(text: str) -> str:
    """Which genre a player's own words are describing, or "" when they are not describing one.

    Needed because "자동 기획" and "커스텀" are the two dropdown values that carry no genre, and a
    player who picks either and then types a request has chosen a genre anyway - in words. Read it
    here or the run seed assigns one over the top of it.

    Keyword matching rather than a model call or an embedding index. The corpus is a literal of
    fifteen rows in this repository, the query is one sentence, and the answer only selects which
    three exemplars to show; a retrieval stack would add an index to keep in sync and a second
    source of run-to-run variance to a function whose whole job is to stop the wrong exemplars
    being attached. When it cannot tell, it says so, and the caller attaches nothing.
    """
    if not (text := text.strip()):
        return ""
    named = {label: sum(_names(text, entry) for entry in entries)
             for label, entries in GENRE_REFERENCES.items()}
    if label := _leader(named, 1):
        return label
    return _leader({label: sum(word in text for word in words)
                    for label, words in GENRE_KEYWORDS.items()}, GENRE_KEYWORD_THRESHOLD)


def genre_label(brief: str) -> str:
    """Which reference row this brief belongs to, or "" if none does.

    The dashboard writes the choice into the brief as "Requested genre: <label>", so a chosen
    genre is just a substring. Only when nothing was chosen does the player's own text get read.
    """
    for label in _GENRE_LABELS:
        if label in brief:
            return label
    return infer_genre(player_requested(brief))


def sample_references(label: str, seed: str, brief: str = "") -> list[str]:
    """This run's exemplars for one genre: the games the brief named, then a seeded shuffle.

    Seeded, so the same run reproduces and different runs differ - the same thing the run seed
    does for the genre itself, one level down. Without it a genre meant three fixed games and
    every auto run inside that genre started from the same three.

    A game the player named is not one of three - it is the answer. Asked for "슈퍼마리오와 똑같은
    게임", a run came back citing Super Mario Bros *and* Donkey Kong Jr *and* Bubble Bobble, and
    each one arrived carrying mechanics: the contract then owed a timer-and-lives system and a
    stage-and-collectible scoring structure that nobody asked for, on top of Mario. Padding a named
    request up to the sample size is us adding scope the player did not request, and it is the
    upstream cause of the contracts that shipped unplayable.

    So naming a game replaces the sample rather than anchoring it. The shuffle only runs when there
    is nothing named to run it for.
    """
    entries = list(GENRE_REFERENCES.get(label, ()))
    if named := [entry for entry in entries if brief and _names(brief, entry)]:
        return named
    match = _RUN_SEED.search(seed)
    digest = hashlib.sha256(f"{label}\n{match.group(1) if match else seed}".encode())
    rest = list(entries)
    random.Random(int(digest.hexdigest(), 16)).shuffle(rest)
    return rest[:GENRE_REFERENCE_SAMPLE]


def genre_references(brief: str, seed: str = "") -> str:
    """Exemplars for the genre this brief is working in, if there is one we have a row for."""
    label = genre_label(brief)
    if not label:
        return ""
    return "\n".join(f"- {entry}" for entry in sample_references(label, seed or brief, brief))


# What the dashboard and the CLI write when the player did not choose a genre.
AUTO_GENRE_MARKERS = ("자동 기획", "requested genre: auto")
_RUN_SEED = re.compile(r"Run seed:\s*(\S+)")


# How many finished games the idea agent is shown so it does not repeat one. Read straight off the
# output folder's manifests - there is no store to keep in sync, and a game that gets deleted stops
# counting by itself.
RECENT_TITLE_LIMIT = int(os.getenv("RECENT_TITLE_LIMIT", "6"))


def recent_productions(output_root: str | Path | None) -> list[str]:
    """Titles and genres of the most recently finished games, newest first.

    Assigning a genre from the run seed spreads runs across the reference table, but within one
    genre the model still reaches for the same design - it has no way to know what it built last
    time. The pipeline already records exactly that in every production manifest, so the cheapest
    long-term memory available is the output folder itself: no store to keep in sync, and nothing
    to migrate.
    """
    root = Path(output_root) if output_root else None
    if not root or not root.is_dir():
        return []
    try:
        manifests = sorted(root.glob("*/production-manifest.json"),
                           key=lambda path: path.stat().st_mtime, reverse=True)
    except OSError:
        return []
    seen: list[str] = []
    for manifest in manifests[: RECENT_TITLE_LIMIT * 2]:
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        title = str((data.get("concept") or {}).get("title", "")).strip()
        genre = str((data.get("implementation_plan") or {}).get("genre", "")).strip()
        entry = f"{title} ({genre})" if genre else title
        if title and entry not in seen:
            seen.append(entry)
        if len(seen) >= RECENT_TITLE_LIMIT:
            break
    return seen


# What the dashboard writes into the brief when the player left the box empty, and what the CLI
# writes for the same case. Either means "you decide".
# Matched as substrings of the full sentences those two write, so a caller that phrases it slightly
# differently still reads as "no request" rather than as a player asking for a game called
# "독자적으로 기획하세요". That distinction now decides whether the run seed may assign a genre.
_NO_REQUEST_MARKERS = (
    "독자적으로 기획",
    "사용자 경험 미입력",
)


def player_requested(brief: str) -> str:
    """The player's own words, or "" when they left it to the studio.

    The brief always carries a Player brief line; the question is whether a human wrote it. That
    decides which way two of this module's own nudges should point, so it is worth answering
    precisely rather than by guessing from length.
    """
    for line in brief.splitlines():
        if line.startswith("Player brief:"):
            text = line.removeprefix("Player brief:").strip()
            return "" if any(marker in text for marker in _NO_REQUEST_MARKERS) else text
    return ""


def resolve_auto_genre(brief: str) -> str:
    """Pick this run's genre when the player did not, deterministically from its run seed.

    "자동 기획" used to mean the idea agent got no genre at all - and, because the reference table
    is keyed by genre, no exemplars either. It was the one mode with nothing to anchor on, which
    sounds like freedom and is the opposite: left with only the standing constraints (one Canvas
    file, a 60-120 second session, a score to compete against, passive play must lose, borrow a
    loop that is known to work), the model converges on the single design that satisfies all of
    them. Three consecutive auto runs came back as the same falling-object shooter, two of them
    with the same title.

    The run seed was supposed to prevent that and could not: it is a hex string in a prompt, not a
    sampling seed, and a model has no way to turn it into a different design. Here it selects the
    genre instead, so it does what it was always meant to do - a different seed is a different
    game, and the same seed reproduces one.

    Only when the player left it open. "자동 기획" is the default dropdown value, so it is also what
    a player leaves selected while typing the game they want into the brief box, and the seed used
    to overrule them: a brief reading "블록을 회전시켜 빈틈없이 쌓는 게임을 만들어줘" was assigned
    플랫포머 and handed Super Mario Bros as the loop to borrow. A description is a genre choice, so
    it is read (see infer_genre) rather than overwritten, and the seed decides only in silence.
    """
    if not any(marker in brief.lower() or marker in brief for marker in AUTO_GENRE_MARKERS):
        return ""
    if player_requested(brief):
        return ""
    labels = list(GENRE_REFERENCES)
    match = _RUN_SEED.search(brief)
    seed = match.group(1) if match else brief
    return labels[int(hashlib.sha256(seed.encode("utf-8")).hexdigest(), 16) % len(labels)]


def create_art(
    concept: GameConcept,
    use_llm: bool,
    model_id: str | None = None,
    findings: list[str] | None = None,
    existing_sprites: list[str] | None = None,
) -> ArtDirection:
    """Plan the visual system. With findings, this is a revision after QA rejected the build, so the
    plan has to change rather than come back the same."""
    if not use_llm:
        raise RuntimeError("아트 기획 모델 연결이 필요합니다.")
    user = f"Game concept:\n{concept.model_dump_json(indent=2)}"
    if findings:
        user += (
            "\n\n[아트 방향 재수립] 이 게임은 검증을 통과하지 못했고, 지적 사항은 다음과 같습니다:\n"
            + json.dumps(findings, ensure_ascii=False)
            + f"\n이미 생성된 스프라이트: {json.dumps(existing_sprites or [], ensure_ascii=False)}\n"
            "asset_plan을 이 지적에 맞게 고치세요. 게임에 실제로 필요한데 빠진 객체를 추가하고, "
            "쓰이지 않을 객체는 빼고, 이미 생성된 스프라이트는 그 이름을 그대로 유지하세요. "
            "같은 계획을 반복하지 마세요."
        )
    return _structured(ArtDirection, ART_SYSTEM, user, model_id)


def create_optional_image(art: ArtDirection, asset_dir: Path, enabled: bool) -> str | None:
    """Generate one optional decorative asset; the game never depends on it to function."""
    asset_dir.mkdir(parents=True, exist_ok=True)
    (asset_dir / "image-prompt.txt").write_text(art.image_prompt, encoding="utf-8")
    image_model_id = os.getenv("BEDROCK_IMAGE_MODEL_ID", "").strip()
    if not enabled or not image_model_id:
        return None
    try:
        client = boto3.client("bedrock-runtime", region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1"))
        response = client.invoke_model(
            modelId=image_model_id,
            body=json.dumps(
                {
                    "taskType": "TEXT_IMAGE",
                    "textToImageParams": {"text": art.image_prompt},
                    "imageGenerationConfig": {
                        "numberOfImages": 1,
                        "quality": "standard",
                        "height": 1024,
                        "width": 1024,
                        "cfgScale": 8.0,
                        "seed": 0,
                    },
                }
            ),
        )
        encoded = json.loads(response["body"].read()).get("images", [None])[0]
        if not encoded:
            return None
        target = asset_dir / "generated-backdrop.png"
        target.write_bytes(base64.b64decode(encoded))
        return str(target)
    except Exception as error:  # Image failure must not prevent a playable build.
        (asset_dir / "image-error.txt").write_text(str(error), encoding="utf-8")
        return None


def normalize_html(value: str) -> str:
    value = value.strip()
    if value.startswith("```"):
        value = value.split("\n", 1)[1] if "\n" in value else ""
        value = value.rsplit("```", 1)[0]
    start = value.lower().find("<!doctype html")
    return value[start:].strip() if start >= 0 else value


@lru_cache(maxsize=1)
def _node_path() -> str | None:
    """Cached, because shutil.which walks PATH and static_qa runs on every tool call."""
    return shutil.which("node")


# One node process checks every script block. Spawning node costs about 300ms on this machine
# (`node -e 0` alone measures 298ms), which is the entire cost of the check - so the number of
# spawns, not the amount of code, is what makes QA slow.
_JS_CHECKER = """
let raw = '';
process.stdin.on('data', d => raw += d);
process.stdin.on('end', () => {
  const vm = require('vm');
  const blocks = JSON.parse(raw);
  const errors = [];
  blocks.forEach((code, i) => {
    try {
      new vm.Script(code);
    } catch (err) {
      const message = String(err && err.message || err);
      // A <script type="module"> block legitimately uses import/export, which a classic-script
      // compile rejects. Reporting that as a syntax error would be a false failure.
      if (/import statement outside a module|Unexpected token 'export'|import\\.meta/.test(message)) return;
      errors.push({ index: i + 1, message: message.slice(0, 600) });
    }
  });
  process.stdout.write(JSON.stringify(errors));
});
"""


def _js_syntax_errors(scripts: list[str]) -> list[str]:
    """Compile every inline script in a single node process and report real syntax errors."""
    node = _node_path()
    if not node:
        return []
    try:
        result = subprocess.run(
            [node, "--input-type=commonjs", "-e", _JS_CHECKER],
            input=json.dumps(scripts), text=True, capture_output=True, timeout=20,
        )
        if result.returncode:
            return [f"JavaScript syntax check failed: {result.stderr[:300]}"]
        return [
            f"JavaScript syntax error in script {item['index']}: {item['message']}"
            for item in json.loads(result.stdout or "[]")
        ]
    except (OSError, subprocess.TimeoutExpired, ValueError):
        # The check is a convenience, not a gate on shipping: a broken toolchain must not fail
        # every build. Real syntax problems still surface when the game is opened.
        return []


# Same HTML, same answer. The code agent calls run_static_qa repeatedly inside its loop - often on
# a draft it has not changed since the last check - and each of those was paying a node spawn.
_QA_CACHE: dict[tuple[str, tuple[str, ...]], QAReport] = {}
_QA_CACHE_LIMIT = 32


def static_qa(html: str, sprites: list[str] | None = None) -> QAReport:
    key = (hashlib.sha256(html.encode("utf-8")).hexdigest(), tuple(sprites or ()))
    cached = _QA_CACHE.get(key)
    if cached is not None:
        return cached.model_copy(deep=True)
    # Only things that stop the game from working block a build. Keyword checks for niceties kept
    # failing games that were perfectly playable - a working reset() button read as "Missing
    # restart" - and each failure cost a repair and a rethink cycle. Those are advisory now: they
    # are reported so the agent can fix them, but they do not hold the release.
    blocking = {
        "canvas": ("<canvas",),
        "animation loop": ("requestanimationframe",),
        "keyboard controls": ("keydown",),
    }
    advisory = {
        "restart": ("restart", "reset", "replay", "playagain", "play again", "다시 시작", "재시작"),
        "score": ("score", "points", "점수", "highscore", "combo", "clear", "완주"),
    }
    lowered = html.lower()
    findings = [f"Missing {label}." for label, needles in blocking.items()
                if not any(needle in lowered for needle in needles)]
    notes = [f"Advisory: no {label} detected." for label, needles in advisory.items()
             if not any(needle in lowered for needle in needles)]
    forbidden = [needle for needle in ("<script src=", "fetch(", "http://", "https://") if needle in lowered]
    findings += [f"Forbidden external dependency: {item}" for item in forbidden]
    if "</html>" not in lowered:
        findings.append("HTML is incomplete: missing closing html element.")
    # Generated art the game never draws is paid-for work thrown away - one run produced six
    # sprites and referenced none of them - but it is not a reason to fail a release. The game
    # runs. Blocking on it spent a whole rethink budget on "you did not use art you paid for" and
    # then shipped with the finding open anyway, so it is advisory and the write result, where the
    # agent still has calls left, is where it is worth acting on.
    advisories: list[str] = []
    if unused := unused_sprites(html, sprites or []):
        advisories.append(
            f"Advisory: generated sprites are never drawn: {', '.join(unused)}. Load each with "
            "new Image() and draw it with ctx.drawImage, keeping a Canvas fallback."
        )
    # WASD has to work alongside the arrow keys, and it has to be read from event.code. A game that
    # keys off event.key looks correct and then silently ignores W/A/S/D whenever the keyboard is in
    # 한글 mode (W arrives as 'ㅈ') or Caps Lock is on. Arrow keys keep working, which is exactly how
    # this shipped unnoticed.
    if "keydown" in lowered:
        wasd = [code for code in ("keyw", "keya", "keys", "keyd") if code not in lowered]
        if wasd:
            findings.append(
                "Missing WASD support: handle event.code values KeyW/KeyA/KeyS/KeyD alongside the "
                "arrow keys (event.key is layout-dependent and breaks under a Korean IME)."
            )
    class Scripts(HTMLParser):
        def __init__(self):
            super().__init__(); self.active = False; self.scripts = []; self.buffer = ''
        def handle_starttag(self, tag, attrs):
            attrs = dict(attrs)
            if tag == 'script' and attrs.get('type', '') in {'', 'text/javascript', 'module'}:
                self.active = True; self.buffer = ''
        def handle_data(self, data):
            if self.active: self.buffer += data
        def handle_endtag(self, tag):
            if tag == 'script' and self.active:
                self.scripts.append(self.buffer); self.active = False
    parser = Scripts(); parser.feed(html)
    # Missing dev tooling is an environment gap, not a defect in the generated game: it must never
    # block shipping (a Node-less host would otherwise force every game into a repair loop forever).
    if parser.scripts and _node_path():
        findings += _js_syntax_errors(parser.scripts)
    report = QAReport(
        status="repair" if findings else "pass",
        # Advisories ride along so the agent can still act on them, without blocking the release.
        # The keyword guesses ("no score detected") only surface next to a real failure, because on
        # their own they are as often wrong as right. Art that was generated and never drawn is not
        # a guess - it is a file on disk nothing references - so it is always reported.
        findings=findings + advisories + (notes if findings else []),
        repair_instructions="Restore the missing core game mechanics and remove all network dependencies.",
    )
    if len(_QA_CACHE) >= _QA_CACHE_LIMIT:
        _QA_CACHE.clear()
    _QA_CACHE[key] = report
    return report.model_copy(deep=True)



# The director is one model call that decides scope, and DIRECTOR_TIMEOUT_SECONDS=0 skips it.
#
# It used to be a deepagents supervisor with four subagents whose system prompts were IDEA_SYSTEM,
# ART_SYSTEM, CODE_SYSTEM and QA_SYSTEM - the same prompts the real stages run on. Delegating to
# its "idea" subagent therefore ran the idea agent, and then idea_node ran it again: the planning
# happened twice, and the first copy was discarded except for one paragraph of text. See
# DIRECTOR_SYSTEM for what that paragraph is for now.
#
# The step and time budgets went with it. They existed because deepagents ships its graph with
# recursion_limit=1000, so one .invoke() could fan out into tens of minutes of sequential Bedrock
# round trips with nothing to show. A single call needs no budget to contain it.
DIRECTOR_TIMEOUT_SECONDS = int(os.getenv("DIRECTOR_TIMEOUT_SECONDS", "60"))
DIRECTOR_MAX_TOKENS = int(os.getenv("DIRECTOR_MAX_TOKENS", "1024"))
# The brief is guidance for the planners, not a document. Past this length the director is
# designing the game again, which is the thing this pass stopped doing.
DIRECTOR_BRIEF_CHARS = int(os.getenv("DIRECTOR_BRIEF_CHARS", "900"))

# What the run actually has to deliver. The old director only ever knew the Canvas path - its code
# subagent ran on CODE_SYSTEM and never on GODOT_CODE_SYSTEM - so a Godot run was coordinated as
# "a shippable standalone HTML game" and the planners were then told to align their concept with
# that. Two of the three finished Godot projects on disk were planned under that misdescription.
_ENGINE_NOTE = {
    "godot": ("이 런의 산출물은 Godot 4 프로젝트입니다 — project.godot, 씬(.tscn), GDScript. "
              "브라우저 단일 파일이 아니고, 엔진이 헤드리스로 컴파일해 실제로 실행하며 검증합니다."),
    "html5": ("이 런의 산출물은 외부 의존성이 전혀 없는 단일 HTML Canvas 페이지입니다. "
              "파일 하나가 그대로 게임입니다."),
}


def run_director(
    brief: str,
    use_llm: bool,
    model_id: str | None = None,
    on_step: Callable[[str, str, str], None] | None = None,
    engine: str = "html5",
) -> str:
    """Decide what this run will not build, in one call, and say so in a paragraph.

    Returns "" whenever there is no brief to give - switched off, offline, or the call failed -
    and never an explanation of why. This value is handed to the planners as "Production
    director's brief", and a run once opened by telling the idea agent that its brief was
    "Director fallback: TypeError: 'Overwrite' object is not iterable". Every stage after this one
    works without a brief, so an absent one is a non-event; a wrong one is not.
    """
    if not use_llm or DIRECTOR_TIMEOUT_SECONDS <= 0:
        return ""
    director_model = os.getenv("BEDROCK_DIRECTOR_MODEL_ID", "").strip() or model_id
    user = f"{_ENGINE_NOTE.get(engine, _ENGINE_NOTE['html5'])}\n\n플레이어 요청:\n{brief}"
    try:
        answer = _model(director_model, max_tokens=DIRECTOR_MAX_TOKENS).invoke(
            [("system", DIRECTOR_SYSTEM), ("human", user)]
        )
    except Exception as error:
        if on_step:
            on_step("director", "",
                    f"총괄 감독 실패 (제작 지침 없이 진행합니다): {type(error).__name__}: {error}"[:300])
        return ""
    return _content_text(answer).strip()[:DIRECTOR_BRIEF_CHARS]
