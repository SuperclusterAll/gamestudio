"""Model-backed specialist adapters and deterministic syntax checks."""

from __future__ import annotations

import base64
import hashlib
import json
import os
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
from deepagents import create_deep_agent
from langchain_aws import ChatBedrockConverse
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.messages.tool import tool_call_chunk
from pydantic import ValidationError

from .models import ArtDirection, GameConcept, QAReport
from .prompts import (
    ART_SYSTEM,
    CODE_SYSTEM,
    DIRECTOR_SYSTEM,
    GENRE_REFERENCES,
    IDEA_SYSTEM,
    QA_SYSTEM,
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
    stream, with_structured_output, bind_tools, and the director's deepagents subagents - without
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
        prices = price_per_mtok(model_id)
        cost = None
        if prices:
            cost = (prompt_tokens * prices[0] + output_tokens * prices[1]) / 1_000_000
        try:
            from langgraph.config import get_stream_writer

            get_stream_writer()({
                "kind": "usage", "model": model_id, "step": CURRENT_STEP.get(),
                "input_tokens": prompt_tokens, "output_tokens": output_tokens,
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
    model = _model(model_id, max_tokens=max_tokens).bind_tools([schema], tool_choice=schema.__name__)

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
) -> GameConcept:
    if not use_llm:
        raise RuntimeError("기획 모델 연결이 필요합니다. 오프라인 템플릿은 제작에 사용하지 않습니다.")
    user = f"Player brief:\n{brief}"
    references = genre_references(brief)
    if references:
        user += (
            f"\n\n이 장르의 대표작과 빌릴 메커닉:\n{references}\n"
            "이 중에서 골라 reference_games에 적고, 그 루프를 알아볼 수 있게 유지하세요. "
            "자기 변형은 하나까지만 더하고, 장르를 섞거나 낯선 조작을 만들지 마세요."
        )
    if production_brief:
        user += f"\n\nProduction director's brief - align your concept with it:\n{production_brief}"
    return _structured(GameConcept, IDEA_SYSTEM, user, model_id, on_chunk=on_chunk)


def genre_references(brief: str) -> str:
    """Exemplars for the genre this brief asked for, if it named one we have a row for.

    The dashboard writes the choice into the brief as "Requested genre: <label>", and 자동 기획 /
    커스텀 mean the agent picks freely - there is no single genre to anchor those on.
    """
    for label, examples in GENRE_REFERENCES.items():
        if label in brief:
            return examples
    return ""


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
    # Generated art that the game never draws is wasted GPU time and a broken promise to the
    # design: one run produced six sprites and referenced none of them.
    unused = [name for name in (sprites or []) if name.lower() not in lowered]
    if unused:
        findings.append(
            f"Generated sprites are never drawn: {', '.join(unused)}. Load each with new Image() "
            "and draw it with ctx.drawImage, keeping a Canvas fallback."
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
        findings=findings + (notes if findings else []),
        repair_instructions="Restore the missing core game mechanics and remove all network dependencies.",
    )
    if len(_QA_CACHE) >= _QA_CACHE_LIMIT:
        _QA_CACHE.clear()
    _QA_CACHE[key] = report
    return report.model_copy(deep=True)


# The director is a Deep Agents supervisor: one .invoke() on it fans out into many sequential
# Bedrock round trips (its own turns, plus a full agent loop inside every subagent it delegates
# to), and deepagents ships its graph with recursion_limit=1000, so an unbounded run can burn tens
# of minutes with nothing to show. These budgets are enforced by the streaming loop below rather
# than trusted to the inner graph, and setting DIRECTOR_TIMEOUT_SECONDS=0 skips the pass entirely.
DIRECTOR_MAX_STEPS = int(os.getenv("DIRECTOR_MAX_STEPS", "12"))
DIRECTOR_TIMEOUT_SECONDS = int(os.getenv("DIRECTOR_TIMEOUT_SECONDS", "60"))
DIRECTOR_MAX_TOKENS = int(os.getenv("DIRECTOR_MAX_TOKENS", "1024"))


def _director_step_summary(payload: object) -> tuple[str, str]:
    """Pull (tool names, text) out of one deepagents node update for progress logging."""
    messages = (payload or {}).get("messages") or [] if isinstance(payload, dict) else []
    tools, texts = [], []
    for message in messages:
        for call in getattr(message, "tool_calls", None) or []:
            tools.append(call.get("name", "tool"))
        text = _content_text(message).strip()
        if text:
            texts.append(text)
    return ", ".join(tools), "\n".join(texts)


def run_deep_director(
    brief: str,
    use_llm: bool,
    model_id: str | None = None,
    on_step: Callable[[str, str, str], None] | None = None,
) -> str:
    """A real Deep Agents supervisor with explicit specialist subagents, on a fixed budget.

    Streamed rather than invoked so each delegation is visible while it happens and so the step
    count and wall clock can actually be enforced. Returns the production brief it settled on,
    which the idea and design-document agents then have to work from.
    """
    if not use_llm:
        return "Offline mode: deterministic production plan selected."
    if DIRECTOR_TIMEOUT_SECONDS <= 0:
        return "Director pass skipped (DIRECTOR_TIMEOUT_SECONDS=0)."
    subagents = [
        {"name": "idea", "description": "Create the smallest compelling original game concept.", "system_prompt": IDEA_SYSTEM},
        {"name": "art", "description": "Create asset-safe canvas-first art direction.", "system_prompt": ART_SYSTEM},
        {"name": "code", "description": "Review implementation constraints for standalone Canvas HTML.", "system_prompt": CODE_SYSTEM},
        {"name": "qa", "description": "List launch and gameplay acceptance criteria.", "system_prompt": QA_SYSTEM},
    ]
    director_model = os.getenv("BEDROCK_DIRECTOR_MODEL_ID", "").strip() or model_id
    stream = None
    latest, steps, stopped = "", 0, ""
    try:
        director = create_deep_agent(
            model=_model(director_model, max_tokens=DIRECTOR_MAX_TOKENS),
            system_prompt=DIRECTOR_SYSTEM,
            subagents=subagents,
        )
        deadline = time.monotonic() + DIRECTOR_TIMEOUT_SECONDS
        stream = director.stream(
            {"messages": [("user", f"Coordinate a production plan for: {brief}")]},
            {"recursion_limit": max(4, DIRECTOR_MAX_STEPS * 2)},
            stream_mode="updates",
        )
        for update in stream:
            steps += 1
            for node, payload in (update or {}).items():
                tools, text = _director_step_summary(payload)
                if text:
                    latest = text
                if on_step:
                    on_step(str(node), tools, text)
            if steps >= DIRECTOR_MAX_STEPS:
                stopped = f"step budget ({DIRECTOR_MAX_STEPS} steps)"
                break
            if time.monotonic() >= deadline:
                stopped = f"time budget ({DIRECTOR_TIMEOUT_SECONDS}s)"
                break
    except Exception as error:
        return f"Director fallback: {type(error).__name__}: {error}"[:500]
    finally:
        if stream is not None:
            stream.close()
    if not latest:
        return f"Director produced no plan text (stopped by {stopped})." if stopped else "Director completed."
    return f"{latest}\n\n[총괄 감독이 {stopped}에 도달해 여기서 정리했습니다.]" if stopped else latest
