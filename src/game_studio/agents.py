"""Model-backed specialist adapters and deterministic syntax checks."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import random
import re
import shutil
import subprocess
import time
from collections.abc import Callable, Sequence
from contextvars import ContextVar
from functools import lru_cache
from html.parser import HTMLParser
from pathlib import Path
from typing import TypeVar

import boto3
from botocore.config import Config as BotocoreConfig
from langchain_aws import ChatBedrockConverse
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.messages.tool import tool_call_chunk
from langsmith import traceable
from pydantic import ValidationError

from . import art_memory
from .models import ArtDirection, GameConcept, QAReport, ReferenceSketch
from .prompts import (
    ART_SYSTEM,
    DIRECTOR_SYSTEM,
    GENRE_KEYWORDS,
    GENRE_REFERENCES,
    IDEA_SYSTEM,
)
from .required_art import unused_sprites

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

# How this account actually pays for Bedrock, which decides what the dollar figure above means.
#
#   ondemand   (종량제)  - billed per token. The estimate is what the run plausibly costs, give or
#                          take the accuracy of the table and any private discount.
#   provisioned(정액제)  - Provisioned Throughput, an EDP commitment, or a shared teaching account.
#                          Capacity is bought up front by the hour, so a run's marginal cost is
#                          effectively zero and the real constraint is throughput, not money. The
#                          estimate is then a list-price *conversion*, useful for comparing two
#                          runs and meaningless as a bill.
#
# Stated rather than inferred, because the two are indistinguishable from inside the API: the same
# call on the same model returns the same token counts either way. Printing a dollar figure with no
# basis attached is how "런당 $2~4" ended up in a service document written for an account that is
# not billed per token at all.
PRICING_MODES = {
    "ondemand": "종량제 · 토큰당 과금",
    "provisioned": "정액제 · 약정/프로비저닝 (토큰당 과금 아님)",
}
DEFAULT_PRICING_MODE = "ondemand"


def pricing_mode() -> str:
    """"ondemand" or "provisioned", from BEDROCK_PRICING_MODE."""
    mode = os.getenv("BEDROCK_PRICING_MODE", "").strip().lower()
    return mode if mode in PRICING_MODES else DEFAULT_PRICING_MODE


def pricing_basis() -> dict[str, object]:
    """What the dashboard has to say next to any number it prints in dollars."""
    mode = pricing_mode()
    priced = mode == "ondemand"
    return {
        "mode": mode,
        "label": PRICING_MODES[mode],
        # Whether the figure is an estimate of a bill, or only a comparable conversion.
        "billed_per_token": priced,
        "note": ("온디맨드 정가표 기준 추정입니다. 실제 청구서가 아니며 사설 요율이 있으면 다릅니다."
                 if priced else
                 "약정/정액 계정이라 토큰당 청구가 없습니다. 아래 금액은 온디맨드 정가로 환산한 "
                 "비교용 수치이고, 실제 제약은 비용이 아니라 호출 수와 공유 처리량입니다."),
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
    """Return the LangChain Bedrock chat model used by every text specialist.

    `max_tokens` is what the pipeline WANTS, not what the model will hear: the ceilings differ by an
    order of magnitude across the models a run can end up on, and asking for more than one accepts
    is refused at validation rather than trimmed. See max_output_tokens.
    """
    name = model_id or os.getenv("BEDROCK_MODEL_ID", DEFAULT_MODEL_ID)
    return _build_model(
        name,
        os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
        min(max_tokens, max_output_tokens(name)),
    )


# The inference profiles a Bedrock model is reachable through. They are the same weights behind
# different routing, and - this is the point - **different quota pools**. On a shared account a
# throttled `global.` profile says nothing about whether `us.` has room, so a run that stops because
# one pool is full can often finish by asking through the other.
#
# Retrying is not the same thing. MODEL_RETRY already backs off and tries again on the same profile,
# which is right for a brief spike and useless when the pool is genuinely exhausted for the hour.
# Only the two this account is known to reach. eu./apac. profiles exist but a model has to be
# enabled per region, and falling back onto one nobody has access to turns a quota error into an
# AccessDenied error - a worse failure wearing a different name. Override with
# BEDROCK_FALLBACK_MODEL_IDS if another profile is enabled.
_PROFILE_PREFIXES = ("global.", "us.")


def model_fallbacks(model_id: str | None) -> list[str]:
    """The other profiles to try, in order, when this one has no room left.

    Only ever the same model through another door. Falling back to a *different* model would change
    what the run produces, silently, at the point where it is hardest to notice - a game half built
    by Sonnet and half by something else is worse than a run that stops and says why.
    """
    primary = (model_id or "").strip()
    if not primary:
        return []
    if override := os.getenv("BEDROCK_FALLBACK_MODEL_IDS", "").strip():
        return [name.strip() for name in override.split(",")
                if name.strip() and name.strip() != primary]
    for prefix in _PROFILE_PREFIXES:
        if primary.startswith(prefix):
            bare = primary[len(prefix):]
            return [f"{other}{bare}" for other in _PROFILE_PREFIXES if other != prefix]
    return []


# What "this pool is full" looks like coming back from Bedrock, as opposed to a transient blip.
# ServiceQuotaExceeded is the explicit one; throttling is the shape a shared account hits first,
# and on a teaching account with several people running at once it is the common case.
QUOTA_ERROR_CODES = frozenset({
    "ThrottlingException", "TooManyRequestsException", "ServiceQuotaExceededException",
})


# Bedrock reports two very different things through the same exception, and only one of them has
# a way out. A per-minute throttle is a spike: another profile, or the same one a moment later,
# usually has room. A daily token cap does not move for hours.
#
# Measured on this account while chasing a failed run: `global.` and `us.` Sonnet 4.6 *and* Sonnet
# 4.5 all reported "Too many tokens per day" within the same minute, while Haiku 4.5 answered
# normally. So this pool is counted per model family across the whole account - the profiles share
# it, and every profile in the fallback chain is already spent before the first one is tried.
#
# Walking the chain there buys nothing and costs something real: three more failed calls, and a
# final message naming the last profile tried, which sends whoever reads the log looking at `us.`
# for a problem that has nothing to do with it.
_DAILY_CAP_MARKERS = ("tokens per day", "requests per day")

# Where a run goes when the daily pool is gone, best first. Unlike a profile switch these are
# DIFFERENT models, and that is a real cost: the studio's quality baselines were all measured on
# Sonnet, so a game finished further down this ladder is not comparable to one beside it. It is
# here anyway because the alternative is not a better game - it is no game at all until the cap
# resets, which on a teaching account means the rest of the day.
#
# It used to be ONE model, and that was one rung too few. A QA stage fell to Haiku, Haiku was spent
# too, and the run ended on "한도가 초기화될 때까지 기다리세요" with nowhere else to go - while the
# account still had Nova sitting unused.
#
# The order is two rules, and the second one only applies after the first runs out:
#
#   1. Anthropic first, largest to smallest - Sonnet 4.6, Sonnet 4.5, Haiku 4.5. These are the
#      models the quality baselines actually describe, so every rung here is a run that is still
#      comparable to the ones beside it.
#   2. Then whatever is left, by MEASURED OUTPUT CEILING, largest first. By this point the run is
#      on a different vendor and quality is no longer what separates the candidates - finishing at
#      all is. A whole game leaves as one tool argument, so the ceiling is what decides whether it
#      fits: Nova 2 Lite takes 65535 tokens and Nova Pro takes 10000. Pro is the better model and
#      it is BELOW 2 Lite here for that reason. (See _OUTPUT_LIMITS for where those numbers come
#      from - they were asked for, not looked up.)
#
# Nova Lite is last: the same 10000 ceiling as Pro and less behind it.
#
# The primary model is usually the first rung, and listing it costs nothing - daily_cap_fallback
# skips a pool already spent. It is listed so the ladder is still right for a run that STARTED
# somewhere else, on Haiku or on Nova.
#
# One entry per pool, never two profiles of the same model: `global.` and `us.` share the quota
# (see _DAILY_CAP_MARKERS above), so a second profile is a guaranteed failed call at the exact
# moment the run can least afford one. daily_cap_fallback enforces that rather than trusting this
# list to be written carefully.
#
# Every call is counted per model onto the run's usage, so the manifest says which model built which
# part rather than leaving a quietly different game to be discovered later. Set the variable empty
# to switch this off and have a capped run stop instead.
DAILY_CAP_LADDER = tuple(
    name.strip()
    for name in os.getenv(
        "BEDROCK_DAILY_CAP_MODEL_IDS",
        "global.anthropic.claude-sonnet-4-6,"
        "global.anthropic.claude-sonnet-4-5-20250929-v1:0,"
        "global.anthropic.claude-haiku-4-5-20251001-v1:0,"
        "us.amazon.nova-2-lite-v1:0,"
        "us.amazon.nova-pro-v1:0,"
        "us.amazon.nova-lite-v1:0",
    ).split(",")
    if name.strip()
)


def _quota_pool(model_id: str) -> str:
    """The name a model's daily allowance is counted under.

    The inference profile is routing, not accounting: `global.` and `us.` Sonnet 4.6 hit the same
    ceiling within the same minute. So the pool is the model underneath, and two entries that
    differ only by prefix are one rung, not two.
    """
    bare = (model_id or "").strip()
    for prefix in _PROFILE_PREFIXES:
        if bare.startswith(prefix):
            return bare[len(prefix):]
    return bare


def daily_cap_fallback(model_id: str | None, tried: Sequence[str] = ()) -> str:
    """The next model to finish the run on, or "" when the ladder is spent.

    `tried` is every model this run has already asked - including the ones it fell back to - so a
    second cap moves DOWN the ladder instead of proposing the rung that just failed.
    """
    spent = {_quota_pool(name) for name in (model_id or "", *tried) if name}
    for candidate in DAILY_CAP_LADDER:
        if _quota_pool(candidate) not in spent:
            return candidate
    return ""


# The most output each model will accept in one turn, measured against this account rather than
# read off a datasheet: a maxTokens over the limit is rejected at validation, before any generation,
# so asking is free and exact.
#
#   sonnet 4.6   128000      haiku 4.5     64000      nova pro      10000
#   sonnet 4.5    64000      nova 2 lite   65535      nova lite     10000
#
# This exists because the ceilings are not close to each other and the pipeline has one budget.
# CODE_MAX_TOKENS is 32000 - it has to be, a whole game arrives as one tool argument - and when the
# daily-cap ladder dropped a run onto Nova Pro, Bedrock refused the call outright:
#
#   ValidationException: The maximum tokens you requested exceeds the model limit of 10000.
#
# Not a throttle, not a retry: the request never ran. So the budget is the pipeline's ASK and this
# is what the model will hear, and a run that falls to a smaller model gets smaller answers rather
# than no answers.
#
# Keyed by the model under the inference profile, because the profile is routing and the ceiling
# belongs to the weights.
_OUTPUT_LIMITS = {
    "anthropic.claude-sonnet-4-6": 128000,
    "anthropic.claude-sonnet-4-5-20250929-v1:0": 64000,
    "anthropic.claude-haiku-4-5-20251001-v1:0": 64000,
    "amazon.nova-pro-v1:0": 10000,
    "amazon.nova-2-lite-v1:0": 65535,
    "amazon.nova-lite-v1:0": 10000,
}
# What a model nobody measured is assumed to accept. Deliberately low: too small costs a shorter
# answer, too large costs the call. 8192 is the smallest ceiling this pipeline has ever met with
# room to spare, and anything real is learned on the first refusal anyway - see learn_output_limit.
DEFAULT_OUTPUT_LIMIT = int(os.getenv("BEDROCK_DEFAULT_MAX_OUTPUT", "8192"))
# Ceilings discovered at runtime from Bedrock's own refusal. Process-local: a table that has to be
# edited before a new model works is a table that will be out of date exactly when it matters.
_LEARNED_LIMITS: dict[str, int] = {}
_OUTPUT_LIMIT_ERROR = re.compile(r"exceeds the model limit of (\d+)")


def max_output_tokens(model_id: str | None) -> int:
    """The largest max_tokens this model accepts."""
    pool = _quota_pool(model_id)
    return _LEARNED_LIMITS.get(pool) or _OUTPUT_LIMITS.get(pool, DEFAULT_OUTPUT_LIMIT)


def learn_output_limit(model_id: str | None, error: BaseException) -> bool:
    """Record the ceiling Bedrock just named, and say whether it is news.

    The refusal carries the exact number, so a model this pipeline has never seen costs one failed
    call - which ran nothing and generated nothing - and then works. The alternative is a hardcoded
    table that is correct until the next model is added.
    """
    found = _OUTPUT_LIMIT_ERROR.search(str(error))
    if not found:
        return False
    pool, limit = _quota_pool(model_id), int(found.group(1))
    if _LEARNED_LIMITS.get(pool) == limit:
        return False
    _LEARNED_LIMITS[pool] = limit
    return True


def is_daily_cap(error: BaseException) -> bool:
    """Whether this throttle is the daily allowance rather than a momentary spike."""
    text = str(error).lower()
    return any(marker in text for marker in _DAILY_CAP_MARKERS)


def is_quota_error(error: BaseException) -> bool:
    """Whether another inference profile is worth trying for this failure."""
    from botocore.exceptions import ClientError

    if isinstance(error, ClientError):
        return error.response.get("Error", {}).get("Code", "") in QUOTA_ERROR_CODES
    # langchain_aws wraps some of these, so the code is only reachable in the message text.
    return any(code in str(error) for code in QUOTA_ERROR_CODES)


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
def next_model(tried: Sequence[str], error: BaseException) -> str:
    """The model to try after this failure, or "" when another model would not help.

    The rotation policy, in one place, because it was in two and they did not agree: _structured
    walked profiles and then the cap ladder, while every plain .invoke() in the pipeline walked
    nothing at all and took the whole graph down with it. A run does not care which node was
    holding the model when the account ran out.

    Three cases, and only the first two have a way forward:

    * a per-minute spike - another inference profile of the SAME model, which is a different queue
      in front of the same weights, so the answer is unchanged;
    * the daily cap - the profiles share that pool (see _DAILY_CAP_MARKERS), so the only move is
      down the ladder to a different model;
    * anything else - a permissions problem, a bad request, a bug. Asking somewhere else turns one
      honest error into several and reports the last one.
    """
    if not tried or not is_quota_error(error):
        return ""
    current = tried[-1]
    if not is_daily_cap(error):
        remaining = [name for name in model_fallbacks(current) if name not in tried]
        if remaining:
            return remaining[0]
        # A spike with no profile left is still a run that can finish on another model.
    return daily_cap_fallback(current, tried)


def invoke_with_fallbacks(
    model_id: str | None,
    messages,
    *,
    max_tokens: int,
    label: str,
    call=None,
):
    """One model turn that survives losing its model.

    `call(model, messages)` runs the turn - streaming, structured, plain, whatever the caller needs
    - and defaults to a plain invoke. Everything around it is the part worth sharing: clamping the
    budget to what this model accepts, learning that ceiling when the guess was wrong, and walking
    to the next model when the account has nothing left on this one.

    Raises only when nothing is left to try, which is what the caller then decides to do about.
    """
    runner = call or (lambda model, sent: model.invoke(sent))
    tried = [(model_id or os.getenv("BEDROCK_MODEL_ID", DEFAULT_MODEL_ID)).strip()]
    while True:
        try:
            return runner(_model(tried[-1], max_tokens=max_tokens), messages)
        except Exception as error:
            # Refused before generating anything, and the refusal names the number: worth one free
            # call to find out. Cannot spin - a ceiling already learned is not learned twice.
            if learn_output_limit(tried[-1], error):
                max_tokens = min(max_tokens, max_output_tokens(tried[-1]))
                _note(label, f"{tried[-1]} 의 출력 상한은 {max_tokens} 토큰입니다. "
                             "그 상한에 맞춰 다시 요청합니다.")
                continue
            following = next_model(tried, error)
            if not following:
                raise
            _note(label, f"{tried[-1]} 이(가) 한도에 걸렸습니다. {following} 로 이어서 시도합니다"
                         + (" — 다른 모델이라 결과물 품질이 평소와 다를 수 있습니다."
                            if is_daily_cap(error) else "."))
            tried.append(following)


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
    # The same model through another inference profile, tried only when this one's quota is gone.
    # Not a different model: a run half built by Sonnet and half by something else is worse than a
    # run that stops and says why.
    profiles, profile_index = [model_id, *model_fallbacks(model_id)], 0
    # Attempts count schema failures only. A throttle says nothing about the answer, so moving to
    # another profile must not eat an attempt the model needs to satisfy the validator - otherwise
    # one unlucky throttle costs the retry this function exists for. The loop still ends: a profile
    # is only ever left behind, and there are two of them.
    attempt = 1
    while attempt <= STRUCTURED_MAX_ATTEMPTS:
        outcome: dict[str, bool] = {}
        try:
            return _structured_once(schema, system, prompt, profiles[profile_index], on_chunk,
                                    budget, outcome)
        except ValidationError as error:
            if attempt == STRUCTURED_MAX_ATTEMPTS:
                raise
            complaints = "; ".join(
                f"{'.'.join(str(p) for p in issue['loc'])}: {issue['msg']}"
                for issue in error.errors()
            )
            _note(schema.__name__, f"구조화 응답 검증 실패({attempt}/{STRUCTURED_MAX_ATTEMPTS}): {complaints[:200]} — 다시 요청합니다.")
            attempt += 1
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
        except Exception as error:
            # Below ValidationError on purpose: a schema violation is an Exception too, and catching
            # it here first would silently disable the retry above - the thing this function exists
            # for. Only a quota failure reaches this, and only to move to another profile.
            #
            # Except this one, which is neither. Bedrock refuses a maxTokens above the model's
            # ceiling at validation - nothing ran, nothing was generated - and the refusal names the
            # number. So it is read and remembered, and the call is simply made again inside it.
            # The table above is a cache of a measurement, not a rule: this is what makes being
            # wrong about a model cost one free call instead of the run.
            #
            # It cannot spin: learn_output_limit returns False for a ceiling already recorded, so a
            # second identical refusal falls through to the raise below.
            if learn_output_limit(profiles[profile_index], error):
                # The budget itself, not only the client _model builds from it. It is also what the
                # truncation path grows from, and a budget that believes in room the model does not
                # have would keep asking for a bigger answer than it can ever be given.
                budget = min(budget, max_output_tokens(profiles[profile_index]))
                _note(schema.__name__,
                      f"{profiles[profile_index]} 의 출력 상한은 {budget} 토큰입니다. "
                      "그 상한에 맞춰 다시 요청합니다.")
                continue
            if not is_quota_error(error):
                raise
            if is_daily_cap(error):
                # The rest of the profile chain shares this pool and is already spent, so the only
                # move left is a different model. Announced rather than done quietly: the run is
                # about to produce something the Sonnet baselines do not describe.
                # Everything asked so far, not just the model that failed: a second cap has to
                # move DOWN the ladder, and offering the rung that just refused is how a run used
                # to bounce between two spent models until the attempts ran out.
                cap_model = daily_cap_fallback(profiles[profile_index], profiles)
                if not cap_model:
                    raise RuntimeError(
                        "오늘 쓸 수 있는 토큰을 모두 썼습니다(일일 한도). 인퍼런스 프로파일은 같은 "
                        "한도를 나눠 쓰므로 바꿔도 소용이 없고, 예비 모델도 모두 소진했습니다 "
                        f"(시도: {', '.join(profiles)}) — 한도가 초기화될 때까지 기다리거나, "
                        "BEDROCK_DAILY_CAP_MODEL_IDS에 한도가 남은 모델을 추가하세요."
                    ) from error
                profiles.append(cap_model)
                profile_index = len(profiles) - 1
                _note(schema.__name__,
                      f"{profiles[0]} 의 일일 토큰 한도를 모두 썼습니다(프로파일 공용). "
                      f"{cap_model} 로 이어서 만듭니다 — 이 결과물은 일부를 다른 모델이 "
                      "만들었으므로 품질이 평소와 다를 수 있습니다.")
                continue
            if profile_index + 1 >= len(profiles):
                raise
            profile_index += 1
            _note(schema.__name__,
                  f"{profiles[profile_index - 1]} 한도에 걸렸습니다. "
                  f"{profiles[profile_index]} 로 이어서 시도합니다.")
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
    # What is NOT here any more: a list of the studio's recent games with an instruction to differ
    # from them in at least two of loop, controlled object and win condition.
    #
    # It was aimed at a real problem - inside one genre the model reaches for the same design - and
    # the cure got worse than the disease as the output folder filled up. The constraint is
    # RELATIVE TO EVERYTHING ALREADY BUILT, so each new run had to dodge one more game than the
    # last, and the only space left to dodge into is the space of designs nobody wants. Testing
    # made it worse: every test run added another game to avoid. Games got stranger the more the
    # pipeline was exercised, which is exactly backwards.
    #
    # Variety across runs is worth having and this was the wrong lever for it. What remains is the
    # one that spreads runs WITHOUT pushing any single run somewhere odd: the run seed picks the
    # genre (resolve_auto_genre) and then picks which exemplars that genre opens with, so two runs
    # differ because they started from different places rather than because one was forbidden the
    # other's answer.
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


# How many uploaded references one run reads, and how much room the answer gets.
#
# One call covers the whole upload, so more images cost tokens rather than calls - but a person who
# attaches a dozen screenshots is describing a dozen different games, and the answer is an average
# of all of them. Three is enough to pin a look without blurring it.
MAX_REFERENCE_IMAGES = int(os.getenv("MAX_REFERENCE_IMAGES", "3"))
REFERENCE_MAX_TOKENS = int(os.getenv("REFERENCE_MAX_TOKENS", "2000"))
REFERENCE_MAX_ATTEMPTS = 3


REFERENCE_SYSTEM = """You are reading a reference image a game designer uploaded to show what they
have in mind. Describe the STRUCTURE and the LOOK, so the agents that plan and build the game can
work from your words alone - they never see the picture.

What is wanted from the picture is mostly STRUCTURE, not decoration. Read the level: where the
ground, platforms, walls and gaps are, how a player gets from where they start to wherever the
level ends, and where the enemies sit or come from. Those are obvious at a glance and laborious to
write down, which is why someone uploaded a picture instead of typing. The palette and the art style
matter too, but they are one stage's concern; the layout is every stage's.

Describe positions on the screen plainly - "five platform rows with gaps at alternating ends",
"a wall around all four sides", "enemies on the upper rows only". Write every field in English.

Never name or describe a specific copyrighted character, title or logo, and never say to copy one.
Name shapes, colours and roles - "a round green creature with a pale belly", not a mascot's name.
Mechanics and layout are fair to describe; a particular company's artwork is not.

For `objects`, list only things the GAME POSITIONS SEPARATELY - the player, each enemy type, a
platform, a pickup. Describe each one ALONE, with no background, no floor and no shadow, because
these become sprite prompts and anything behind the object is cut out with it and follows it around
the screen. Scenery that is painted into the backdrop is not an object.

Every field is short: at most 200 characters, one line per object. These become prompt clauses, not
documentation."""


def describe_reference(images: list[bytes], model_id: str | None = None) -> ReferenceSketch | None:
    """Turn uploaded reference images into the words the rest of the run reads.

    One vision call for the whole upload, at the start of the run. `_structured` cannot do this -
    it builds a text-only message pair - so the retry it provides is reproduced here rather than
    borrowed: a first answer that misses the length limits is normal, and re-asking with the
    validator's own complaint attached is what fixes it.

    Returns None rather than raising. A reference is an improvement to a run, never a requirement
    for one, and a person who uploaded a picture that could not be read should still get their game.
    """
    if not images:
        return None
    content: list[dict] = [{"type": "text", "text": "Describe this reference image."}]
    for raw in images[:MAX_REFERENCE_IMAGES]:
        content.append({"type": "image", "source_type": "base64",
                        "mime_type": _image_mime(raw), "data": base64.b64encode(raw).decode()})
    note = ""
    for attempt in range(1, REFERENCE_MAX_ATTEMPTS + 1):
        try:
            # Rotated like every other call. This one is optional to the run, so losing it is not
            # fatal - but "이미지 없이 진행합니다" on a day the primary model is capped throws away
            # the upload for no reason, when the next model down would have read it fine.
            return invoke_with_fallbacks(
                model_id,
                [{"role": "system", "content": REFERENCE_SYSTEM + note},
                 {"role": "user", "content": content}],
                max_tokens=REFERENCE_MAX_TOKENS,
                label="참조 이미지",
                call=lambda model, sent: model.with_structured_output(
                    ReferenceSketch).invoke(sent),
            )
        except ValidationError as error:
            complaints = "; ".join(
                f"{'.'.join(str(part) for part in issue['loc'])}: {issue['msg']}"
                for issue in error.errors())
            _note("참조 이미지", f"참조 분석 검증 실패({attempt}/{REFERENCE_MAX_ATTEMPTS}): "
                             f"{complaints[:160]}")
            note = (f"\n\n[The previous answer failed validation]\n{complaints}\n"
                    "Answer again, shorter, within every limit.")
        except Exception as error:
            _note("참조 이미지", f"참조 이미지를 읽지 못했습니다: {type(error).__name__}. "
                             "이미지 없이 진행합니다.")
            return None
    return None


def _image_mime(raw: bytes) -> str:
    """The format Bedrock is told the bytes are in, read from the bytes themselves.

    Trusting an uploaded filename would let a caller mislabel the payload, and Bedrock rejects the
    call when the declared type does not match what it decodes.
    """
    if raw[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if raw[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if raw[:4] == b"RIFF" and raw[8:12] == b"WEBP":
        return "image/webp"
    if raw[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    return "image/png"


def create_art(
    concept: GameConcept,
    use_llm: bool,
    model_id: str | None = None,
    findings: list[str] | None = None,
    existing_sprites: list[str] | None = None,
    store_root: str | Path | None = None,
    genre: str = "",
    reference: dict | None = None,
) -> ArtDirection:
    """Plan the visual system. With findings, this is a revision after QA rejected the build, so the
    plan has to change rather than come back the same."""
    if not use_llm:
        raise RuntimeError("아트 기획 모델 연결이 필요합니다.")
    user = f"Game concept:\n{concept.model_dump_json(indent=2)}"
    # The picture the person uploaded, in the words it was turned into. This is the only place
    # the art director gets to see what they had in mind rather than infer it from a paragraph,
    # so the style and palette it names are taken as the answer rather than as a suggestion.
    if reference:
        sketch = ReferenceSketch.model_validate(reference)
        user += (
            chr(10) + chr(10) + sketch.as_brief() + chr(10)
            + "이 참조가 이 게임이 어떻게 보여야 하는지를 정합니다. style_token과 palette는 "
              "위의 화풍·색을 따르고, asset_plan은 위 '등장 객체'를 출발점으로 삼되 이 게임에 "
              "실제로 필요한 것만 남기세요. 특정 작품의 캐릭터를 그대로 재현하지는 마세요."
        )
    if findings:
        user += (
            "\n\n[아트 방향 재수립] 이 게임은 검증을 통과하지 못했고, 지적 사항은 다음과 같습니다:\n"
            + json.dumps(findings, ensure_ascii=False)
            + f"\n이미 생성된 스프라이트: {json.dumps(existing_sprites or [], ensure_ascii=False)}\n"
            "asset_plan을 이 지적에 맞게 고치세요. 게임에 실제로 필요한데 빠진 객체를 추가하고, "
            "쓰이지 않을 객체는 빼고, 이미 생성된 스프라이트는 그 이름을 그대로 유지하세요. "
            "같은 계획을 반복하지 마세요."
        )
    # What this studio has already learned about asking this image model for this kind of object.
    # Prompts it wrote itself, for images it generated, filtered to the ones that came back usable -
    # the only corpus that can answer "what worked here, at this size, on this model". The wording
    # transfers, not the picture: a remembered prompt produces a new sprite, not the old one.
    if remembered := art_memory.recall(store_root, visual_direction=concept.visual_direction,
                                       genre=genre):
        user += (
            "\n\n이 스튜디오가 전에 쓴 이미지 프롬프트 중 결과가 쓸 만했던 것들입니다:\n"
            + art_memory.as_examples(remembered)
            + "\n표현 방식을 참고하세요 — 무엇이 통했는지가 여기 있습니다. 그대로 복사하지 말고, "
              "이 게임의 객체에 맞게 다시 쓰세요. asset_plan의 각 항목은 그 객체가 무엇인지와 "
              "어떻게 생겼는지만 적고, 배경·바닥·그림자는 절대 넣지 마세요."
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
        # Rotated too. Losing this one is survivable - every stage after it works without a brief
        # - but "총괄 감독 실패" on a day the primary model is capped throws away a call the run
        # already decided was worth making, when the next model down would have answered.
        answer = invoke_with_fallbacks(
            director_model,
            [("system", DIRECTOR_SYSTEM), ("human", user)],
            max_tokens=DIRECTOR_MAX_TOKENS,
            label="총괄 감독",
        )
    except Exception as error:
        if on_step:
            on_step("director", "",
                    f"총괄 감독 실패 (제작 지침 없이 진행합니다): {type(error).__name__}: {error}"[:300])
        return ""
    return _content_text(answer).strip()[:DIRECTOR_BRIEF_CHARS]
