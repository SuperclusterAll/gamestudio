"""What happens when the quota runs out mid-run, and the two very different things that means.

Bedrock reports both through one `ThrottlingException`, and they need opposite responses.

A **per-minute spike** is what the fallback is for. The profiles route differently, so a throttled
`global.anthropic.claude-sonnet-4-6` often means nothing about whether `us.` has room right now.
What that cost before: a run forty model calls into writing a game stopped with a half-written file
on disk and no way to finish it.

A **daily token cap** is not that, and this is the correction these tests exist to pin down.
Measured on the teaching account this runs on: `global.` and `us.` Sonnet 4.6 *and* Sonnet 4.5 all
answered "Too many tokens per day" inside the same minute, while Haiku 4.5 answered normally. That
pool is counted per model family across the account - the profiles share it. Walking the chain
there costs three more failed calls and ends by naming the last profile tried, which had nothing to
do with it: a quota error wearing a different name, the exact failure the fallback was built to
avoid.
"""

import pytest
from pydantic import BaseModel

from game_studio.agents import is_daily_cap, is_quota_error, model_fallbacks


class Tiny(BaseModel):
    answer: str


def client_error(code: str) -> Exception:
    from botocore.exceptions import ClientError

    return ClientError({"Error": {"Code": code, "Message": code}}, "Converse")


def test_the_fallback_is_the_same_model_through_the_other_profile(monkeypatch):
    """Never a different model. A game half written by Sonnet and half by something cheaper is
    worse than a run that stops and says why, so only the profile prefix moves."""
    monkeypatch.delenv("BEDROCK_FALLBACK_MODEL_IDS", raising=False)

    assert model_fallbacks("global.anthropic.claude-sonnet-4-6") == [
        "us.anthropic.claude-sonnet-4-6"]
    assert model_fallbacks("us.anthropic.claude-sonnet-4-6") == [
        "global.anthropic.claude-sonnet-4-6"]

    # eu. and apac. are deliberately not in the list. This account has no access to them, so
    # falling back onto one would turn a quota error into an AccessDeniedException - a worse
    # failure, further from the cause, at exactly the moment a run is already in trouble.
    assert model_fallbacks("eu.anthropic.claude-sonnet-4-6") == []
    # A bare model id is not an inference profile; there is no other door to try.
    assert model_fallbacks("anthropic.claude-sonnet-4-6-v1:0") == []
    assert model_fallbacks(None) == [] and model_fallbacks("  ") == []


def test_the_fallback_chain_can_be_set_without_touching_code(monkeypatch):
    """Which profiles an account can reach is an account fact, not a code fact."""
    monkeypatch.setenv("BEDROCK_FALLBACK_MODEL_IDS",
                       "us.anthropic.claude-sonnet-4-6, apac.anthropic.claude-sonnet-4-6")
    assert model_fallbacks("global.anthropic.claude-sonnet-4-6") == [
        "us.anthropic.claude-sonnet-4-6", "apac.anthropic.claude-sonnet-4-6"]

    # The primary must never appear in its own fallback list, or a quota failure retries the
    # profile that just refused it and the run ends one error later with nothing gained.
    monkeypatch.setenv("BEDROCK_FALLBACK_MODEL_IDS",
                       "global.anthropic.claude-sonnet-4-6,us.anthropic.claude-sonnet-4-6")
    assert model_fallbacks("global.anthropic.claude-sonnet-4-6") == [
        "us.anthropic.claude-sonnet-4-6"]


def test_only_a_quota_failure_is_worth_another_profile():
    """Switching profiles on any failure would hide real bugs behind a second identical failure and
    double the time taken to report them."""
    for code in ("ThrottlingException", "TooManyRequestsException",
                 "ServiceQuotaExceededException"):
        assert is_quota_error(client_error(code)), code
    for code in ("AccessDeniedException", "ValidationException", "ModelErrorException"):
        assert not is_quota_error(client_error(code)), code
    assert not is_quota_error(ValueError("게임 파일을 쓰지 못했습니다"))

    # langchain_aws re-raises some of these as plain exceptions, so the code survives only in the
    # message text. Measured: that is how a throttle actually arrives from a streaming call.
    assert is_quota_error(RuntimeError(
        "An error occurred (ThrottlingException) when calling the ConverseStream operation"))


def test_a_throttled_structured_call_finishes_on_the_next_profile(monkeypatch):
    """The whole point, end to end: the answer still comes back, and the caller cannot tell."""
    from game_studio import agents

    monkeypatch.delenv("BEDROCK_FALLBACK_MODEL_IDS", raising=False)
    tried: list[str | None] = []

    def once(schema, system, user, model_id, on_chunk, max_tokens, outcome=None):
        tried.append(model_id)
        if model_id == "global.anthropic.claude-sonnet-4-6":
            raise client_error("ThrottlingException")
        return schema(answer="완성")

    monkeypatch.setattr(agents, "_structured_once", once)
    result = agents._structured(Tiny, "s", "u", "global.anthropic.claude-sonnet-4-6")
    assert result.answer == "완성"
    assert tried == ["global.anthropic.claude-sonnet-4-6", "us.anthropic.claude-sonnet-4-6"]


def test_a_profile_switch_does_not_spend_the_schema_retry_budget(monkeypatch):
    """These two retries exist for opposite reasons and must not compete. A quota failure says
    nothing about the answer, so the model still gets its full allowance of attempts to satisfy the
    validator on the profile that can actually answer.

    This is the ordering the exception handlers are arranged around: `except Exception` placed above
    `except ValidationError` catches schema violations as quota failures and silently disables the
    retry that function exists for.
    """
    from game_studio import agents

    monkeypatch.delenv("BEDROCK_FALLBACK_MODEL_IDS", raising=False)
    calls: list[str | None] = []

    def once(schema, system, user, model_id, on_chunk, max_tokens, outcome=None):
        calls.append(model_id)
        if len(calls) == 1:
            raise client_error("ThrottlingException")
        if len(calls) <= 1 + agents.STRUCTURED_MAX_ATTEMPTS - 1:
            Tiny.model_validate({})  # the validator's own complaint, not a hand-made one
        return schema(answer="완성")

    monkeypatch.setattr(agents, "_structured_once", once)
    assert agents._structured(Tiny, "s", "u", "global.anthropic.claude-sonnet-4-6").answer == "완성"
    # One throttle plus a full set of schema attempts. Counting the throttle as an attempt would
    # end this run on the last validation error instead, one call short of the answer.
    assert len(calls) == 1 + agents.STRUCTURED_MAX_ATTEMPTS
    assert calls[0] == "global.anthropic.claude-sonnet-4-6"
    assert set(calls[1:]) == {"us.anthropic.claude-sonnet-4-6"}, (
        "the schema retries stay on the profile that answered")


def test_a_run_out_of_profiles_reports_the_quota_rather_than_something_vaguer(monkeypatch):
    """When there is nowhere left to go the original error has to survive. A throttle rewritten as
    "구조화 응답을 받지 못했습니다" sends whoever reads the log looking at the schema."""
    from game_studio import agents

    monkeypatch.delenv("BEDROCK_FALLBACK_MODEL_IDS", raising=False)

    def always_throttled(*args, **kwargs):
        raise client_error("ThrottlingException")

    monkeypatch.setattr(agents, "_structured_once", always_throttled)
    with pytest.raises(Exception, match="ThrottlingException"):
        agents._structured(Tiny, "s", "u", "global.anthropic.claude-sonnet-4-6")


def test_a_non_quota_failure_is_not_retried_on_another_profile(monkeypatch):
    from game_studio import agents

    monkeypatch.delenv("BEDROCK_FALLBACK_MODEL_IDS", raising=False)
    calls = []

    def broken(schema, system, user, model_id, on_chunk, max_tokens, outcome=None):
        calls.append(model_id)
        raise client_error("AccessDeniedException")

    monkeypatch.setattr(agents, "_structured_once", broken)
    with pytest.raises(Exception, match="AccessDenied"):
        agents._structured(Tiny, "s", "u", "global.anthropic.claude-sonnet-4-6")
    assert len(calls) == 1, "a permissions problem is not fixed by asking somewhere else"


def test_the_code_agent_carries_the_same_fallback(monkeypatch):
    """Where a quota limit actually stops a build. The code agent makes fifty of a run's model
    calls; the structured calls around it make one each, and losing one of those costs a node
    rather than a game."""
    from game_studio import agents
    from game_studio.code_agent import _fallback_middleware

    monkeypatch.delenv("BEDROCK_FALLBACK_MODEL_IDS", raising=False)
    middleware = _fallback_middleware("global.anthropic.claude-sonnet-4-6")
    assert len(middleware) == 1
    # The other profile first for a spike, the different model last for a daily cap. Middleware
    # cannot see which error it is reacting to, so the order is what makes the cheap, same-model
    # option the one that gets tried first.
    assert [getattr(m, "model_id", m) for m in middleware[0].models] == [
        "us.anthropic.claude-sonnet-4-6", agents.DAILY_CAP_MODEL_ID]

    monkeypatch.setattr(agents, "DAILY_CAP_MODEL_ID", "")
    assert _fallback_middleware("anthropic.claude-sonnet-4-6") == [], (
        "nowhere to go and nothing switched on adds no middleware at all")


def throttle(message: str) -> Exception:
    from botocore.exceptions import ClientError

    return ClientError({"Error": {"Code": "ThrottlingException", "Message": message}}, "Converse")


def test_a_daily_cap_is_told_apart_from_a_momentary_spike():
    """Both arrive as ThrottlingException and they need opposite responses. Measured on this
    account: `global.` and `us.` Sonnet 4.6 and Sonnet 4.5 all reported "Too many tokens per day"
    inside the same minute while Haiku answered normally - the daily pool is counted per model
    family across the account, so the profiles share it and none of them has room."""
    daily = throttle("Too many tokens per day, please wait before trying again.")
    assert is_quota_error(daily) and is_daily_cap(daily)

    spike = throttle("Too many requests, please wait before trying again.")
    assert is_quota_error(spike) and not is_daily_cap(spike), "a spike still gets the other profile"
    assert not is_daily_cap(ValueError("게임 파일을 쓰지 못했습니다"))


def test_a_daily_cap_skips_the_profile_chain_and_finishes_on_another_model(monkeypatch):
    """The failure that prompted all of this. The log said "us. 로 이어서 시도합니다" and then died on
    the same cap, naming a profile that had nothing to do with it.

    The profiles share the daily pool, so the chain is skipped entirely - and because the only move
    left is a different model, that is what happens. Sonnet's quality baselines do not describe a
    game Haiku finished, so the switch is announced and counted, never silent.
    """
    from game_studio import agents

    monkeypatch.delenv("BEDROCK_FALLBACK_MODEL_IDS", raising=False)
    calls = []

    def capped(schema, system, user, model_id, on_chunk, max_tokens, outcome=None):
        calls.append(model_id)
        if "sonnet" in model_id:
            raise throttle("Too many tokens per day, please wait before trying again.")
        return schema(answer="완성")

    monkeypatch.setattr(agents, "_structured_once", capped)
    assert agents._structured(Tiny, "s", "u", "global.anthropic.claude-sonnet-4-6").answer == "완성"
    assert calls == ["global.anthropic.claude-sonnet-4-6", agents.DAILY_CAP_MODEL_ID], (
        "us. shares the exhausted pool, so spending a call on it buys nothing")


def test_a_capped_run_can_be_made_to_stop_instead_of_changing_model(monkeypatch):
    """Whoever is comparing runs against a Sonnet baseline needs the option of no game over a game
    built by something else. One empty variable, no code change."""
    from game_studio import agents

    monkeypatch.delenv("BEDROCK_FALLBACK_MODEL_IDS", raising=False)
    monkeypatch.setattr(agents, "DAILY_CAP_MODEL_ID", "")

    def capped(*args, **kwargs):
        raise throttle("Too many tokens per day, please wait before trying again.")

    monkeypatch.setattr(agents, "_structured_once", capped)
    with pytest.raises(RuntimeError, match="일일 한도"):
        agents._structured(Tiny, "s", "u", "global.anthropic.claude-sonnet-4-6")


def test_a_cap_on_the_fallback_model_too_reports_it_rather_than_looping(monkeypatch):
    """A whole account with nothing left has to end, and say so in the one sentence that is true."""
    from game_studio import agents

    monkeypatch.delenv("BEDROCK_FALLBACK_MODEL_IDS", raising=False)
    calls = []

    def capped(schema, system, user, model_id, on_chunk, max_tokens, outcome=None):
        calls.append(model_id)
        raise throttle("Too many tokens per day, please wait before trying again.")

    monkeypatch.setattr(agents, "_structured_once", capped)
    with pytest.raises(RuntimeError, match="일일 한도"):
        agents._structured(Tiny, "s", "u", "global.anthropic.claude-sonnet-4-6")
    assert calls == ["global.anthropic.claude-sonnet-4-6", agents.DAILY_CAP_MODEL_ID], (
        "each model is asked once, and a model already tried is never appended again")
