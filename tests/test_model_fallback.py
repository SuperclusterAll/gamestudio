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
        "us.anthropic.claude-sonnet-4-6",  # the other profile: a spike, same weights, same answer
        "global.anthropic.claude-sonnet-4-5-20250929-v1:0",
        "global.anthropic.claude-haiku-4-5-20251001-v1:0",
        "us.amazon.nova-2-lite-v1:0",
        "us.amazon.nova-pro-v1:0",
        "us.amazon.nova-lite-v1:0",
    ], "Anthropic largest-first, then whatever is left by output ceiling"
    # Sonnet 4.6 is the primary here, so its rung is skipped rather than asked twice.
    assert "global.anthropic.claude-sonnet-4-6" == agents.DAILY_CAP_LADDER[0]

    # The WHOLE ladder, not its first rung. This middleware is built once when the agent is
    # assembled and never rebuilt, so a rung it was not given is a rung this build can never reach -
    # and a build that fell to Haiku and found Haiku spent too had nowhere left to go while the
    # account still had Nova unused.
    assert len(agents.DAILY_CAP_LADDER) > 1

    monkeypatch.setattr(agents, "DAILY_CAP_LADDER", ())
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
    monkeypatch.setattr(agents, "DAILY_CAP_LADDER",
                        ("global.anthropic.claude-haiku-4-5-20251001-v1:0",))
    assert agents._structured(Tiny, "s", "u", "global.anthropic.claude-sonnet-4-6").answer == "완성"
    assert calls == ["global.anthropic.claude-sonnet-4-6", *agents.DAILY_CAP_LADDER], (
        "us. shares the exhausted pool, so spending a call on it buys nothing")


def test_a_capped_run_can_be_made_to_stop_instead_of_changing_model(monkeypatch):
    """Whoever is comparing runs against a Sonnet baseline needs the option of no game over a game
    built by something else. One empty variable, no code change."""
    from game_studio import agents

    monkeypatch.delenv("BEDROCK_FALLBACK_MODEL_IDS", raising=False)
    monkeypatch.setattr(agents, "DAILY_CAP_LADDER", ())

    def capped(*args, **kwargs):
        raise throttle("Too many tokens per day, please wait before trying again.")

    monkeypatch.setattr(agents, "_structured_once", capped)
    with pytest.raises(RuntimeError, match="일일 한도"):
        agents._structured(Tiny, "s", "u", "global.anthropic.claude-sonnet-4-6")


def test_a_cap_walks_down_the_ladder_instead_of_stopping_at_the_first_spare(monkeypatch):
    """The failure this was written for: a QA stage fell to Haiku, Haiku was spent too, and the run
    ended on "한도가 초기화될 때까지 기다리세요" - while the account still had Nova unused.

    One spare model was one rung too few. Each rung is a separate quota pool, asked once, in the
    order of what the run loses by dropping to it.
    """
    from game_studio import agents

    monkeypatch.delenv("BEDROCK_FALLBACK_MODEL_IDS", raising=False)
    calls = []

    def capped(schema, system, user, model_id, on_chunk, max_tokens, outcome=None):
        calls.append(model_id)
        if "nova" not in model_id:
            raise throttle("Too many tokens per day, please wait before trying again.")
        return schema(answer="완성")

    monkeypatch.setattr(agents, "_structured_once", capped)
    assert agents._structured(Tiny, "s", "u", "global.anthropic.claude-sonnet-4-6").answer == "완성"
    assert calls == ["global.anthropic.claude-sonnet-4-6", "global.anthropic.claude-sonnet-4-5-20250929-v1:0", "global.anthropic.claude-haiku-4-5-20251001-v1:0", "us.amazon.nova-2-lite-v1:0"], calls
    assert "nova" in calls[-1], "a different vendor is a genuinely different pool"


def test_a_ladder_with_nothing_left_reports_every_model_it_asked(monkeypatch):
    """A whole account with nothing left has to end, and the message has to be actionable: the old
    one named a single inference profile, which sent whoever read the log looking at `us.` for a
    problem that had nothing to do with it."""
    from game_studio import agents

    monkeypatch.delenv("BEDROCK_FALLBACK_MODEL_IDS", raising=False)
    calls = []

    def capped(schema, system, user, model_id, on_chunk, max_tokens, outcome=None):
        calls.append(model_id)
        raise throttle("Too many tokens per day, please wait before trying again.")

    monkeypatch.setattr(agents, "_structured_once", capped)
    with pytest.raises(RuntimeError, match="예비 모델도 모두 소진") as raised:
        agents._structured(Tiny, "s", "u", "global.anthropic.claude-sonnet-4-6")
    assert calls == ["global.anthropic.claude-sonnet-4-6", "global.anthropic.claude-sonnet-4-5-20250929-v1:0", "global.anthropic.claude-haiku-4-5-20251001-v1:0",
                     "us.amazon.nova-2-lite-v1:0", "us.amazon.nova-pro-v1:0", "us.amazon.nova-lite-v1:0"], (
        "every rung is asked exactly once, and a pool already spent is never proposed again")
    for asked in calls:
        assert asked in str(raised.value), "the message lists what was actually tried"


def test_two_profiles_of_one_model_are_a_single_rung(monkeypatch):
    """`global.` and `us.` share the daily pool, so a ladder written with both in it would spend a
    guaranteed failed call at the moment the run can least afford one. Enforced in the lookup
    rather than trusted to whoever edits the list."""
    from game_studio.agents import daily_cap_fallback

    haiku_global = "global.anthropic.claude-haiku-4-5-20251001-v1:0"
    haiku_us = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
    assert daily_cap_fallback(haiku_us) != haiku_global, "the other profile is the same allowance"
    assert daily_cap_fallback("global.anthropic.claude-sonnet-4-6", [haiku_us]) != haiku_global

    # And a model that is not on the ladder at all still gets its first rung.
    assert daily_cap_fallback("us.amazon.nova-lite-v1:0").startswith("global.anthropic")


def validation_error(message: str) -> Exception:
    from botocore.exceptions import ClientError

    return ClientError({"Error": {"Code": "ValidationException", "Message": message}}, "Converse")


def test_the_output_ceiling_is_read_from_bedrock_rather_than_assumed(monkeypatch):
    """The ceilings differ by more than an order of magnitude across the models one run can end up
    on - Sonnet 4.6 takes 128000, Nova Pro takes 10000 - and the pipeline has one budget.
    CODE_MAX_TOKENS is 32000 because a whole game arrives as a single tool argument, so when the
    daily-cap ladder dropped a run onto Nova Pro the call was refused outright:

        ValidationException: The maximum tokens you requested exceeds the model limit of 10000.

    Not a throttle and not a retry - the request never ran. The budget is what the pipeline asks
    for; this is what the model hears.
    """
    from game_studio import agents

    assert agents.max_output_tokens("global.anthropic.claude-sonnet-4-6") == 128000
    assert agents.max_output_tokens("us.amazon.nova-pro-v1:0") == 10000
    # The profile is routing; the ceiling belongs to the weights underneath it.
    assert (agents.max_output_tokens("us.anthropic.claude-haiku-4-5-20251001-v1:0")
            == agents.max_output_tokens("global.anthropic.claude-haiku-4-5-20251001-v1:0"))
    # A model nobody measured is assumed small, because too small costs a shorter answer and too
    # large costs the whole call.
    assert agents.max_output_tokens("us.acme.brand-new-v9") == agents.DEFAULT_OUTPUT_LIMIT
    assert agents.DEFAULT_OUTPUT_LIMIT < 16000


def test_a_ceiling_nobody_measured_is_learned_from_the_refusal_itself(monkeypatch):
    """The table is a cache of a measurement, not a rule. Bedrock's refusal names the number, so
    being wrong about a model costs one call that generated nothing - rather than the run."""
    from game_studio import agents

    monkeypatch.setattr(agents, "_LEARNED_LIMITS", {})
    unknown = "us.acme.brand-new-v9"
    refusal = validation_error(
        "The maximum tokens you requested exceeds the model limit of 4096. Try again with a "
        "maximum tokens value that is lower than 4096.")

    assert agents.learn_output_limit(unknown, refusal) is True
    assert agents.max_output_tokens(unknown) == 4096
    # Learning the same thing twice is not news, and that is what stops a retry loop: the second
    # identical refusal falls through to the raise instead of asking again forever.
    assert agents.learn_output_limit(unknown, refusal) is False
    assert agents.learn_output_limit(unknown, throttle("Too many tokens per day")) is False


def test_a_refused_budget_is_retried_at_the_ceiling_instead_of_failing_the_node(monkeypatch):
    from game_studio import agents

    monkeypatch.setattr(agents, "_LEARNED_LIMITS", {})
    monkeypatch.delenv("BEDROCK_FALLBACK_MODEL_IDS", raising=False)
    asked = []

    def refuses_once(schema, system, user, model_id, on_chunk, max_tokens, outcome=None):
        asked.append(max_tokens)
        if len(asked) == 1:
            raise validation_error(
                "The maximum tokens you requested exceeds the model limit of 10000.")
        return schema(answer="완성")

    monkeypatch.setattr(agents, "_structured_once", refuses_once)
    assert agents._structured(Tiny, "s", "u", "us.acme.brand-new-v9",
                              max_tokens=32000).answer == "완성"
    assert asked[1] == 10000, f"the second attempt has to use the ceiling it was told: {asked}"


def test_a_plain_model_turn_rotates_instead_of_taking_the_graph_down(monkeypatch):
    """Rotation used to belong to _structured alone. Every plain .invoke() in the pipeline - the
    repair node, the reference reader, the director - walked nothing, so a quota error in any of
    them ended the run. A run does not care which node was holding the model when the account ran
    out."""
    from game_studio import agents

    monkeypatch.delenv("BEDROCK_FALLBACK_MODEL_IDS", raising=False)
    monkeypatch.setattr(agents, "_LEARNED_LIMITS", {})
    asked = []

    def capped(model, messages):
        asked.append(model.model_id)
        if "nova" not in model.model_id:
            raise throttle("Too many tokens per day, please wait before trying again.")
        return "완성"

    monkeypatch.setattr(agents, "_model",
                        lambda name, **kw: type("M", (), {"model_id": name})())
    answer = agents.invoke_with_fallbacks(
        "global.anthropic.claude-sonnet-4-6", [], max_tokens=4096, label="테스트", call=capped)
    assert answer == "완성"
    assert asked == ["global.anthropic.claude-sonnet-4-6", "global.anthropic.claude-sonnet-4-5-20250929-v1:0", "global.anthropic.claude-haiku-4-5-20251001-v1:0", "us.amazon.nova-2-lite-v1:0"], asked


def test_a_spike_takes_the_other_profile_before_it_changes_model(monkeypatch):
    """The two throttles need opposite answers and the policy has to be in one place, or the call
    sites drift apart. A spike is a queue in front of the same weights, so the cheap move - another
    profile of the same model - has to be tried before anything that changes what is produced."""
    from game_studio.agents import next_model

    sonnet = "global.anthropic.claude-sonnet-4-6"
    spike = throttle("Too many requests, please wait before trying again.")
    assert next_model([sonnet], spike) == "us.anthropic.claude-sonnet-4-6"
    # Profiles exhausted: a spike that has nowhere cheap left still gets the ladder rather than
    # ending the run.
    assert "sonnet-4-5" in next_model([sonnet, "us.anthropic.claude-sonnet-4-6"], spike)
    # A daily cap skips the profile entirely - it shares the pool that just ran out.
    cap = throttle("Too many tokens per day, please wait before trying again.")
    assert next_model([sonnet], cap) != "us.anthropic.claude-sonnet-4-6"
    # And nothing that another model cannot fix is rotated at all.
    assert next_model([sonnet], client_error("AccessDeniedException")) == ""
    assert next_model([sonnet], ValueError("게임 파일을 쓰지 못했습니다")) == ""


def test_below_anthropic_the_ladder_is_ordered_by_what_still_fits(monkeypatch):
    """Two rules, and the second only applies once the first runs out.

    Anthropic first, largest to smallest: those are the models the quality baselines describe, so
    every rung there is still a run comparable to the ones beside it. After that the run is on a
    different vendor and quality has stopped being what separates the candidates - finishing at all
    is. A whole game leaves as ONE tool argument, so the output ceiling decides whether it fits.

    Which is why Nova Pro sits BELOW Nova 2 Lite despite being the better model: 10000 tokens
    against 65535. The measurement is in _OUTPUT_LIMITS, taken from the account rather than a
    datasheet.
    """
    from game_studio.agents import DAILY_CAP_LADDER, max_output_tokens

    anthropic = [name for name in DAILY_CAP_LADDER if "anthropic" in name]
    rest = [name for name in DAILY_CAP_LADDER if "anthropic" not in name]
    assert DAILY_CAP_LADDER == (*anthropic, *rest), "the vendors do not interleave"

    ceilings = [max_output_tokens(name) for name in rest]
    assert ceilings == sorted(ceilings, reverse=True), (
        f"below Anthropic the order is the ceiling, largest first: {list(zip(rest, ceilings))}")
    assert max_output_tokens("us.amazon.nova-2-lite-v1:0") > max_output_tokens("us.amazon.nova-pro-v1:0")
    assert rest.index("us.amazon.nova-2-lite-v1:0") < rest.index("us.amazon.nova-pro-v1:0")

    # And the Anthropic rungs really are largest-first, which is also best-first here.
    assert anthropic == ["global.anthropic.claude-sonnet-4-6", "global.anthropic.claude-sonnet-4-5-20250929-v1:0", "global.anthropic.claude-haiku-4-5-20251001-v1:0"]
    assert max_output_tokens(anthropic[0]) > max_output_tokens(anthropic[1])
