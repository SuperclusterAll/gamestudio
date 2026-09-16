"""Measure what the planning prompts actually produce, instead of eyeballing three samples.

Every test in tests/ stubs the model out. That is right for the code - a suite that called Bedrock
would be slow, flaky and expensive - but it means the prompts themselves are unmeasured, and the
prompts are where this pipeline's behaviour lives. Every prompt change in this project so far was
validated by running it three times and reading the titles:

    "별똥별 사냥꾼" / "별똥별 사냥꾼" / "별똥별 낚시꾼"   → 문제 발견
    "블록 낙하 합산" / "블록 붕괴 연쇄" / "불꽃 생존자"   → 고쳐짐

That found the problem and would never have caught it coming back. This is the missing half: a
fixed set of briefs, run through the planning stages, scored on things that can be checked rather
than judged.

    python -m game_studio.evaluate                 # 전체 (실제 모델 호출, 비용 발생)
    python -m game_studio.evaluate --case clone-tetris
    python -m game_studio.evaluate --baseline evals/baseline.json   # 회귀 비교

Deliberately deterministic. Almost every metric is a string or a count, not a second model's
opinion - an LLM judge would add its own variance to a measurement whose entire purpose is to
separate a prompt regression from noise. The one thing that would need judgement, "is this really
Tetris", is answered structurally instead: the prompt requires a faithful clone to name the game it
reproduces, so the check is whether that name reached reference_games.

Only the planning stages run. They are where the prompts under test live, they cost two or three
model calls per case, and including the code agent would multiply the bill by an order of magnitude
for behaviour these prompts do not control.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .agents import create_concept, genre_references, resolve_auto_genre
from .models import GameConcept, ImplementationPlan

EVAL_ROOT = Path(__file__).resolve().parents[2] / "evals"
DEFAULT_SET = EVAL_ROOT / "briefs.json"


def build_brief(case: dict) -> str:
    """The brief exactly as the dashboard would have written it."""
    genre = case.get("genre") or "auto"
    display = genre if genre != "auto" else "자동 기획"
    written = (case.get("brief") or "").strip()
    return (f"Requested genre: {display}\n"
            f"Player brief: {written or '사용자 경험 없이 독자적으로 기획하세요.'}\n"
            f"Run seed: {case['seed']}")


def normalize(text: str) -> str:
    return "".join(ch for ch in str(text).lower() if ch.isalnum())


@dataclass
class CaseResult:
    """One brief's outcome, and every check that could be made about it."""

    id: str
    brief: str
    ok: bool = True
    error: str = ""
    seconds: float = 0.0
    title: str = ""
    genre: str = ""
    expected_genre: str = ""
    core_loop: list[str] = field(default_factory=list)
    reference_games: list[str] = field(default_factory=list)
    checks: dict[str, bool] = field(default_factory=dict)
    contract: dict[str, int] = field(default_factory=dict)

    @property
    def score(self) -> float:
        return (sum(self.checks.values()) / len(self.checks)) if self.checks else 0.0


def evaluate_case(case: dict, model_id: str | None) -> CaseResult:
    """Run one brief through the planning stages and check what came back."""
    brief = build_brief(case)
    result = CaseResult(id=case["id"], brief=brief)
    started = time.monotonic()
    try:
        concept = create_concept(brief, True, model_id)
        plan = _plan_for(concept, brief, model_id)
    except Exception as error:  # a failed generation is a result, not a crash
        result.ok = False
        result.error = f"{type(error).__name__}: {error}"
        result.seconds = time.monotonic() - started
        return result
    result.seconds = time.monotonic() - started
    result.title = concept.title
    result.core_loop = list(concept.core_loop)
    result.reference_games = list(concept.reference_games)
    result.genre = plan.genre
    result.checks, result.expected_genre = _check(case, brief, concept, plan)
    result.contract = {"mechanics": len(plan.mechanics),
                       "acceptance_tests": len(plan.acceptance_tests),
                       "state_transitions": len(plan.state_transitions)}
    return result


def _plan_for(concept: GameConcept, brief: str, model_id: str | None) -> ImplementationPlan:
    """The implementation contract, built the way design_document_node builds it.

    Imported here rather than at module scope: graph.py pulls in the engine adapters and the whole
    tool surface, and an eval of the planning prompts has no use for any of it.
    """
    from .graph import _implementation_plan

    return _implementation_plan(concept, brief, "", model_id, None)


def _check(case: dict, brief: str, concept: GameConcept,
           plan: ImplementationPlan) -> tuple[dict[str, bool], str]:
    """Every check that can be made without a second model's opinion."""
    expect = case.get("expect") or {}
    checks: dict[str, bool] = {}

    # Which genre this run owed. An auto brief is assigned one from its seed, so the expectation is
    # "whatever the assignment said", not a value written into the eval set.
    expected = expect.get("genre") or resolve_auto_genre(brief)
    if expected:
        checks["genre_kept"] = normalize(expected) in normalize(plan.genre) \
            or normalize(plan.genre) in normalize(expected)

    # A genre that the reference table can anchor. Auto mode used to get no exemplars at all, which
    # is what collapsed four runs into the same game.
    if expect.get("assigned_genre"):
        checks["genre_assigned"] = bool(expected) and bool(genre_references(expected))

    # The fidelity check, answered structurally. A faithful clone has to name what it reproduces.
    if wanted := expect.get("references"):
        joined = normalize(" ".join(concept.reference_games))
        checks["clone_named"] = any(normalize(name) in joined for name in wanted)

    # The standing prompt asks for one to three real games whatever the brief, so an empty list
    # means the anchoring instruction was ignored.
    checks["has_references"] = bool(concept.reference_games)
    # A contract the code agent can finish inside its call budget. Schema allows up to 8 of each;
    # everything at the ceiling is a signal worth watching, not a failure on its own.
    checks["contract_sized"] = len(plan.mechanics) <= 6 and len(plan.acceptance_tests) <= 6
    return checks, expected


def _loop_words(core_loop: list[str]) -> set[str]:
    """The distinct words of one core loop, normalised individually so they stay words."""
    return {word for word in (normalize(token) for token in " ".join(core_loop).split()) if word}


def diversity(results: list[CaseResult], cases: list[dict]) -> dict[str, Any]:
    """How much the free-choice runs repeated each other.

    Titles alone are too forgiving - "별똥별 사냥꾼" and "별똥별 낚시꾼" are different strings and the
    same game - so the loop is compared too, on its words.

    Words, not characters. Normalising the whole loop first and then iterating it yields syllables,
    and any two Korean sentences share most of their syllables: the measure still moved in the
    right direction, but everything scored as similar to everything.
    """
    groups: dict[str, list[CaseResult]] = {}
    by_id = {case["id"]: case for case in cases}
    for result in results:
        group = ((by_id.get(result.id) or {}).get("expect") or {}).get("diverse_group")
        if group and result.ok:
            groups.setdefault(group, []).append(result)
    report: dict[str, Any] = {}
    for group, members in groups.items():
        titles = [normalize(m.title) for m in members]
        loops = [_loop_words(m.core_loop) for m in members]
        overlaps = [
            len(a & b) / max(1, len(a | b))
            for index, a in enumerate(loops) for b in loops[index + 1:]
        ]
        report[group] = {
            "cases": len(members),
            "distinct_titles": len(set(titles)),
            "genres": sorted({m.genre for m in members}),
            "distinct_genres": len({normalize(m.genre) for m in members}),
            "mean_loop_overlap": round(statistics.fmean(overlaps), 3) if overlaps else 0.0,
        }
    return report


def run(cases: list[dict], model_id: str | None) -> dict[str, Any]:
    results = [evaluate_case(case, model_id) for case in cases]
    passed = [r for r in results if r.ok]
    checks: dict[str, list[bool]] = {}
    for result in passed:
        for name, value in result.checks.items():
            checks.setdefault(name, []).append(value)
    return {
        "cases": len(results),
        "generated": len(passed),
        "failed": [{"id": r.id, "error": r.error} for r in results if not r.ok],
        "score": round(statistics.fmean([r.score for r in passed]), 3) if passed else 0.0,
        "metrics": {name: round(sum(values) / len(values), 3) for name, values in checks.items()},
        "diversity": diversity(results, cases),
        "seconds": round(sum(r.seconds for r in results), 1),
        "results": [asdict(r) for r in results],
    }


def render(report: dict[str, Any], baseline: dict[str, Any] | None) -> str:
    lines = [
        (f"평가 {report['generated']}/{report['cases']}건 생성 · "
         f"종합 {report['score']:.1%} · {report['seconds']}초"),
        "",
        f"{'지표':<18}{'점수':>8}" + ("      기준 대비" if baseline else ""),
    ]
    base_metrics = (baseline or {}).get("metrics", {})
    for name, value in sorted(report["metrics"].items()):
        delta = ""
        if name in base_metrics:
            diff = value - base_metrics[name]
            delta = f"      {diff:+.1%}" if abs(diff) > 0.001 else "      —"
        lines.append(f"{name:<18}{value:>8.1%}{delta}")

    for group, stats in report["diversity"].items():
        lines += ["", f"다양성 [{group}] {stats['cases']}건",
                  (f"  서로 다른 제목 {stats['distinct_titles']}/{stats['cases']} · "
                   f"장르 {stats['distinct_genres']}종 {stats['genres']}"),
                  f"  코어 루프 평균 중복도 {stats['mean_loop_overlap']:.1%} (낮을수록 좋음)"]

    lines += ["", "케이스별"]
    for entry in report["results"]:
        if not entry["ok"]:
            lines.append(f"  X  {entry['id']:<18} {entry['error'][:70]}")
            continue
        failed = [name for name, value in entry["checks"].items() if not value]
        mark = "OK " if not failed else "X  "
        lines.append(f"  {mark}{entry['id']:<18}{entry['genre']:<10}{entry['title'][:22]:<24}"
                     + (f"실패: {', '.join(failed)}" if failed else ""))
    if report["failed"]:
        lines += ["", f"생성 실패 {len(report['failed'])}건 — 지표에서 제외됨"]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="기획 프롬프트 평가. 실제 모델을 호출하므로 비용이 발생합니다.")
    parser.add_argument("--set", type=Path, default=DEFAULT_SET, help="평가 세트 JSON")
    parser.add_argument("--case", action="append", help="이 id만 실행 (반복 가능)")
    parser.add_argument("--model", default=None, help="Bedrock 모델 ID")
    parser.add_argument("--out", type=Path, help="결과 JSON 저장 경로")
    parser.add_argument("--baseline", type=Path, help="비교할 이전 결과 JSON")
    args = parser.parse_args(argv)

    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[2] / ".env")

    cases = json.loads(args.set.read_text(encoding="utf-8"))["cases"]
    if args.case:
        wanted = set(args.case)
        cases = [case for case in cases if case["id"] in wanted]
    if not cases:
        print("실행할 케이스가 없습니다.")
        return 1

    print(f"{len(cases)}건을 실행합니다. 케이스당 모델 호출 2회 — 실제 비용이 발생합니다.\n")
    report = run(cases, args.model)
    baseline = json.loads(args.baseline.read_text(encoding="utf-8")) if args.baseline else None
    print(render(report, baseline))

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n결과 저장: {args.out}")
    # A generation that failed outright is an error; a low score is a finding to look at.
    return 1 if report["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
