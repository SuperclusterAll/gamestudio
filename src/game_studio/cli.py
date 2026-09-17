"""Command line entrypoint for the browser-game studio."""

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
import webbrowser
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.tracers.langchain import wait_for_all_tracers
from langgraph.types import Command


def _root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent.parent
    return Path(__file__).resolve().parents[2]


def _bedrock_credentials_configured() -> bool:
    return bool(
        (os.getenv("AWS_ACCESS_KEY_ID") and os.getenv("AWS_SECRET_ACCESS_KEY"))
        or os.getenv("AWS_PROFILE")
        or os.getenv("AWS_WEB_IDENTITY_TOKEN_FILE")
    )


def create(args: argparse.Namespace) -> int:
    load_dotenv(_root() / ".env")
    from .graph import compiled_graph

    output_dir = Path(args.output_dir or os.getenv("GAME_OUTPUT_DIR", r"C:\dev\games")).resolve()
    workspace_dir = output_dir / f"cli-{uuid.uuid4().hex[:12]}"
    from .server import bedrock_credentials_configured
    if args.offline or not bedrock_credentials_configured():
        print('실제 게임 제작에는 Bedrock 인증이 필요합니다. 고정 게임으로 대체하지 않았습니다.')
        return 2
    run_config = {"configurable": {"thread_id": uuid.uuid4().hex}, "recursion_limit": 200}
    result = compiled_graph.invoke({
        "brief": args.brief,
        "output_dir": str(output_dir),
        "workspace_dir": str(workspace_dir),
        "use_llm": True,
        "code_model_id": args.code_model_id or args.model_id or os.getenv("BEDROCK_CODE_MODEL_ID", "global.anthropic.claude-sonnet-4-6"),
        "model_id": args.model_id
        or os.getenv("BEDROCK_MODEL_ID", "global.anthropic.claude-sonnet-4-6"),
        "generate_images": args.images,
        "repair_attempts": 0,
        "trace_notes": [],
    }, run_config)
    snapshot = compiled_graph.get_state(run_config)
    if snapshot.next:
        print("\n--- Game plan review ---")
        print(json.dumps(snapshot.values["design_document"], indent=2, ensure_ascii=False))
        if args.auto_approve:
            decision = "approve"
            comment = "CLI auto-approved"
        else:
            decision = input("Approve production? [y/N]: ").strip().lower()
            comment = input("Review note (optional): ").strip()
            decision = "approve" if decision in {"y", "yes", "approve"} else "reject"
        result = compiled_graph.invoke(Command(resume={"decision": decision, "comment": comment}), run_config)
    wait_for_all_tracers()
    if result.get("approval", {}).get("decision") == "rejected":
        print("Production stopped: design was rejected.")
        return 0
    game_path = Path(result["game_path"])
    print(f"Game ready: {game_path}")
    print(f"QA: {result['qa']['status']}")
    if args.open:
        webbrowser.open(game_path.as_uri())
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a complete browser canvas game with AI specialists.")
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("create", help="Generate and QA a browser game")
    command.add_argument("--brief", default="", help="Player/game brief; leave empty for autonomous planning")
    command.add_argument("--code-model-id", help="Bedrock model used to code and audit the game")
    command.add_argument("--output-dir", help="Directory for generated games")
    # There is no fallback game to fall back to - the module that built one was deleted, because
    # quietly substituting a template for a failed run is worse than reporting the failure. The
    # flag is kept so a script that passes it gets a clear refusal instead of an argparse error.
    command.add_argument("--offline", action="store_true",
                         help="아무것도 만들지 않고 종료합니다 (고정 게임 대체 경로는 없습니다)")
    command.add_argument("--images", action="store_true", help="Enable the optional Amazon Bedrock image asset")
    command.add_argument(
        "--model-id", help="Bedrock model ID (defaults to BEDROCK_MODEL_ID or Claude Sonnet 4.6)"
    )
    command.add_argument("--open", action="store_true", help="Open the generated index.html in the default browser")
    command.add_argument("--auto-approve", action="store_true", help="Skip the terminal HITL prompt")
    command.set_defaults(handler=create)
    args = parser.parse_args()
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
