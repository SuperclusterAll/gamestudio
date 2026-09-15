"""Godot engine support: project scaffolding, headless verification, and an optional web export.

Godot changes what verification can be. The HTML pipeline checks a game it can never run - keyword
probes and a JavaScript parse - so "the win condition can never trigger" is something only a model
reading the source can even guess at. Godot ships a headless binary that imports the project,
compiles every script and runs the game, and it reports what went wrong with a file and a line. A
build that cannot start is caught here for free, before any model is asked to read anything.

Two measured facts about the binary shape everything below:

  * `--headless --quit-after N` runs the game and prints parse errors, runtime errors and missing
    resources to stderr with a backtrace - and still exits 0. The exit code carries no information
    about the game; the output is the only signal.
  * `--headless --check-only --script res://x.gd` exits 1 on a parse error. That is the one
    reliable exit code available, so script syntax is gated with it per file.

Web export is opportunistic. It needs `web_nothreads_{debug,release}.zip` under the user's
export_templates directory, which Godot only distributes inside a ~1GB all-platform archive, so a
machine without them still produces a complete, runnable project - it just cannot be played in the
dashboard's iframe.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

# Where the editor binary is looked for. The console build is preferred on Windows: the plain .exe
# detaches from the console and its stderr - the only channel that says what is wrong with the
# game - never reaches us.
GODOT_HOME = Path(os.getenv("GODOT_HOME", r"C:\dev\godot"))
GODOT_EXECUTABLE = os.getenv("GODOT_EXECUTABLE", "").strip()
# How long a verification run is allowed to play the game before it is stopped and judged. The game
# is asked to run for this many seconds of engine time; anything that crashes does so immediately.
GODOT_RUN_SECONDS = int(os.getenv("GODOT_RUN_SECONDS", "5"))
# Wall-clock ceilings. Importing a fresh project costs a filesystem scan, so the first run is the
# slow one; a hung game must not hold a production run open forever.
GODOT_IMPORT_TIMEOUT = int(os.getenv("GODOT_IMPORT_TIMEOUT", "180"))
GODOT_EXPORT_TIMEOUT = int(os.getenv("GODOT_EXPORT_TIMEOUT", "300"))

# Lines Godot prints when something is actually wrong. Everything else on stderr is progress noise.
_ERROR_LINE = re.compile(r"^\s*(SCRIPT ERROR|ERROR|USER ERROR|USER SCRIPT ERROR):\s*(.+)$")
_LOCATION_LINE = re.compile(r"^\s*at:\s*(.+)$")
# Engine chatter that is reported as an error but says nothing about the game's correctness.
_IGNORABLE = (
    "Unable to open file: res://.godot",
    "editor/editor_node.cpp",
    "Cannot open file 'res://.godot/",
    "No loader found for resource: res://.godot",
)


@lru_cache(maxsize=1)
def godot_executable() -> str | None:
    """The Godot binary this machine should use, or None if there is not one.

    GODOT_EXECUTABLE wins. Otherwise the console build inside GODOT_HOME is preferred over the
    windowed one, because only the console build writes the errors we verify against to stderr.
    """
    if GODOT_EXECUTABLE:
        return GODOT_EXECUTABLE if Path(GODOT_EXECUTABLE).is_file() else None
    if GODOT_HOME.is_dir():
        candidates = sorted(GODOT_HOME.glob("Godot*.exe")) or sorted(GODOT_HOME.glob("Godot*"))
        console = [p for p in candidates if "console" in p.name.lower()]
        for path in console + candidates:
            if path.is_file():
                return str(path)
    return shutil.which("godot") or shutil.which("godot4")


def godot_available() -> bool:
    return godot_executable() is not None


@lru_cache(maxsize=1)
def godot_version() -> str:
    """The engine version string, for the manifest and the dashboard. Empty if unavailable."""
    binary = godot_executable()
    if not binary:
        return ""
    try:
        result = subprocess.run([binary, "--version"], capture_output=True, text=True, timeout=60,
                                check=False)
        return (result.stdout or result.stderr or "").strip().splitlines()[-1] if result else ""
    except (OSError, subprocess.SubprocessError, IndexError):
        return ""


def _run(args: list[str], timeout: int) -> tuple[int, str]:
    """Run the engine and hand back (exit code, stdout+stderr). Never raises."""
    binary = godot_executable()
    if not binary:
        return 127, "Godot 실행 파일을 찾을 수 없습니다."
    try:
        result = subprocess.run([binary, *args], capture_output=True, text=True, timeout=timeout,
                                check=False, encoding="utf-8", errors="replace")
        return result.returncode, f"{result.stdout or ''}\n{result.stderr or ''}"
    except subprocess.TimeoutExpired:
        return 124, f"Godot이 {timeout}초 안에 끝나지 않았습니다."
    except OSError as error:
        return 126, f"Godot 실행 실패: {type(error).__name__}: {error}"


def parse_errors(output: str, limit: int = 12) -> list[str]:
    """Pull the real problems out of an engine run.

    Godot's stderr interleaves progress chatter with errors, and an error is two lines: the message
    and an `at:` line carrying the file and line number. Joining them is what makes a finding
    something the code agent can act on instead of a sentence with no address.
    """
    findings: list[str] = []
    lines = output.splitlines()
    for index, line in enumerate(lines):
        match = _ERROR_LINE.match(line)
        if not match:
            continue
        message = match.group(2).strip()
        if any(noise in line for noise in _IGNORABLE):
            continue
        location = ""
        for following in lines[index + 1: index + 3]:
            where = _LOCATION_LINE.match(following)
            if where:
                location = where.group(1).strip()
                break
        finding = f"{match.group(1)}: {message}" + (f" ({location})" if location else "")
        if finding not in findings:
            findings.append(finding)
        if len(findings) >= limit:
            break
    return findings


@dataclass(frozen=True)
class GodotCheck:
    """The verdict of one headless verification pass."""

    ok: bool
    findings: list[str]
    output: str


def check_scripts(project_dir: Path) -> GodotCheck:
    """Compile every GDScript in the project, one file at a time.

    `--check-only` is the single place the engine returns a trustworthy exit code, so syntax is
    gated with it rather than by reading output. Per file, because one bad script otherwise hides
    every script after it.
    """
    findings: list[str] = []
    transcript: list[str] = []
    for script in sorted(project_dir.rglob("*.gd")):
        if ".godot" in script.parts:
            continue
        resource = "res://" + script.relative_to(project_dir).as_posix()
        code, output = _run(
            ["--headless", "--path", str(project_dir), "--check-only", "--script", resource],
            GODOT_IMPORT_TIMEOUT,
        )
        transcript.append(f"$ --check-only {resource} (exit {code})\n{output.strip()}")
        if code != 0:
            findings += parse_errors(output) or [f"{resource}: 스크립트 컴파일에 실패했습니다."]
    return GodotCheck(ok=not findings, findings=findings, output="\n".join(transcript))


def run_project(project_dir: Path, seconds: int = GODOT_RUN_SECONDS) -> GodotCheck:
    """Import and actually play the project headlessly, then judge what it printed.

    This is the check the HTML pipeline cannot have: a missing scene, a broken node path, a null
    dereference in _ready, an autoload that does not exist - all of them surface here with a file
    and a line, without a model being asked to guess at them from source.
    """
    code, output = _run(
        ["--headless", "--path", str(project_dir), "--quit-after", str(max(1, seconds) * 60)],
        GODOT_IMPORT_TIMEOUT,
    )
    findings = parse_errors(output)
    if code == 124:
        findings.append(f"게임이 {GODOT_IMPORT_TIMEOUT}초 안에 종료되지 않았습니다.")
    elif code not in (0, 1) and not findings:
        findings.append(f"Godot이 예상치 못한 코드로 종료했습니다 (exit {code}).")
    return GodotCheck(ok=not findings, findings=findings, output=output)


WEB_PRESET = """[preset.0]

name="Web"
platform="Web"
runnable=true
custom_features=""
export_filter="all_resources"
export_path="{export_path}"

[preset.0.options]

variant/extensions_support=false
vram_texture_compression/for_desktop=true
vram_texture_compression/for_mobile=false
html/export_icon=true
html/custom_html_shell=""
html/head_include=""
html/canvas_resize_policy=2
html/focus_canvas_on_start=true
html/experimental_virtual_keyboard=false
progressive_web_app/enabled=false
"""


# The launcher written next to a finished project. `--path` runs the game: Godot's own help says
# -e/--editor starts "the editor instead of running the scene", so its absence is what makes this a
# play button rather than an edit button. The windowed binary is preferred here for the opposite
# reason to verification - a player wants the game on screen, not its stderr.
LAUNCH_SCRIPT = "run.bat"
_LAUNCH_TEMPLATE = """@echo off
rem chcp first: this file is UTF-8 and the console defaults to the system codepage, so without it
rem every Korean message below arrives as mojibake at exactly the moment something went wrong.
chcp 65001 >nul
rem Generated by autonomous-game-studio. Runs this Godot project.
rem Set GODOT_EXECUTABLE to override which engine build is used.
setlocal
if not "%GODOT_EXECUTABLE%"=="" set "GODOT={0}%GODOT_EXECUTABLE%{0}"
if "%GODOT%"=="" set "GODOT={0}{1}{0}"
if not exist %GODOT% (
  echo Godot 실행 파일을 찾을 수 없습니다: %GODOT%
  echo GODOT_EXECUTABLE 환경 변수로 경로를 지정하세요.
  pause
  exit /b 1
)
echo %GODOT% --path "%~dp0."
%GODOT% --path "%~dp0."
if errorlevel 1 pause
endlocal
"""


def playable_binary() -> str:
    """The build a player should be launched with.

    Verification wants the console build because only it writes the errors we read to stderr.
    Launching wants the opposite: the windowed build, so the game is a game and not a terminal.
    """
    binary = godot_executable()
    if not binary:
        return ""
    windowed = Path(binary.replace("_console.exe", ".exe"))
    return str(windowed) if windowed.is_file() else binary


def write_launch_script(project_dir: Path) -> Path | None:
    """Write run.bat beside the project so it can be played without the dashboard.

    The path inside it is `%~dp0.` - the script's own directory - so the folder stays movable, and
    the engine location is overridable with GODOT_EXECUTABLE for a machine where Godot lives
    somewhere else.
    """
    binary = playable_binary()
    if not binary:
        return None
    target = project_dir / LAUNCH_SCRIPT
    target.write_text(_LAUNCH_TEMPLATE.format('"', binary), encoding="utf-8")
    return target


def export_web(project_dir: Path, target: Path) -> tuple[bool, str]:
    """Try to export a playable web build, and say plainly why not when it cannot.

    Godot only ships export templates inside a single ~1GB all-platform archive, so a machine that
    has the editor very often does not have them. That is not a defect in the generated game and
    must not read like one: the project is complete and runnable either way, it just cannot be
    embedded in the dashboard without this step.
    """
    if not godot_available():
        return False, "Godot 실행 파일이 없어 웹 빌드를 만들지 못했습니다."
    target.mkdir(parents=True, exist_ok=True)
    presets = project_dir / "export_presets.cfg"
    if not presets.is_file():
        presets.write_text(WEB_PRESET.format(export_path="build/index.html"), encoding="utf-8")
    code, output = _run(
        ["--headless", "--path", str(project_dir), "--export-release", "Web",
         str(target / "index.html")],
        GODOT_EXPORT_TIMEOUT,
    )
    if code == 0 and (target / "index.html").is_file():
        return True, f"웹 빌드를 만들었습니다: {target / 'index.html'}"
    if "export_templates" in output or "내보내기 템플릿" in output or "export templates" in output.lower():
        return False, (
            "웹 빌드를 건너뛰었습니다: Godot 익스포트 템플릿이 설치되어 있지 않습니다 "
            f"(필요: web_nothreads_debug.zip / web_nothreads_release.zip, {godot_version()}). "
            "게임 프로젝트 자체는 완성되어 Godot에서 바로 실행할 수 있습니다. 브라우저에서 "
            "플레이하려면 Godot 편집기의 Editor > Manage Export Templates에서 내려받으세요."
        )
    return False, f"웹 빌드 실패 (exit {code}): {'; '.join(parse_errors(output)) or output[-300:]}"
