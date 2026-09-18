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

from .required_art import unused_sprites

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
    presets = project_dir / "export_presets.cfg"
    try:
        target.mkdir(parents=True, exist_ok=True)
        if not presets.is_file():
            presets.write_text(WEB_PRESET.format(export_path="build/index.html"), encoding="utf-8")
        code, output = _run(
            ["--headless", "--path", str(project_dir), "--export-release", "Web",
             str(target / "index.html")],
            GODOT_EXPORT_TIMEOUT,
        )
    except OSError as error:
        # Packaging is the last thing a run does and the project is already on disk, so nothing
        # here is worth ending a finished build over - a full disk or a locked file becomes "no web
        # build" like any other reason.
        _clear_failed_build(target)
        return False, f"웹 빌드를 만들지 못했습니다: {error}"
    if code == 0 and (target / "index.html").is_file():
        return True, f"웹 빌드를 만들었습니다: {target / 'index.html'}"
    # Nothing usable was produced, so leave nothing behind. An empty build/ is worse than no
    # build/: it is what the dashboard and the restore path look in to decide whether this run has
    # something playable in the browser, and a half-written one is a broken page rather than an
    # honest absence.
    _clear_failed_build(target)
    if "export_templates" in output or "내보내기 템플릿" in output or "export templates" in output.lower():
        return False, (
            "웹 빌드를 건너뛰었습니다: Godot 익스포트 템플릿이 설치되어 있지 않습니다 "
            f"(필요: web_nothreads_debug.zip / web_nothreads_release.zip, {godot_version()}). "
            "게임 프로젝트 자체는 완성되어 Godot에서 바로 실행할 수 있습니다. 브라우저에서 "
            "플레이하려면 Godot 편집기의 Editor > Manage Export Templates에서 내려받으세요."
        )
    return False, f"웹 빌드 실패 (exit {code}): {'; '.join(parse_errors(output)) or output[-300:]}"


def _clear_failed_build(target: Path) -> None:
    """Remove a build directory that has no playable page in it."""
    if not target.is_dir() or (target / "index.html").is_file():
        return
    try:
        shutil.rmtree(target)
    except OSError:
        return


# What a Godot build is checked for beyond "it starts without errors".
#
# The engine catches everything that throws, which is far more than the HTML path can see - but a
# project that throws nothing is not the same as a game. A scene whose script is just `func _ready():
# pass` imports cleanly, runs for its five seconds, exits zero and reported PASS: no input, no
# scoring, no win or loss, nothing to play. The HTML path has static_qa for exactly this class of
# problem and Godot had no equivalent, so this is it.
#
# Blocking is reserved for what genuinely stops the game being a game. Anything a reviewer might
# reasonably disagree about is advisory, in the same spirit as static_qa: a keyword check that
# failed a working game used to cost a whole repair cycle.
_FRAME_LOOP = re.compile(r"func\s+_(process|physics_process)\s*\(")
_INPUT_USE = re.compile(
    r"Input\.(is_action|is_key|is_mouse|get_vector|get_axis)|"
    r"func\s+_(input|unhandled_input|unhandled_key_input|gui_input)\s*\(|"
    r"InputEvent"
)
# Godot's own networking surface. A generated game must be standalone, exactly as on the web path.
_NETWORK_USE = re.compile(r"HTTPRequest|HTTPClient|WebSocket|ENetMultiplayerPeer|\bhttps?://")
# Every res:// path mentioned anywhere in the project. A reference to a file nobody wrote fails at
# runtime, and only the paths on the code path the headless run happens to reach would ever show up
# in its output - a scene loaded from a branch that needs input is never touched in five seconds.
_RESOURCE_REF = re.compile(r"res://([^\"'\s\)\]]+)")
# Arrow keys and WASD, as Godot spells them: physical keycodes in project.godot, named constants in
# GDScript, and the built-in ui_* actions which map to the arrows only.
_ARROW_EVIDENCE = re.compile(r"KEY_(LEFT|RIGHT|UP|DOWN)\b|41943(19|20|21|22)\b|\bui_(left|right|up|down)\b")
_WASD_EVIDENCE = re.compile(r"KEY_[WASD]\b|\"physical_keycode\":(87|65|83|68)\b")
_SCORE_HINT = re.compile(r"score|점수|combo|rank|distance|lap|time_left", re.IGNORECASE)
_RESTART_HINT = re.compile(r"restart|reload_current_scene|재시작|다시\s*시작|change_scene", re.IGNORECASE)

_SOURCE_SUFFIXES = (".gd", ".tscn", ".tres", ".godot")
# Type names from the OTHER languages this agent writes, mapped to what GDScript actually calls
# them. The same code agent writes JavaScript for the HTML5 path and reads Python all day, and
# GDScript's vocabulary overlaps just enough to be dangerous: `int`, `float` and `bool` are real,
# `str`, `list` and `dict` are not.
#
# This is not a style note. An unresolvable type is a PARSE error, which means the script does not
# load AT ALL - Godot brings the scene up and nothing in it runs: no _ready, no _process, no input,
# no drawing. The window opens black and the game looks like it was never written.
#
# Measured on a delivered project: `dict` on line 377 of main.gd. Both of that run's rethink cycles
# went on those two error lines, and it shipped nothing - over one word.
#
# Checked here rather than left to the engine because it costs no model call and no engine start,
# and because naming the replacement is the part the engine's own message leaves out: "Could not
# find type 'dict' in the current scope" says what is wrong and never says the answer is
# "Dictionary".
_FOREIGN_TYPES = {
    "dict": "Dictionary",
    "list": "Array",
    "tuple": "Array",
    "set": "Dictionary",
    "str": "String",
    "string": "String",
    "boolean": "bool",
    "number": "float",
    "double": "float",
    "any": "Variant",
    "object": "Object",
    "function": "Callable",
    "undefined": "Variant",
}
# Where a type name is allowed to stand: after an annotation colon or a return arrow, inside an
# Array[...] element type, or after `as` / `is`.
#
# The horizontal-whitespace class is load-bearing. Written with a plain \s, the colon that ends
# `func _ready():` would swallow the newline and capture the first word of the NEXT line - so a
# variable innocently named `list` on the line after any `if cond:` would be reported as a type.
_TYPE_POSITION = re.compile(
    r"(?::|->)[^\S\n]*([A-Za-z_][A-Za-z0-9_]*)"
    # A lookbehind, not a match: in `rows: Array[str]` the annotation colon has already
    # consumed "Array" by the time the scan reaches the bracket, so asking for it again finds
    # nothing.
    r"|(?<=Array)\[[^\S\n]*([A-Za-z_][A-Za-z0-9_]*)"
    r"|(?<![A-Za-z0-9_])(?:as|is)[^\S\n]+([A-Za-z_][A-Za-z0-9_]*)"
)
# Anything quoted, and anything after a #. A Label reading "Score: str" is not an annotation.
_GD_QUOTED = re.compile(r'"[^"\n]*"|\'[^\'\n]*\'|"[^\n]*$|\'[^\n]*$')
_GD_COMMENT = re.compile(r"#[^\n]*")


def _code_only(line: str) -> str:
    """The code on one line, with string literals and comments blanked out.

    Blanked to spaces rather than removed so that what is left keeps its original shape - the
    trailing alternatives cover a quote that opens a multi-line string and never closes on this
    line, which would otherwise leave its whole body being read as code.
    """
    return _GD_COMMENT.sub("", _GD_QUOTED.sub(lambda m: " " * len(m.group(0)), line))


def foreign_types(files: dict[Path, str]) -> list[str]:
    """Type annotations GDScript cannot resolve, one finding per occurrence, each with its fix."""
    findings: list[str] = []
    for path, text in sorted(files.items()):
        if path.suffix.lower() != ".gd":
            continue
        for number, line in enumerate(text.splitlines(), 1):
            for match in _TYPE_POSITION.finditer(_code_only(line)):
                name = next(group for group in match.groups() if group)
                if correct := _FOREIGN_TYPES.get(name):
                    findings.append(
                        f"{path.name}:{number} — GDScript에 '{name}' 타입은 없습니다. "
                        f"'{correct}'로 고치세요. 타입을 찾지 못하면 스크립트 전체가 파싱에 "
                        "실패해서, 장면은 떠도 아무것도 실행되지 않고 화면이 빈 채로 열립니다."
                    )
    return findings



def _project_text(project_dir: Path) -> tuple[str, dict[Path, str]]:
    """Every authored file in the project, concatenated and also keyed by path."""
    files: dict[Path, str] = {}
    for path in sorted(project_dir.rglob("*")):
        if (path.is_file() and path.suffix.lower() in _SOURCE_SUFFIXES
                and ".godot" not in path.parts and "build" not in path.parts):
            try:
                files[path] = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
    return "\n".join(files.values()), files


def _main_scene(project_config: str) -> str:
    match = re.search(r"run/main_scene\s*=\s*\"([^\"]+)\"", project_config)
    return match.group(1) if match else ""


_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".svg"}


def _missing_resource(project_dir: Path, reference: str) -> str:
    """Say what to do about it, not only that it is missing.

    Every one of these is a single file away from running, and the two shapes they come in are both
    recognisable from the path alone. A .tscn whose .gd sibling is sitting right there is a script
    the agent wrote and never packaged into a scene; an image path is a sprite it planned and never
    generated. Reported as a bare missing path, a repair cycle went on working out which of the two
    it was - one run ended with four of these open and its budget spent.
    """
    path = Path(reference)
    suffix = path.suffix.lower()
    if suffix == ".tscn" and (project_dir / path.with_suffix(".gd")).is_file():
        return (f"존재하지 않는 리소스를 참조합니다: res://{reference} — "
                f"{path.with_suffix('.gd').name}는 있는데 장면 파일이 없습니다. "
                f"그 스크립트를 붙인 {path.name}을 write_godot_file로 만드세요.")
    if suffix in _IMAGE_SUFFIXES:
        return (f"존재하지 않는 리소스를 참조합니다: res://{reference} — "
                f'generate_comfyui_image(asset_name="{path.stem}", ...)로 생성하거나, '
                "이미 있는 파일 이름으로 참조를 고치세요.")
    return f"존재하지 않는 리소스를 참조합니다: res://{reference}"


def static_project_qa(project_dir: Path, sprites: list[str] | None = None) -> GodotCheck:
    """Check the project's shape, without starting the engine.

    Runs before compiling and before playing because it is free, and because its findings are the
    ones an engine run cannot produce: the engine has nothing to report about a game that simply
    never reads input.
    """
    config_path = project_dir / "project.godot"
    if not config_path.is_file():
        return GodotCheck(ok=False, findings=["project.godot이 없습니다."], output="")
    config = config_path.read_text(encoding="utf-8", errors="replace")
    body, files = _project_text(project_dir)
    scripts = "\n".join(text for path, text in files.items() if path.suffix.lower() == ".gd")
    findings: list[str] = []

    scene = _main_scene(config)
    if not scene:
        findings.append("project.godot에 run/main_scene이 지정되지 않아 실행할 장면이 없습니다.")
    elif not (project_dir / scene.removeprefix("res://")).is_file():
        findings.append(f"메인 장면 파일이 없습니다: {scene}")

    if not _FRAME_LOOP.search(scripts):
        findings.append("_process 또는 _physics_process가 어디에도 없습니다. 게임이 매 프레임 갱신되지 않습니다.")
    if not _INPUT_USE.search(scripts):
        findings.append("입력 처리가 없습니다 (Input.* 또는 _input/_unhandled_input). 플레이할 수 없습니다.")
    # Reported next to the checks that ask whether the scripts DO anything, because an
    # unresolvable type means they never get the chance: one bad name and the file does not
    # load. Capped so a project that got the convention wrong everywhere still returns a
    # readable report rather than three hundred lines of the same sentence.
    findings += foreign_types(files)[:8]
    for network in sorted(set(_NETWORK_USE.findall(body))):
        findings.append(f"외부 네트워크 의존성은 허용되지 않습니다: {network}")

    # A res:// path that resolves to nothing fails the moment that code path is reached, which may
    # be long after a five-second headless run has ended.
    #
    # Only paths the project actually spells out. A game that assembles one - "res://assets/block-"
    # + kind + ".png", or a "%s" template - leaves a fragment in the source that names no file and
    # never will, and reporting it as a missing resource fails a build whose art is perfectly fine.
    missing = sorted({
        reference for reference in _RESOURCE_REF.findall(body)
        if Path(reference).suffix and not any(mark in reference for mark in "%{}$*")
        and not (project_dir / reference).exists() and not reference.startswith(".godot")
    })
    findings += [_missing_resource(project_dir, reference) for reference in missing[:6]]


    # Generated art nobody draws is paid-for work thrown away, and worth saying so - but it is not
    # a reason to fail a release. The game runs. Blocking on it spent the entire rethink budget on
    # "you did not use art you paid for" and then shipped with the finding open anyway: the release
    # failed AND the waste was not fixed. It is a fact about files on disk rather than a keyword
    # guess, so unlike the notes below it is reported whether or not anything else failed.
    advisories: list[str] = []
    if unused := unused_sprites(body, sprites or []):
        advisories.append(
            f"참고: 생성된 스프라이트를 프로젝트가 참조하지 않습니다: {', '.join(unused)}. "
            "Sprite2D의 texture로 res://assets/<name>.png를 불러오면 낭비를 줄일 수 있습니다."
        )
    notes: list[str] = []
    if not _WASD_EVIDENCE.search(body):
        # Advisory, not blocking. On the web this is a real defect - event.key is layout-dependent
        # and W arrives as 'ㅈ' under a Korean IME - but Godot's physical_keycode is layout
        # independent, so arrows-only here is a design choice rather than a broken control.
        notes.append("참고: WASD 입력이 보이지 않습니다. 방향키와 함께 지원하는 것이 좋습니다.")
    elif not _ARROW_EVIDENCE.search(body):
        notes.append("참고: 방향키 입력이 보이지 않습니다.")
    if not _SCORE_HINT.search(body):
        notes.append("참고: 점수/기록 표시가 보이지 않습니다.")
    if not _RESTART_HINT.search(body):
        notes.append("참고: 재시작 경로가 보이지 않습니다.")

    # ok reflects only what blocks. The notes travel with it either way: "you generated art the
    # project never uses" is a fact about files on disk, not a keyword guess, and hiding it unless
    # something else already failed buries it in exactly the run where it is worth acting on.
    # The keyword guesses ride along only next to a real failure - on their own they are as often
    # wrong as right, and a keyword check that failed a working game used to cost a repair cycle.
    return GodotCheck(ok=not findings, findings=findings + advisories + (notes if findings else []),
                      output="\n".join(f"{path.name}: {len(text.splitlines())} lines"
                                       for path, text in files.items()))
