@echo off
setlocal EnableDelayedExpansion
rem Start the Game Studio dashboard. Double-click, or run from a terminal with an optional port:
rem     run_dashboard.bat            -> http://127.0.0.1:8765
rem     run_dashboard.bat 8766       -> http://127.0.0.1:8766
cd /d "%~dp0"

set "PORT=%~1"
if "%PORT%"=="" set "PORT=8765"
set "PY=.venv\Scripts\python.exe"

echo ==========================================
echo  Game Studio dashboard  (port %PORT%)
echo ==========================================

if not exist "%PY%" (
    echo [setup] .venv is missing. Creating it...
    python -m venv .venv || goto :fail
    "%PY%" -m pip install -e ".[dev]" || goto :fail
)

rem Refuse to start on an occupied port instead of dying with WinError 10048. Killing the parent
rem alone leaves the uvicorn child holding the socket, so /T takes the whole tree.
rem netstat prints the state after the addresses, so filter on LISTENING first and then on the
rem local port (with the trailing space, so 8765 does not also match 87651). Token 5 is the PID.
set "BUSY_PID="
for /f "tokens=5" %%P in ('netstat -ano ^| findstr "LISTENING" ^| findstr ":%PORT% "') do set "BUSY_PID=%%P"
if defined BUSY_PID (
    echo [warn] Port %PORT% is already in use by PID !BUSY_PID!.
    set /p "KILL=       Stop it and continue? [y/N] "
    if /i "!KILL!"=="y" (
        taskkill /PID !BUSY_PID! /T /F >nul 2>&1
        rem Absolute path: a shell with GNU coreutils on PATH (Git Bash) shadows Windows timeout.
        "%SystemRoot%\System32\timeout.exe" /t 1 /nobreak >nul 2>&1
        echo [ok]   Stopped PID !BUSY_PID!.
    ) else (
        echo [stop] Leaving it alone. Use a different port: run_dashboard.bat 8766
        goto :done
    )
)

echo [info] Open http://127.0.0.1:%PORT%  ^(Ctrl+C to stop^)
echo.
"%PY%" -m game_studio.server --port %PORT%
if errorlevel 1 goto :fail
goto :done

:fail
echo.
echo [FAIL] The dashboard exited with an error ^(code %errorlevel%^).
echo        Check that .env has AWS credentials and that the port is free.

:done
echo.
rem Keep the window open when double-clicked so the message above is readable.
if /i "%~2"=="--no-pause" goto :eof
pause
