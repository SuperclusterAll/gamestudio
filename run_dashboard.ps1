$ErrorActionPreference = "Stop"
$Project = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Project
if (-not (Test-Path ".venv\Scripts\python.exe")) { python -m venv .venv }
& .venv\Scripts\python.exe -m pip install -e ".[dev]"
& .venv\Scripts\python.exe -m game_studio.server
