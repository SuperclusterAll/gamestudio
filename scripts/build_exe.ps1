$ErrorActionPreference = "Stop"
$Project = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Project

& .venv\Scripts\python.exe -m pip install -e ".[packaging]"
& .venv\Scripts\pyinstaller.exe --noconfirm --clean --onefile --name game-studio `
  --paths src src\game_studio\cli.py
& .venv\Scripts\pyinstaller.exe --noconfirm --onefile --name game-studio-dashboard `
  --paths src --add-data "web;web" src\game_studio\server.py

Write-Host "Created $Project\dist\game-studio.exe and game-studio-dashboard.exe"
