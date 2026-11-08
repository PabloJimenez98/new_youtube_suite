@echo off
rem Launch the whole YouTube Suite dev stack inside WSL from Windows.
rem Double-click this file (or run it from cmd/PowerShell). It runs
rem scripts/dev.sh in the WSL clone, which holds the venv, node_modules
rem and the CUDA build of llama-cpp-python. Ctrl+C stops the app.
rem
rem Override the distro or the clone path with env vars, e.g.
rem   set YTS_WSL_DISTRO=Ubuntu-24.04
rem   set YTS_WSL_REPO=~/projects/new_youtube_suite

if not defined YTS_WSL_DISTRO set YTS_WSL_DISTRO=Ubuntu-24.04
if not defined YTS_WSL_REPO set YTS_WSL_REPO=~/projects/new_youtube_suite

title YouTube Suite (WSL)
echo Starting YouTube Suite in WSL (%YTS_WSL_DISTRO%: %YTS_WSL_REPO%)
echo App: http://localhost:4200   API: http://127.0.0.1:8000
echo.

wsl.exe -d %YTS_WSL_DISTRO% -e bash -lc "cd %YTS_WSL_REPO% && exec scripts/dev.sh"

echo.
echo YouTube Suite stopped (exit code %ERRORLEVEL%).
pause
