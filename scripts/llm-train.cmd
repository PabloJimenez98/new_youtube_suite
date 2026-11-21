@echo off
rem M16 trainer from Windows (docs: experiments/m16-own-llm/TRAINING.md).
rem
rem   llm-train status              progress, speed, ETA
rem   llm-train pause               checkpoint and stop: the GPU is free in seconds
rem   llm-train resume              continue the latest run exactly where it stopped
rem   llm-train start configs\stage_a.yaml
rem
rem Double-clicking it (no arguments) shows the status. The double-click shortcuts
rem llm-pause.cmd, llm-resume.cmd and llm-status.cmd sit next to it.
rem
rem It runs experiments/m16-own-llm from THIS checkout inside WSL, with the WSL
rem venv. Override with env vars, e.g.
rem   set YTS_WSL_DISTRO=Ubuntu-24.04
rem   set YTS_LLM_PYTHON=~/projects/new_youtube_suite/experiments/.venv/bin/python

setlocal
if not defined YTS_WSL_DISTRO set YTS_WSL_DISTRO=Ubuntu-24.04
if not defined YTS_LLM_PYTHON set YTS_LLM_PYTHON=~/projects/new_youtube_suite/experiments/.venv/bin/python

for /f "usebackq delims=" %%p in (`wsl.exe -d %YTS_WSL_DISTRO% -e wslpath -a "%~dp0..\experiments\m16-own-llm"`) do set "M16_DIR=%%p"

set "ARGS=%*"
if "%ARGS%"=="" set "ARGS=status"
set "ARGS=%ARGS:\=/%"

wsl.exe -d %YTS_WSL_DISTRO% -e bash -lc "cd '%M16_DIR%' && %YTS_LLM_PYTHON% -m llm.train %ARGS%"
set RC=%ERRORLEVEL%

if "%~1"=="" pause
if defined LLM_TRAIN_PAUSE_AT_END pause
exit /b %RC%
