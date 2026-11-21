@echo off
rem Double-click: show the M16 training status.
set LLM_TRAIN_PAUSE_AT_END=1
call "%~dp0llm-train.cmd" status
