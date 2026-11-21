@echo off
rem Double-click: resume the latest M16 training run where it stopped.
set LLM_TRAIN_PAUSE_AT_END=1
call "%~dp0llm-train.cmd" resume
