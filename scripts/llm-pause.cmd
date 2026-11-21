@echo off
rem Double-click: pause M16 training (checkpoint, then exit; the GPU is freed).
set LLM_TRAIN_PAUSE_AT_END=1
call "%~dp0llm-train.cmd" pause
