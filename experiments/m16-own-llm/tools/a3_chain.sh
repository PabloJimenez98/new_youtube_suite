#!/bin/bash
# A3 GPU chain after task SFT (2026-11-18): eval the SFT model on the test split, the Qwen
# relevance reference, then DPO pairs from the SFT model. Sequential (one GPU). Each step
# resumes per episode/call file if rerun. Log: ~/yts-experiments/m16-llm/logs/a3_chain.log
set -e
M16=$(cd "$(dirname "$0")/.." && pwd); cd "$M16"
P=~/projects/new_youtube_suite/experiments/.venv/bin/python
OWN=~/projects/new_youtube_suite/backend/models/llm/yts-own-110m-sft
log() { echo "$(date '+%F %T') $*"; }
log "eval own-sft"; $P -m llm.evaluate run --model own --label own-sft --own-dir $OWN
log "eval qwen relevance reference"; $P -m llm.evaluate run --model qwen --label qwen --tasks relevance
log "gates own-sft"; $P -m llm.evaluate gates --label own-sft
log "dpo pairs"; $P -m llm.dpo_pairs build --own-dir $OWN --per-task 1000
log "chain done"
