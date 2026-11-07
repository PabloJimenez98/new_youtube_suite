#!/usr/bin/env bash
# Rebuild the gitignored test media from LibriSpeech dev-clean (CC BY 4.0),
# exactly as documented in backend/app/assets/ATTRIBUTION.md and
# backend/tests/fixtures/ATTRIBUTION.md:
#   backend/app/assets/diarization_healthcheck.wav  (A/B/A, 37.03 s)
#   backend/tests/fixtures/conversation_clip.mp4    (same audio, muxed)
#   backend/tests/fixtures/librispeech_clip.mp4     (1272-128104-0009, 18.3 s)
#
# Usage: scripts/build_test_fixtures.sh [path/to/LibriSpeech/dev-clean]
# Without an argument it downloads dev-clean (~340 MB) to a temp dir.
# Needs ffmpeg (libx264).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BACKEND="$ROOT/backend"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

SRC="${1:-}"
if [[ -z "$SRC" ]]; then
  curl -fSL -o "$WORK/dev-clean.tar.gz" https://www.openslr.org/resources/12/dev-clean.tar.gz
  tar -xzf "$WORK/dev-clean.tar.gz" -C "$WORK" \
    LibriSpeech/dev-clean/1272/128104 LibriSpeech/dev-clean/3000/15664 LibriSpeech/dev-clean/3536/8226
  SRC="$WORK/LibriSpeech/dev-clean"
fi

dur() { ffprobe -v error -show_entries format=duration -of csv=p=0 "$1"; }
to_wav() { ffmpeg -v error -y -i "$1" -ac 1 -ar 16000 -sample_fmt s16 "$2"; }

to_wav "$SRC/3000/15664/3000-15664-0008.flac" "$WORK/a1.wav"
to_wav "$SRC/3536/8226/3536-8226-0005.flac"   "$WORK/b.wav"
to_wav "$SRC/3000/15664/3000-15664-0011.flac" "$WORK/a2.wav"
ffmpeg -v error -y -f lavfi -i anullsrc=r=16000:cl=mono -t 0.6 -sample_fmt s16 "$WORK/sil.wav"
for f in a1 b a2; do echo "$f $(dur $WORK/$f.wav)"; done
printf "file '%s'\n" "$WORK/a1.wav" "$WORK/sil.wav" "$WORK/b.wav" "$WORK/sil.wav" "$WORK/a2.wav" > "$WORK/list.txt"
ffmpeg -v error -y -f concat -safe 0 -i "$WORK/list.txt" -c pcm_s16le -ar 16000 -ac 1 "$BACKEND/app/assets/diarization_healthcheck.wav"
echo "healthcheck.wav $(dur $BACKEND/app/assets/diarization_healthcheck.wav)"

mux() {  # audio in -> MP4 with a 64x64 black 1 fps H.264 track + AAC 64 kbps
  ffmpeg -v error -y -f lavfi -i color=c=black:s=64x64:r=1 -i "$1" \
    -map 0:v -map 1:a -c:v libx264 -pix_fmt yuv420p -c:a aac -b:a 64k -shortest "$2"
}
mux "$BACKEND/app/assets/diarization_healthcheck.wav" "$BACKEND/tests/fixtures/conversation_clip.mp4"
mux "$SRC/1272/128104/1272-128104-0009.flac" "$BACKEND/tests/fixtures/librispeech_clip.mp4"
for f in tests/fixtures/conversation_clip.mp4 tests/fixtures/librispeech_clip.mp4; do echo "$f $(dur $BACKEND/$f)"; done
