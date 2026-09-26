# Bundled asset attribution

`diarization_healthcheck.wav` / `diarization_healthcheck.json`
(also muxed into `backend/tests/fixtures/conversation_clip.mp4`)

- **Purpose:** the worker's startup diarization healthcheck (M4, see
  `docs/decisions/0008-speaker-diarization.md` §5) and the diarization
  test fixture.
- **Source:** LibriSpeech ASR corpus (Panayotov et al., 2015),
  https://www.openslr.org/12, dev-clean split, via the Hugging Face
  `openslr/librispeech_asr` dataset. The three utterances used are:
  - `3000-15664-0008` (speaker 3000)
  - `3536-8226-0005` (speaker 3536)
  - `3000-15664-0011` (speaker 3000)
- **Why these speakers:** they were picked for clearly different voices,
  with estimated median pitch of ~97 Hz vs. ~241 Hz.
- **License:** CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/).
  The underlying recordings are LibriVox public-domain audiobooks.
- **Changes:** the three utterances were concatenated as an A/B/A
  "conversation" with 0.6 s of silence between turns, written as a
  16 kHz mono 16-bit WAV (37.03 s). The ground-truth turn boundaries
  and each utterance's reference transcript are in the JSON. The MP4
  test variant adds a 64×64 black 1 fps video track and encodes the
  audio as AAC 64 kbps.
