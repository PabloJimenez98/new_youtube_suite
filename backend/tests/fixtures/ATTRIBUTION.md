# Test fixture attribution

`librispeech_clip.mp4` / `librispeech_clip.txt`

- **Source:** LibriSpeech ASR corpus (Panayotov et al., 2015),
  https://www.openslr.org/12. Utterance `1272-128104-0009`
  (dev-clean, speaker 1272, chapter 128104), 18.3 s. Obtained through
  the `hf-internal-testing/librispeech_asr_dummy` subset on Hugging Face.
- **License:** CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/).
  The underlying recordings are LibriVox public-domain audiobooks.
- **Changes:** the FLAC audio was muxed into an MP4 (AAC 64 kbps) with a
  64×64 black 1 fps H.264 video track, so the transcription pipeline
  sees a video upload. `librispeech_clip.txt` is the corpus's
  reference transcript, unchanged.
