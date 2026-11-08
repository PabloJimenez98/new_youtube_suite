/**
 * Data-science glossary shown in tooltips (`<app-term>`) and on the
 * How it works page. Definitions describe what the term means *in this
 * platform*, not in general (docs/decisions/0021).
 */
export type GlossaryKey =
  | 'asr'
  | 'vad'
  | 'word-timestamps'
  | 'int8'
  | 'wer'
  | 'diarization'
  | 'speaker-embedding'
  | 'sentence-embedding'
  | 'bilstm'
  | 'texttiling'
  | 'f1-at-5s'
  | 'tfidf'
  | 'language-id'
  | 'llm'
  | 'zero-shot'
  | 'gguf'
  | 'json-grammar'
  | 'map-reduce'
  | 'sliding-window'
  | 'heuristic'
  | 'feature-weight'
  | 'relative-score'
  | 'pseudo-label'
  | 'peak-detection'
  | 'face-detection'
  | 'text-detection'
  | 'shot-detection'
  | 'ema'
  | 'spacy'
  | 'snapshot'
  | 'ctr'
  | 'retention'
  | 'model-version';

export interface GlossaryEntry {
  term: string;
  definition: string;
}

export const GLOSSARY: Record<GlossaryKey, GlossaryEntry> = {
  asr: {
    term: 'ASR (speech recognition)',
    definition:
      'Automatic speech recognition: turning an audio signal into text. Here it is done by Whisper, a pretrained encoder-decoder model.',
  },
  vad: {
    term: 'VAD (voice activity detection)',
    definition:
      'A small model that finds where someone is speaking. Silence is skipped and the speech is cut into chunks that Whisper can decode in parallel.',
  },
  'word-timestamps': {
    term: 'Word timestamps',
    definition:
      'A start and end time for every recognised word. They are the time axis that every later tool (speakers, chapters, relevance, shorts, hooks) is aligned to.',
  },
  int8: {
    term: 'int8 quantization',
    definition:
      'Storing model weights as 8-bit integers instead of 16/32-bit floats. It cuts memory about 2–4× and speeds up inference, at a small cost in accuracy.',
  },
  wer: {
    term: 'WER (word error rate)',
    definition:
      'The share of words a transcript gets wrong (substituted, missing or extra) against a reference. Lower is better.',
  },
  diarization: {
    term: 'Speaker diarization',
    definition:
      '"Who spoke when": splitting the audio into turns and grouping the turns that belong to the same voice, without knowing who the speakers are.',
  },
  'speaker-embedding': {
    term: 'Speaker embedding',
    definition:
      'A vector that summarises how a voice sounds. Turns whose vectors are close are clustered as the same speaker.',
  },
  'sentence-embedding': {
    term: 'Sentence embedding',
    definition:
      'A fixed-size vector that captures the meaning of a piece of text. Here a frozen multilingual encoder (multilingual-e5-small) produces one per transcript unit.',
  },
  bilstm: {
    term: 'BiLSTM',
    definition:
      'A bidirectional recurrent neural network: it reads a sequence forwards and backwards, so each position sees its past and future context. Used here to tag "a chapter starts here".',
  },
  texttiling: {
    term: 'TextTiling',
    definition:
      'An unsupervised topic-segmentation method: a boundary is placed where adjacent windows of text stop being similar. It is the baseline the trained BiLSTM must beat.',
  },
  'f1-at-5s': {
    term: 'F1@5s',
    definition:
      'Boundary F1 score where a predicted chapter start counts as correct if it is within 5 seconds of a real one. Balances precision and recall.',
  },
  tfidf: {
    term: 'TF-IDF',
    definition:
      'Term frequency × inverse document frequency: a word scores high when it is frequent in this episode but rare across its windows. Used to find the episode’s topic keywords.',
  },
  'language-id': {
    term: 'Language identification',
    definition:
      'Detecting which language a text is in. A fastText classifier picks the episode language so prompts and word lists match it.',
  },
  llm: {
    term: 'LLM (large language model)',
    definition:
      'A pretrained generative text model. The platform runs one locally (Qwen3.5-9B) for chapter titles, show notes and relevance scoring.',
  },
  'zero-shot': {
    term: 'Zero-shot',
    definition:
      'Using a pretrained model for a task it was never trained on, by describing the task in the prompt, with no labelled examples.',
  },
  gguf: {
    term: 'GGUF (Q4_K_M)',
    definition:
      'The llama.cpp model format. Q4_K_M is a 4-bit quantization that lets a 9-billion-parameter model fit in about 5.5 GB of GPU memory.',
  },
  'json-grammar': {
    term: 'Grammar-constrained output',
    definition:
      'The LLM may only emit tokens that keep its output valid against a JSON schema, so the result can always be parsed and validated.',
  },
  'map-reduce': {
    term: 'Map-reduce summarisation',
    definition:
      'Long text is split into chunks, each chunk is summarised (map), and the summaries are combined into one result (reduce). It works around the model’s context limit.',
  },
  'sliding-window': {
    term: 'Sliding window',
    definition:
      'Fixed-length windows moved along the timeline by a step (hop). With 15 s windows and a 7.5 s hop, consecutive windows overlap by 50 %.',
  },
  heuristic: {
    term: 'Heuristic (v0)',
    definition:
      'A hand-designed scoring rule built from features and fixed weights, instead of a model learned from data. It is a transparent baseline until there is enough real data to train.',
  },
  'feature-weight': {
    term: 'Feature weight',
    definition:
      'How much a feature contributes to the final score. A weight of 0 means the feature is measured and stored but not used yet: there is no evidence for its direction.',
  },
  'relative-score': {
    term: 'Relative score',
    definition:
      'A 0–100 score that only ranks the items of one episode against each other. It is not a probability or a percentage, and it can’t be compared across episodes.',
  },
  'pseudo-label': {
    term: 'Pseudo-label',
    definition:
      'A training target produced by a model or heuristic instead of a human. The v0 relevance scores are the pseudo-labels the trained M15 model will learn from.',
  },
  'peak-detection': {
    term: 'Peak detection',
    definition:
      'Finding local maxima in a curve. Shorts are cut around the peaks of the smoothed relevance curve (prominence ≥ 0.05, peaks ≥ 60 s apart).',
  },
  'face-detection': {
    term: 'Face detection (YuNet)',
    definition:
      'A small CNN that returns face boxes with a confidence. Used to keep the speaker in frame for shorts and to measure faces on thumbnails.',
  },
  'text-detection': {
    term: 'Text detection (DB)',
    definition:
      'A segmentation network that finds regions of text in an image (not what the text says). Used to measure how much of a thumbnail is text.',
  },
  'shot-detection': {
    term: 'Shot detection',
    definition:
      'Finding camera cuts: a large colour-histogram distance between consecutive frames (Bhattacharyya > 0.25) marks a new shot.',
  },
  ema: {
    term: 'EMA smoothing',
    definition:
      'An exponential moving average: each new value is blended with the previous one, so a moving crop follows the face without jitter.',
  },
  spacy: {
    term: 'spaCy',
    definition:
      'An NLP library that splits sentences and tags each word (part of speech, person, mood, entities). Used for the linguistic hook features in English and Spanish.',
  },
  snapshot: {
    term: 'Snapshot',
    definition:
      'A dated copy of a metric (views, likes, CTR…) for one video. Repeated snapshots form a time series of how a video performs.',
  },
  ctr: {
    term: 'CTR (click-through rate)',
    definition:
      'Clicks ÷ thumbnail impressions. The real CTR comes from YouTube’s Reporting API and will be the label for a trained CTR model.',
  },
  retention: {
    term: 'Audience retention',
    definition:
      'The share of viewers still watching at each point of a video, from the Analytics API. A future label for the hook model.',
  },
  'model-version': {
    term: 'Model version',
    definition:
      'Every stored result records which model or formula produced it (e.g. relevance-p2, heuristic-v0), so a score can always be traced and compared after a retrain.',
  },
};
