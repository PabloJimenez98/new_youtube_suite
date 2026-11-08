/**
 * Explanations of every data-science process on the platform: one entry
 * per tool plus the platform-level nodes of the How it works diagram.
 * Single source of truth for the info popups, the pipeline stepper and
 * the How it works page (docs/decisions/0021). The content is taken
 * from the decision records cited in each `source`.
 */
import { ToolType } from '../jobs/job.model';
import { Flow } from './flow';
import { GlossaryKey } from './glossary';

export type PlatformNodeKey = 'upload' | 'market' | 'analytics' | 'reporting' | 'm15';
export type DocKey = ToolType | PlatformNodeKey;

export type ModelKind = 'pretrained' | 'trained' | 'heuristic v0' | 'LLM zero-shot' | 'hybrid' | 'data pipeline' | 'planned';

export interface DocStep {
  title: string;
  detail: string;
  terms?: GlossaryKey[];
}

export interface ToolDoc {
  key: DocKey;
  title: string;
  oneLiner: string;
  inputs: string[];
  outputs: string[];
  steps: DocStep[];
  /** Inference flow: what happens when the tool runs on one episode. */
  flow: Flow;
  /** Custom models only: how the model was (or will be) trained. */
  training?: { summary: string; flow: Flow; planned: boolean };
  model: { name: string; kind: ModelKind; runtime: string };
  metric?: { name: string; means: string; doesNotMean: string };
  limitations: string[];
  dependsOn: ToolType[];
  source: string;
}

const transcribe: ToolDoc = {
  key: 'transcribe',
  title: 'Transcript',
  oneLiner:
    'Turns the episode’s speech into text with a start and end time for every word: the time axis every other tool builds on.',
  inputs: ['The uploaded episode video'],
  outputs: ['transcript_segments (sentences with times)', 'transcript_words (words with times and confidence)', '.srt / .vtt subtitles'],
  steps: [
    {
      title: 'Extract the audio',
      detail:
        'ffmpeg drops the video stream and converts the audio to mono, 16 kHz, 16-bit PCM WAV: the format the speech models expect. The WAV is kept so diarization can reuse it.',
    },
    {
      title: 'Find the speech',
      detail: 'Voice activity detection skips silence and cuts the speech into chunks.',
      terms: ['vad'],
    },
    {
      title: 'Decode in batches',
      detail:
        'Whisper large-v3 decodes 16 chunks at a time on the GPU, with greedy decoding and int8 weights. About 48× faster than real time on the RTX 5070.',
      terms: ['asr', 'int8'],
    },
    {
      title: 'Align words',
      detail:
        'Each segment gets its words with start/end times and a probability, plus the segment’s average log-probability and no-speech probability as quality signals.',
      terms: ['word-timestamps'],
    },
    {
      title: 'Store and export',
      detail:
        'Segments and words are written to two tables. Subtitles are built from those rows on demand, with speaker names once diarization has run.',
    },
  ],
  flow: {
    nodes: [
      { id: 'video', label: 'Episode video', kind: 'data' },
      { id: 'ffmpeg', label: 'Audio extraction', detail: 'ffmpeg · mono 16 kHz', kind: 'process' },
      { id: 'wav', label: 'WAV audio', detail: 'kept for diarization', kind: 'data' },
      { id: 'vad', label: 'Voice activity detection', detail: 'skip silence, chunk', kind: 'model' },
      { id: 'whisper', label: 'Whisper large-v3', detail: 'batch 16 · greedy · int8', kind: 'model' },
      { id: 'rows', label: 'Segments + words', detail: 'with timestamps', kind: 'output' },
      { id: 'subs', label: '.srt / .vtt', detail: 'built on demand', kind: 'output' },
    ],
    edges: [
      { from: 'video', to: 'ffmpeg' },
      { from: 'ffmpeg', to: 'wav' },
      { from: 'wav', to: 'vad' },
      { from: 'vad', to: 'whisper', label: 'speech chunks' },
      { from: 'whisper', to: 'rows' },
      { from: 'rows', to: 'subs' },
    ],
  },
  model: { name: 'Whisper large-v3 (faster-whisper)', kind: 'pretrained', runtime: 'GPU, int8_float16' },
  limitations: [
    'Overlapping speech and laughter lower accuracy.',
    'Some words come back without timestamps; later tools attach them to the previous timed word.',
    'Re-running the transcript replaces the rows, so speaker labels are lost until Speakers runs again.',
  ],
  dependsOn: [],
  source: 'Decisions 0007, 0014, 0016',
};

const diarize: ToolDoc = {
  key: 'diarize',
  title: 'Speakers',
  oneLiner: 'Works out who is speaking when, and labels every word and sentence of the transcript with a speaker.',
  inputs: ['WAV audio (from Transcript)', 'Transcript words and segments'],
  outputs: ['speaker_label on every word and segment', 'Speaker names in the subtitles'],
  steps: [
    {
      title: 'Segment the audio into turns',
      detail: 'pyannote’s segmentation model finds speech turns, including where voices overlap.',
      terms: ['diarization'],
    },
    {
      title: 'Embed each voice',
      detail: 'Each turn becomes a speaker embedding: a vector describing how the voice sounds.',
      terms: ['speaker-embedding'],
    },
    {
      title: 'Cluster the voices',
      detail:
        'Turns with similar embeddings are grouped as one speaker. The overlap-free ("exclusive") output keeps at most one speaker at any instant.',
    },
    {
      title: 'Assign speakers to words',
      detail:
        'Each word takes the turn it overlaps most in time; a word in a gap takes the nearest turn. Each sentence takes the majority speaker of its words.',
      terms: ['word-timestamps'],
    },
    {
      title: 'Normalise and store',
      detail: 'Labels are renumbered by first appearance (the first voice is SPEAKER_00) and written in one transaction.',
    },
  ],
  flow: {
    nodes: [
      { id: 'wav', label: 'WAV audio', kind: 'data' },
      { id: 'words', label: 'Transcript words', kind: 'data' },
      { id: 'seg', label: 'Turn segmentation', detail: 'pyannote', kind: 'model' },
      { id: 'emb', label: 'Speaker embeddings', detail: 'one vector per turn', kind: 'model' },
      { id: 'clu', label: 'Clustering', detail: 'similar voices → one speaker', kind: 'process' },
      { id: 'turns', label: 'Speaker turns', detail: 'overlap-free', kind: 'data' },
      { id: 'assign', label: 'Assign by overlap', detail: 'max overlap · else nearest', kind: 'decision' },
      { id: 'out', label: 'speaker_label', detail: 'words + segments', kind: 'output' },
    ],
    edges: [
      { from: 'wav', to: 'seg' },
      { from: 'seg', to: 'emb' },
      { from: 'emb', to: 'clu' },
      { from: 'clu', to: 'turns' },
      { from: 'turns', to: 'assign' },
      { from: 'words', to: 'assign' },
      { from: 'assign', to: 'out' },
    ],
  },
  model: { name: 'pyannote speaker-diarization-community-1', kind: 'pretrained', runtime: 'GPU' },
  limitations: [
    'The number of speakers is estimated, not given: a guest who barely speaks may merge with a host.',
    'Speakers are anonymous (SPEAKER_00, SPEAKER_01…), not named.',
  ],
  dependsOn: ['transcribe'],
  source: 'Decision 0008',
};

const chapters: ToolDoc = {
  key: 'chapters',
  title: 'Chapters',
  oneLiner:
    'Splits the episode into topic chapters with a model trained on YouTube chapters, then titles each one with a local LLM.',
  inputs: ['Transcript segments and words'],
  outputs: ['chapters (start time + title, with model_version)', 'A “Copy for YouTube” description block'],
  steps: [
    {
      title: 'Build text units',
      detail:
        'Sentences are rebuilt from word timestamps and punctuation, 2–15 s long. Training and inference use the same unit, so the model sees the same kind of input in production.',
      terms: ['word-timestamps'],
    },
    {
      title: 'Embed every unit',
      detail: 'A frozen multilingual encoder (multilingual-e5-small) turns each unit into a 384-dimension vector.',
      terms: ['sentence-embedding'],
    },
    {
      title: 'Predict boundaries',
      detail: 'A BiLSTM reads the sequence of vectors and outputs, for each unit, the probability that a chapter starts there.',
      terms: ['bilstm'],
    },
    {
      title: 'Decode with YouTube’s rules',
      detail:
        'Peak picking with chapters at least 60 s long, about 6 per hour, at least 3 in total, the first at 0:00. Each boundary snaps to a real transcript timestamp.',
      terms: ['peak-detection'],
    },
    {
      title: 'Title each chapter',
      detail:
        'The episode language is detected, then the local LLM writes one short title per chapter from that chapter’s transcript text.',
      terms: ['language-id', 'llm', 'zero-shot'],
    },
  ],
  flow: {
    nodes: [
      { id: 'tx', label: 'Transcript words', kind: 'data' },
      { id: 'units', label: 'Sentence units', detail: '2–15 s', kind: 'process' },
      { id: 'enc', label: 'multilingual-e5-small', detail: 'frozen · 384-d', kind: 'model' },
      { id: 'lstm', label: 'BiLSTM tagger', detail: 'P(chapter starts)', kind: 'model' },
      { id: 'dec', label: 'Decode', detail: '≥ 60 s · ~6/h · first at 0:00', kind: 'decision' },
      { id: 'lid', label: 'Language ID', detail: 'fastText', kind: 'model' },
      { id: 'llm', label: 'Title LLM', detail: 'Qwen3.5-9B', kind: 'model' },
      { id: 'out', label: 'chapters', detail: 'start + title', kind: 'output' },
    ],
    edges: [
      { from: 'tx', to: 'units' },
      { from: 'units', to: 'enc' },
      { from: 'enc', to: 'lstm', label: 'vector sequence' },
      { from: 'lstm', to: 'dec' },
      { from: 'tx', to: 'lid' },
      { from: 'dec', to: 'llm', label: 'chapter text' },
      { from: 'lid', to: 'llm' },
      { from: 'llm', to: 'out' },
    ],
  },
  training: {
    planned: false,
    summary:
      'Trained in experiments/m5-chapters on VidChapters-7M (YouTube videos with creator chapters). The BiLSTM ships because it beats the unsupervised TextTiling baseline on validation F1@5s (16.86 vs 7.77). Titles were first a fine-tuned mT5-small; its Spanish titles were unusable, so the owner’s blind review replaced it with the local LLM.',
    flow: {
      nodes: [
        { id: 'raw', label: 'VidChapters-7M', detail: 'ASR + creator chapters', kind: 'data' },
        { id: 'filter', label: 'Stream + filter', detail: '15–120 min · ≥ 3 chapters · EN + ES', kind: 'process' },
        { id: 'split', label: 'Train / val / test', detail: '+ Spanish-only test set', kind: 'process' },
        { id: 'units', label: 'Same text units', detail: 'as production', kind: 'process' },
        { id: 'enc', label: 'Frozen encoder', detail: 'e5-small embeddings', kind: 'model' },
        { id: 'base', label: 'TextTiling baseline', detail: 'unsupervised', kind: 'model' },
        { id: 'train', label: 'Train BiLSTM', detail: 'weighted BCE / focal', kind: 'model' },
        { id: 'eval', label: 'Evaluate', detail: 'F1@5s · Pk · WindowDiff', kind: 'decision' },
        { id: 'bundle', label: 'Versioned bundle', detail: 'backend/models/chapters', kind: 'output' },
      ],
      edges: [
        { from: 'raw', to: 'filter' },
        { from: 'filter', to: 'split' },
        { from: 'split', to: 'units' },
        { from: 'units', to: 'enc' },
        { from: 'enc', to: 'train' },
        { from: 'enc', to: 'base' },
        { from: 'train', to: 'eval' },
        { from: 'base', to: 'eval', label: 'reference' },
        { from: 'eval', to: 'bundle', label: 'winner ships' },
      ],
    },
  },
  model: { name: 'BiLSTM over multilingual-e5-small + Qwen3.5-9B titles', kind: 'trained', runtime: 'CPU (boundaries) + GPU (titles)' },
  metric: {
    name: 'F1@5s (offline evaluation)',
    means: 'How often a predicted chapter start lands within 5 s of a creator’s chapter, measured on held-out videos.',
    doesNotMean: 'There is no score per episode: a chapter list is not right or wrong, so review it before publishing.',
  },
  limitations: [
    'Spanish training data is scarce, so Spanish boundaries are less precise than English ones.',
    'LLM titles can be generic or, rarely, invent a detail.',
  ],
  dependsOn: ['transcribe'],
  source: 'Decisions 0009, 0010, 0016 · experiments/m5-chapters',
};

const relevance: ToolDoc = {
  key: 'relevance',
  title: 'Relevance',
  oneLiner:
    'Scores every 15-second moment by how well it would work as a standalone short, combining transparent heuristics with a local LLM.',
  inputs: ['Transcript words (with speakers, if diarized)', 'Chapter titles (optional context)'],
  outputs: ['relevance_scores: one score in [0, 1] per window', 'The relevance curve and top moments'],
  steps: [
    {
      title: 'Cut the timeline into windows',
      detail:
        '7.5 s blocks are built from word timestamps; two consecutive blocks form a 15 s window, moved by 7.5 s (50 % overlap). About 510 windows for a 64-min episode.',
      terms: ['sliding-window', 'word-timestamps'],
    },
    {
      title: 'Compute heuristic features',
      detail:
        'Topic anchor (density of the episode’s top-30 TF-IDF keywords, weight 0.35), question → answer (0.25), speaker changes per minute (0.20), and 1 − filler ratio (0.20). A window under 5 words scores 0.',
      terms: ['tfidf', 'heuristic', 'feature-weight'],
    },
    {
      title: 'Ask the LLM, in batches',
      detail:
        'Non-overlapping 15 s blocks are sent 20 at a time, in the episode language. The model must answer with valid JSON: one 0–100 score per block. A second pass with the blocks shifted by 7.5 s is averaged in, which makes the scores more stable.',
      terms: ['llm', 'zero-shot', 'json-grammar', 'language-id'],
    },
    {
      title: 'Map LLM scores onto windows',
      detail: 'A window overlaps at most two LLM blocks; its LLM score is their overlap-weighted mean, scaled to [0, 1].',
    },
    {
      title: 'Combine',
      detail:
        'score = 0.4 · heuristic + 0.6 · LLM (the heuristic alone if the LLM failed for that batch). The weights are versioned with the scores.',
      terms: ['model-version'],
    },
  ],
  flow: {
    nodes: [
      { id: 'tx', label: 'Transcript words', detail: '+ speakers', kind: 'data' },
      { id: 'win', label: '15 s windows', detail: '7.5 s hop · 50 % overlap', kind: 'process' },
      { id: 'blocks', label: '15 s blocks', detail: 'no overlap · 20 per call', kind: 'process' },
      { id: 'heur', label: 'Heuristic features', detail: 'topic · Q→A · speakers · fillers', kind: 'process' },
      { id: 'llm', label: 'LLM zero-shot', detail: 'Qwen3.5-9B · 2 passes', kind: 'model' },
      { id: 'map', label: 'Overlap-weighted mean', detail: 'blocks → windows', kind: 'process' },
      { id: 'comb', label: 'Combine', detail: '0.4 · heuristic + 0.6 · LLM', kind: 'decision' },
      { id: 'out', label: 'relevance_scores', detail: 'one per window, 0–1', kind: 'output' },
    ],
    edges: [
      { from: 'tx', to: 'win' },
      { from: 'tx', to: 'blocks' },
      { from: 'win', to: 'heur' },
      { from: 'blocks', to: 'llm' },
      { from: 'llm', to: 'map' },
      { from: 'heur', to: 'comb' },
      { from: 'map', to: 'comb' },
      { from: 'comb', to: 'out' },
    ],
  },
  training: {
    planned: true,
    summary:
      'v0 has no training: it is the bootstrap. Its scores are stored as pseudo-labels, and M15 trains the real model on them: a BiLSTM over frozen sentence embeddings with a sigmoid head, evaluated offline and against v0 before it goes live.',
    flow: {
      nodes: [
        { id: 'v0', label: 'v0 scores, many episodes', detail: 'heuristic + LLM', kind: 'data' },
        { id: 'pl', label: 'Pseudo-labels', detail: 'LLM expected value', kind: 'process' },
        { id: 'emb', label: 'Frozen embeddings', detail: 'per window', kind: 'planned' },
        { id: 'train', label: 'Train BiLSTM', detail: 'sigmoid · temporal smoothness', kind: 'planned' },
        { id: 'eval', label: 'Offline eval + A/B', detail: 'vs v0 on real episodes', kind: 'planned' },
        { id: 'live', label: 'New model_version', detail: 'is_live switch', kind: 'planned' },
      ],
      edges: [
        { from: 'v0', to: 'pl' },
        { from: 'pl', to: 'train', label: 'targets' },
        { from: 'emb', to: 'train', label: 'inputs' },
        { from: 'train', to: 'eval' },
        { from: 'eval', to: 'live', label: 'only if it wins' },
      ],
    },
  },
  model: { name: 'Heuristic v0 + Qwen3.5-9B zero-shot (relevance-p2)', kind: 'hybrid', runtime: 'CPU (features) + GPU (LLM)' },
  metric: {
    name: 'Relevance score (0–1 per window)',
    means: 'How well a moment would stand alone as a short, by v0’s rules, compared with the rest of this episode.',
    doesNotMean: 'Not a prediction of views or retention, and not comparable across episodes: the topic feature is scaled to each episode.',
  },
  limitations: [
    'The v0 quality ceiling is low by design: it is a bootstrap until M15.',
    'The LLM scores are somewhat noisy; the two averaged passes reduce, not remove, that noise.',
    'Without diarization the speaker-change feature is dropped and the other weights are rescaled.',
  ],
  dependsOn: ['transcribe'],
  source: 'Decisions 0012, 0014',
};

const shorts: ToolDoc = {
  key: 'shorts',
  title: 'Shorts',
  oneLiner:
    'Cuts up to five 20–60 s vertical clips around the peaks of the relevance curve, keeps the speaker in frame and burns in karaoke captions.',
  inputs: ['Relevance windows', 'Transcript words', 'The episode video', 'Options: aspect ratio, captions'],
  outputs: ['clips: up to 5 rendered MP4s with their crop track'],
  steps: [
    {
      title: 'Smooth the relevance curve',
      detail: 'A 3-window moving average calms the noise of the v0 scores.',
    },
    {
      title: 'Find the peaks',
      detail: 'Local maxima with prominence ≥ 0.05, at least 60 s apart, are the candidate moments.',
      terms: ['peak-detection'],
    },
    {
      title: 'Grow and snap each clip',
      detail:
        'From each peak the span grows while the curve stays above a floor (70 % of the way from the episode’s median to the peak), then snaps to whole sentences and is kept between 20 and 60 s. Overlapping candidates are dropped and the best 5 are kept.',
    },
    {
      title: 'Detect shots and faces',
      detail:
        'Frames are decoded at 320 px. Colour histograms find camera cuts; every 5th frame goes through the YuNet face detector.',
      terms: ['shot-detection', 'face-detection'],
    },
    {
      title: 'Frame the subject',
      detail:
        'Within each shot the main face (largest × most confident) is followed with an EMA-smoothed crop. Shots without a face fit the whole frame over a blurred background.',
      terms: ['ema'],
    },
    {
      title: 'Captions and render',
      detail:
        'Karaoke captions (3–6 words per cue, the spoken word in yellow) come from the word timestamps. ffmpeg renders each clip in one pass, and ffprobe checks the result.',
      terms: ['word-timestamps'],
    },
  ],
  flow: {
    nodes: [
      { id: 'rel', label: 'Relevance curve', kind: 'data' },
      { id: 'smooth', label: 'Moving average', detail: '3 windows', kind: 'process' },
      { id: 'peaks', label: 'Peak detection', detail: 'prominence ≥ 0.05 · ≥ 60 s apart', kind: 'process' },
      { id: 'spans', label: 'Grow + sentence snap', detail: '20–60 s · top 5', kind: 'decision' },
      { id: 'video', label: 'Episode video', kind: 'data' },
      { id: 'shots', label: 'Shot detection', detail: 'histogram distance', kind: 'process' },
      { id: 'faces', label: 'YuNet faces', detail: '5 fps', kind: 'model' },
      { id: 'frame', label: 'Crop or fit', detail: 'EMA-smoothed centre', kind: 'decision' },
      { id: 'words', label: 'Transcript words', kind: 'data' },
      { id: 'render', label: 'Render + captions', detail: 'ffmpeg · ffprobe check', kind: 'process' },
      { id: 'out', label: 'clips', detail: '≤ 5 MP4s', kind: 'output' },
    ],
    edges: [
      { from: 'rel', to: 'smooth' },
      { from: 'smooth', to: 'peaks' },
      { from: 'peaks', to: 'spans' },
      { from: 'spans', to: 'shots' },
      { from: 'video', to: 'shots' },
      { from: 'shots', to: 'faces' },
      { from: 'faces', to: 'frame' },
      { from: 'frame', to: 'render' },
      { from: 'words', to: 'render', label: 'captions' },
      { from: 'render', to: 'out' },
    ],
  },
  model: { name: 'YuNet face detector + selection rules', kind: 'pretrained', runtime: 'CPU (OpenCV + ffmpeg)' },
  limitations: [
    'Clip quality is bounded by the v0 relevance curve.',
    'A two-shot of both hosts can only be cropped onto one face; with no face the clip falls back to fit.',
  ],
  dependsOn: ['relevance'],
  source: 'Decision 0013',
};

const hook: ToolDoc = {
  key: 'hook',
  title: 'Hook score',
  oneLiner:
    'Scores how strong the first 15 seconds of each short are, from timing and language features, to rank the episode’s clips.',
  inputs: ['The episode’s clips', 'Transcript words', 'Episode title (show notes or filename)'],
  outputs: ['hook_scores: a score, a rank and the full feature vector per clip'],
  steps: [
    {
      title: 'Take the opening window',
      detail: 'The words that start in the clip’s first 15 s; the first sentence is found by spaCy.',
      terms: ['spacy'],
    },
    {
      title: 'Detect the language',
      detail: 'Spanish or English, from the stopword ratio over the transcript. Other languages fall back to English, and the breakdown says so.',
      terms: ['language-id'],
    },
    {
      title: 'Measure the scored features',
      detail:
        'Dead air before the first word, words per second, whether the opening is a question or a curiosity gap (“but…”, “resulta que…”), and direct address (“you”, imperatives).',
    },
    {
      title: 'Measure the unweighted features',
      detail:
        'Readability, numbers, superlatives, named entities, emotion (xlm-emo-t), title overlap… are stored with weight 0 until real data shows their direction.',
      terms: ['feature-weight'],
    },
    {
      title: 'Score and rank',
      detail:
        'Each scored dimension maps to [0, 1]; the total is 0.30 · start + 0.25 · pace + 0.25 · framing + 0.20 · address, × 100. Clips are ranked within the episode; ties stay ties.',
      terms: ['heuristic', 'relative-score'],
    },
  ],
  flow: {
    nodes: [
      { id: 'clips', label: 'Clips', kind: 'data' },
      { id: 'words', label: 'Transcript words', kind: 'data' },
      { id: 'win', label: 'Opening window', detail: 'first 15 s', kind: 'process' },
      { id: 'nlp', label: 'spaCy + emotion model', detail: 'ES / EN', kind: 'model' },
      { id: 'scored', label: 'Scored features', detail: 'dead air · pace · framing · address', kind: 'process' },
      { id: 'zero', label: 'Weight-0 features', detail: 'stored for M15', kind: 'process' },
      { id: 'score', label: 'Weighted sum', detail: '0.30 / 0.25 / 0.25 / 0.20', kind: 'decision' },
      { id: 'out', label: 'hook_scores', detail: 'score · rank · features', kind: 'output' },
    ],
    edges: [
      { from: 'clips', to: 'win' },
      { from: 'words', to: 'win' },
      { from: 'win', to: 'nlp' },
      { from: 'nlp', to: 'scored' },
      { from: 'nlp', to: 'zero' },
      { from: 'scored', to: 'score' },
      { from: 'score', to: 'out' },
      { from: 'zero', to: 'out', label: 'feature vector' },
    ],
  },
  training: {
    planned: true,
    summary:
      'v0 is a hand-weighted heuristic. Every feature vector is stored so that M15 can train a LightGBM ranker once published shorts have a real label: retention, or viewed vs swiped away.',
    flow: {
      nodes: [
        { id: 'fv', label: 'Stored feature vectors', detail: 'every clip', kind: 'data' },
        { id: 'pub', label: 'Published shorts', kind: 'data' },
        { id: 'label', label: 'Real label', detail: 'retention / swipe-away', kind: 'planned' },
        { id: 'join', label: 'Join clip ↔ short', kind: 'planned' },
        { id: 'train', label: 'LightGBM ranker', kind: 'planned' },
        { id: 'eval', label: 'Compare with v0', kind: 'planned' },
      ],
      edges: [
        { from: 'fv', to: 'join' },
        { from: 'pub', to: 'label' },
        { from: 'label', to: 'join' },
        { from: 'join', to: 'train' },
        { from: 'train', to: 'eval' },
      ],
    },
  },
  model: { name: 'Hook heuristic v0 (spaCy features, xlm-emo-t stored only)', kind: 'heuristic v0', runtime: 'CPU' },
  metric: {
    name: 'Hook score (0–100, ranked per episode)',
    means: 'How this clip’s opening compares with the episode’s other clips on four hook signals.',
    doesNotMean: 'Not a retention %, and not comparable across episodes.',
  },
  limitations: [
    'The weights are priors from platform mechanics and creator advice, not learned from data yet.',
    'Only Spanish and English are analysed properly.',
  ],
  dependsOn: ['shorts'],
  source: 'Decision 0020',
};

const shownotes: ToolDoc = {
  key: 'shownotes',
  title: 'Show notes',
  oneLiner:
    'Drafts title suggestions, a description, tags and a summary from the whole transcript, with a local LLM and YouTube’s limits enforced.',
  inputs: ['Transcript', 'Chapters (optional, used as chunks and in the description)'],
  outputs: ['show_notes: 3–5 titles, description, 5–15 tags, summary (editable)'],
  steps: [
    {
      title: 'Chunk the transcript',
      detail:
        'One chunk per chapter when chapters exist, otherwise 1,600-word blocks cut on sentence boundaries. A short transcript skips straight to the reduce step.',
    },
    {
      title: 'Map: summarise each chunk',
      detail: 'The LLM writes 3–5 bullet points per chunk, in the episode language.',
      terms: ['map-reduce', 'llm'],
    },
    {
      title: 'Reduce: write the notes',
      detail:
        'All summaries (and chapter titles) go into one call whose output must match a JSON schema: titles, description, tags, summary.',
      terms: ['json-grammar', 'zero-shot'],
    },
    {
      title: 'Validate and normalise',
      detail:
        'YouTube’s limits are enforced: titles ≤ 100 characters, description ≤ 5,000 with the chapter block, tags ≤ 500 characters in total. Duplicates are removed, and text is cut at sentence ends.',
    },
  ],
  flow: {
    nodes: [
      { id: 'tx', label: 'Transcript', kind: 'data' },
      { id: 'ch', label: 'Chapters', detail: 'optional', kind: 'data' },
      { id: 'chunk', label: 'Chunks', detail: 'per chapter or 1,600 words', kind: 'process' },
      { id: 'map', label: 'Map: summarise', detail: 'LLM · 3–5 bullets', kind: 'model' },
      { id: 'reduce', label: 'Reduce: write notes', detail: 'LLM · JSON schema', kind: 'model' },
      { id: 'val', label: 'Validate', detail: 'YouTube limits', kind: 'decision' },
      { id: 'out', label: 'show_notes', detail: 'editable draft', kind: 'output' },
    ],
    edges: [
      { from: 'tx', to: 'chunk' },
      { from: 'ch', to: 'chunk' },
      { from: 'chunk', to: 'map' },
      { from: 'map', to: 'reduce', label: 'summaries' },
      { from: 'ch', to: 'reduce', label: 'titles' },
      { from: 'reduce', to: 'val' },
      { from: 'val', to: 'out' },
    ],
  },
  model: { name: 'Qwen3.5-9B Q4_K_M (llama.cpp)', kind: 'LLM zero-shot', runtime: 'GPU' },
  limitations: [
    'An LLM can state something that was not said: review the draft before publishing.',
    'Running it again replaces the draft, including your edits.',
  ],
  dependsOn: ['transcribe'],
  source: 'Decisions 0011, 0016',
};

const ctr: ToolDoc = {
  key: 'ctr',
  title: 'CTR score',
  oneLiner:
    'Ranks your title/thumbnail candidates for this episode by features linked to click-through, explaining every sub-score.',
  inputs: ['Up to 10 candidates: a title (≤ 100 chars) and an optional thumbnail'],
  outputs: ['ctr_predictions: a relative score, a rank and every feature per candidate'],
  steps: [
    {
      title: 'Title features',
      detail:
        'Length in characters is scored. Words, digits, questions, exclamations, brackets, capitals and second person are measured with weight 0.',
      terms: ['feature-weight'],
    },
    {
      title: 'Thumbnail features',
      detail:
        'The image is resized to fit 1280 × 720. YuNet counts faces and measures the largest; a DB text detector measures how much of the image is text. Brightness, contrast and saturation are stored with weight 0.',
      terms: ['face-detection', 'text-detection'],
    },
    {
      title: 'Sub-scores',
      detail:
        'Each scored feature maps to [0, 1] through a shape based on the research: 40–60 characters is best, 2 faces (two hosts) is best, a face covering 10–35 % of the frame is best, and 2–20 % text is best.',
      terms: ['heuristic'],
    },
    {
      title: 'Weighted sum and rank',
      detail:
        '0.35 · title length + 0.30 · faces + 0.20 · face size + 0.15 · text coverage, × 100. Title-only candidates are scored on the title alone and ranked in their own group.',
      terms: ['relative-score'],
    },
  ],
  flow: {
    nodes: [
      { id: 'title', label: 'Candidate title', kind: 'data' },
      { id: 'thumb', label: 'Candidate thumbnail', detail: 'optional', kind: 'data' },
      { id: 'tf', label: 'Title features', detail: 'length + weight-0 features', kind: 'process' },
      { id: 'faces', label: 'YuNet faces', detail: 'count + largest area', kind: 'model' },
      { id: 'text', label: 'DB text detector', detail: 'text coverage', kind: 'model' },
      { id: 'sub', label: 'Sub-scores', detail: 'shapes from research', kind: 'process' },
      { id: 'sum', label: 'Weighted sum', detail: '0.35 / 0.30 / 0.20 / 0.15', kind: 'decision' },
      { id: 'out', label: 'ctr_predictions', detail: 'relative rank', kind: 'output' },
    ],
    edges: [
      { from: 'title', to: 'tf' },
      { from: 'thumb', to: 'faces' },
      { from: 'thumb', to: 'text' },
      { from: 'tf', to: 'sub' },
      { from: 'faces', to: 'sub' },
      { from: 'text', to: 'sub' },
      { from: 'sub', to: 'sum' },
      { from: 'sum', to: 'out' },
    ],
  },
  training: {
    planned: true,
    summary:
      'v0 is a hand-weighted heuristic. M12 now collects the channel’s real CTR and impressions; once each published title/thumbnail is linked to its video, the stored features become a training set and M15 learns the weights from real CTR.',
    flow: {
      nodes: [
        { id: 'fv', label: 'Stored features', detail: 'every candidate', kind: 'data' },
        { id: 'real', label: 'Real CTR', detail: 'Reporting API · daily CSV', kind: 'data' },
        { id: 'link', label: 'Link candidate ↔ video', detail: 'open question', kind: 'planned' },
        { id: 'train', label: 'Learn weights / model', kind: 'planned' },
        { id: 'eval', label: 'Compare with v0', kind: 'planned' },
      ],
      edges: [
        { from: 'fv', to: 'link' },
        { from: 'real', to: 'link', label: 'labels' },
        { from: 'link', to: 'train' },
        { from: 'train', to: 'eval' },
      ],
    },
  },
  model: { name: 'CTR heuristic v0 (YuNet + DB text detector)', kind: 'heuristic v0', runtime: 'CPU' },
  metric: {
    name: 'CTR score (0–100, ranked per episode)',
    means: 'How this candidate compares with this episode’s other candidates on four features linked to clicks.',
    doesNotMean: 'Not a CTR %, and not comparable across episodes. Title-only candidates are not comparable with full ones.',
  },
  limitations: [
    'Candidates of similar length often tie on the title sub-score; the panel shows ties as ties.',
    'Faces are detected, not recognised; text is detected, not read.',
  ],
  dependsOn: [],
  source: 'Decision 0019 · docs/research/ctr-heuristic-evidence.md',
};

const upload: ToolDoc = {
  key: 'upload',
  title: 'Episode upload',
  oneLiner: 'The raw input: one video file per episode, stored once and read by every tool.',
  inputs: ['A video file (any format ffmpeg reads)'],
  outputs: ['uploads row + the stored file'],
  steps: [
    {
      title: 'Store the file',
      detail: 'The file is saved under a unique, collision-safe key; the database row records the key, never a raw path.',
    },
    {
      title: 'Queue tools',
      detail: 'Each tool run is a job (queued → running → succeeded / failed) processed in the background by a worker.',
    },
  ],
  flow: {
    nodes: [
      { id: 'file', label: 'Video file', kind: 'data' },
      { id: 'store', label: 'File storage', detail: 'unique key', kind: 'process' },
      { id: 'row', label: 'uploads', kind: 'output' },
    ],
    edges: [
      { from: 'file', to: 'store' },
      { from: 'store', to: 'row' },
    ],
  },
  model: { name: '—', kind: 'data pipeline', runtime: 'API + background worker' },
  limitations: [],
  dependsOn: [],
  source: 'Decisions 0002, 0006',
};

const market: ToolDoc = {
  key: 'market',
  title: 'Market data',
  oneLiner: 'Daily public statistics for a curated list of podcast channels, building a time series of how competitors’ videos perform.',
  inputs: ['A curated list of channels', 'YouTube Data API v3'],
  outputs: ['Channel and video snapshots (views, likes, comments) over time'],
  steps: [
    {
      title: 'Backfill each channel',
      detail: 'Adding a channel resolves it and reads its whole catalogue through the uploads playlist.',
    },
    {
      title: 'Detect new videos',
      detail: 'An RSS check every hour gives new uploads their first snapshot within the hour.',
    },
    {
      title: 'Daily snapshots',
      detail: 'Every day, videos from the last 90 days are re-read; older ones weekly. Each read is a dated snapshot row.',
      terms: ['snapshot'],
    },
    {
      title: 'Stay within the quota',
      detail: 'About 160 of the 10,000 daily API units for ~25 channels; a soft budget stops a run cleanly and keeps what it wrote.',
    },
  ],
  flow: {
    nodes: [
      { id: 'list', label: 'Curated channels', kind: 'data' },
      { id: 'api', label: 'YouTube Data API v3', kind: 'process' },
      { id: 'rss', label: 'Hourly RSS check', detail: 'new uploads', kind: 'process' },
      { id: 'daily', label: 'Daily pull', detail: '≤ 90 days daily · older weekly', kind: 'process' },
      { id: 'out', label: 'Market snapshots', detail: 'time series', kind: 'output' },
    ],
    edges: [
      { from: 'list', to: 'api' },
      { from: 'api', to: 'rss' },
      { from: 'api', to: 'daily' },
      { from: 'rss', to: 'out' },
      { from: 'daily', to: 'out' },
    ],
  },
  model: { name: '—', kind: 'data pipeline', runtime: 'Scheduled worker tasks' },
  limitations: ['Public data only: no impressions, CTR or retention for other channels.'],
  dependsOn: [],
  source: 'Decision 0015',
};

const analytics: ToolDoc = {
  key: 'analytics',
  title: 'Channel analytics',
  oneLiner: 'Your own channel’s private engagement and audience-retention curves, pulled daily through OAuth.',
  inputs: ['Your connected channel (OAuth)', 'YouTube Analytics API v2'],
  outputs: ['analytics_snapshots (daily engagement per video)', 'analytics_retention_curve (100 points per video)'],
  steps: [
    {
      title: 'Connect once',
      detail: 'You grant read-only access; the tokens are stored encrypted.',
    },
    {
      title: 'Daily engagement',
      detail: 'One row per video per metric day. Each run re-reads the last 7 days because Analytics lags 48–72 h.',
      terms: ['snapshot'],
    },
    {
      title: 'Retention curves',
      detail: '100 points per video: the share of viewers still watching at each 1 % of the video.',
      terms: ['retention'],
    },
  ],
  flow: {
    nodes: [
      { id: 'oauth', label: 'OAuth connection', kind: 'data' },
      { id: 'api', label: 'Analytics API v2', detail: 'daily · 7-day re-read', kind: 'process' },
      { id: 'eng', label: 'Engagement snapshots', kind: 'output' },
      { id: 'ret', label: 'Retention curves', detail: '100 points', kind: 'output' },
    ],
    edges: [
      { from: 'oauth', to: 'api' },
      { from: 'api', to: 'eng' },
      { from: 'api', to: 'ret' },
    ],
  },
  model: { name: '—', kind: 'data pipeline', runtime: 'Scheduled worker task' },
  limitations: ['Data arrives 48–72 h late.', 'In Testing mode the connection must be renewed about weekly.'],
  dependsOn: [],
  source: 'Decision 0017',
};

const reporting: ToolDoc = {
  key: 'reporting',
  title: 'Real CTR & impressions',
  oneLiner: 'Thumbnail impressions and real click-through rate for your channel, from YouTube’s bulk Reporting API.',
  inputs: ['Your connected channel (OAuth)', 'YouTube Reporting API'],
  outputs: ['analytics_reach_snapshots: impressions and CTR per video per day'],
  steps: [
    {
      title: 'Create a reporting job',
      detail: 'Once, a channel_reach_basic_a1 job is created at Google.',
    },
    {
      title: 'Download daily reports',
      detail: 'YouTube posts one CSV per day (the first within 48 h, then 30 days of history). Each daily poll downloads the new files.',
    },
    {
      title: 'Parse and normalise',
      detail: 'Headers and dates are checked, and CTR is stored as a 0–1 fraction, one row per video per day.',
      terms: ['ctr', 'snapshot'],
    },
  ],
  flow: {
    nodes: [
      { id: 'job', label: 'Reach reporting job', kind: 'data' },
      { id: 'csv', label: 'Daily CSV reports', kind: 'process' },
      { id: 'parse', label: 'Parse + normalise', detail: 'CTR as 0–1', kind: 'decision' },
      { id: 'out', label: 'Reach snapshots', detail: 'impressions · CTR', kind: 'output' },
    ],
    edges: [
      { from: 'job', to: 'csv' },
      { from: 'csv', to: 'parse' },
      { from: 'parse', to: 'out' },
    ],
  },
  model: { name: '—', kind: 'data pipeline', runtime: 'Scheduled worker task' },
  limitations: ['Reports only exist from the job’s creation onwards (plus ~30 days of history).'],
  dependsOn: [],
  source: 'Decision 0018',
};

const m15: ToolDoc = {
  key: 'm15',
  title: 'Trained models (M15, planned)',
  oneLiner:
    'The v0 heuristics are a transparent starting point. M15 retrains the three custom models on the data the platform accumulates.',
  inputs: ['v0 relevance scores (pseudo-labels)', 'Real CTR and impressions', 'Audience retention', 'Stored feature vectors'],
  outputs: ['New model versions, switched live only if they beat v0'],
  steps: [
    {
      title: 'Relevance',
      detail: 'A BiLSTM over frozen sentence embeddings, trained on the v0 pseudo-labels.',
      terms: ['pseudo-label', 'bilstm'],
    },
    {
      title: 'CTR',
      detail: 'Feature weights learned from the channel’s real CTR, once candidates are linked to published videos.',
      terms: ['ctr'],
    },
    {
      title: 'Hook',
      detail: 'A LightGBM ranker over the stored hook features, labelled with Shorts retention.',
      terms: ['retention'],
    },
    {
      title: 'Ship carefully',
      detail: 'Each candidate model is evaluated offline and against v0 on real episodes before its model version goes live.',
      terms: ['model-version'],
    },
  ],
  flow: {
    nodes: [
      { id: 'data', label: 'Accumulated data', detail: 'pseudo-labels · CTR · retention', kind: 'data' },
      { id: 'train', label: 'Train', detail: 'BiLSTM · weights · LightGBM', kind: 'planned' },
      { id: 'eval', label: 'Evaluate vs v0', kind: 'planned' },
      { id: 'live', label: 'Go live', detail: 'new model_version', kind: 'planned' },
    ],
    edges: [
      { from: 'data', to: 'train' },
      { from: 'train', to: 'eval' },
      { from: 'eval', to: 'live', label: 'only if better' },
    ],
  },
  model: { name: 'Planned', kind: 'planned', runtime: 'GPU training' },
  limitations: ['There is no guarantee the first trained relevance model beats the heuristic: it ships only if it does.'],
  dependsOn: [],
  source: 'docs/development-plan.md (M15)',
};

export const TOOL_DOCS: Record<DocKey, ToolDoc> = {
  transcribe,
  diarize,
  chapters,
  relevance,
  shorts,
  hook,
  shownotes,
  ctr,
  upload,
  market,
  analytics,
  reporting,
  m15,
};

/** The whole platform, for the How it works page. Every node links to its doc. */
export const SYSTEM_FLOW: Flow = {
  nodes: [
    { id: 'upload', label: 'Episode video', kind: 'data', link: 'upload' },
    { id: 'transcribe', label: 'Transcript', detail: 'Whisper', kind: 'model', link: 'transcribe' },
    { id: 'diarize', label: 'Speakers', detail: 'pyannote', kind: 'model', link: 'diarize' },
    { id: 'chapters', label: 'Chapters', detail: 'trained BiLSTM + LLM', kind: 'model', link: 'chapters' },
    { id: 'relevance', label: 'Relevance', detail: 'heuristic v0 + LLM', kind: 'model', link: 'relevance' },
    { id: 'shownotes', label: 'Show notes', detail: 'LLM', kind: 'model', link: 'shownotes' },
    { id: 'shorts', label: 'Shorts', detail: 'peaks + face framing', kind: 'process', link: 'shorts' },
    { id: 'ctr', label: 'CTR score', detail: 'heuristic v0', kind: 'decision', link: 'ctr' },
    { id: 'hook', label: 'Hook score', detail: 'heuristic v0', kind: 'decision', link: 'hook' },
    { id: 'market', label: 'Market data', detail: 'Data API', kind: 'data', link: 'market', minLayer: 5 },
    { id: 'analytics', label: 'Channel analytics', detail: 'retention', kind: 'data', link: 'analytics', minLayer: 5 },
    { id: 'reporting', label: 'Real CTR', detail: 'Reporting API', kind: 'data', link: 'reporting', minLayer: 5 },
    { id: 'm15', label: 'Trained models', detail: 'M15 · replace v0', kind: 'planned', link: 'm15' },
  ],
  edges: [
    { from: 'upload', to: 'transcribe' },
    { from: 'transcribe', to: 'diarize' },
    { from: 'transcribe', to: 'chapters' },
    { from: 'diarize', to: 'relevance', label: 'words + speakers' },
    { from: 'chapters', to: 'shownotes', label: 'transcript + chapters' },
    { from: 'relevance', to: 'shorts' },
    { from: 'shownotes', to: 'ctr', label: 'title candidates' },
    { from: 'shorts', to: 'hook' },
    { from: 'relevance', to: 'm15', label: 'pseudo-labels' },
    { from: 'ctr', to: 'm15', label: 'features' },
    { from: 'hook', to: 'm15', label: 'features' },
    { from: 'market', to: 'm15' },
    { from: 'analytics', to: 'm15', label: 'retention' },
    { from: 'reporting', to: 'm15', label: 'real CTR' },
  ],
};
