/** Mirrors backend/app/schemas/job.py's `JobOut`. */
export interface JobOut {
  id: number;
  upload_id: number;
  tool_type: ToolType;
  status: JobStatus;
  progress_pct: number | null;
  started_at: string | null;
  completed_at: string | null;
  error_message: string | null;
  /** Per-run tool options (M8: shorts only); null for other tools. */
  params?: ShortsParams | null;
  created_at: string;
}

/** Mirrors backend/app/schemas/job.py's `ShortsParams` (docs/decisions/0013 §5). */
export type AspectRatio = '9:16' | '1:1' | '4:5' | '16:9';

export const ASPECT_RATIOS: readonly AspectRatio[] = ['9:16', '1:1', '4:5', '16:9'];

export interface ShortsParams {
  aspect_ratio: AspectRatio;
  captions: boolean;
}

export type ToolType =
  | 'transcribe'
  | 'diarize'
  | 'chapters'
  | 'relevance'
  | 'shorts'
  | 'shownotes'
  | 'ctr'
  | 'hook';

export type JobStatus = 'queued' | 'running' | 'succeeded' | 'failed';

/** All 8 v1 tools, in the order they're introduced across M3-M14. */
export const TOOL_TYPES: readonly ToolType[] = [
  'transcribe',
  'diarize',
  'chapters',
  'relevance',
  'shorts',
  'shownotes',
  'ctr',
  'hook',
];

export const TOOL_LABELS: Record<ToolType, string> = {
  transcribe: 'Transcript',
  diarize: 'Speakers',
  chapters: 'Chapters',
  relevance: 'Relevance',
  shorts: 'Shorts',
  shownotes: 'Show notes',
  ctr: 'CTR score',
  hook: 'Hook score',
};
