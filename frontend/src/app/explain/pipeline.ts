/**
 * Pure tool-state logic shared by the Manage list, the episode pipeline
 * stepper and the tool tabs (docs/decisions/0021). Moved out of the old
 * VideoRow component so every view agrees on what is done, running,
 * blocked or ready, and why.
 */
import { JobOut, TOOL_LABELS, TOOL_TYPES, ToolType } from '../jobs/job.model';

export type ToolStatus = 'done' | 'running' | 'queued' | 'failed' | 'ready' | 'blocked';

export interface ToolState {
  tool: ToolType;
  status: ToolStatus;
  /** Latest job, if any. */
  job?: JobOut;
  /** Any job of this tool has succeeded: its results exist (they survive a re-run). */
  hasResult: boolean;
  /** Why the tool can't run yet (status 'blocked'), as a short sentence. */
  blockedReason: string | null;
  /** The tool to run (or tab to open) to unblock this one. */
  unblockedBy: ToolType | null;
}

/** Extra facts only an open episode knows; the list view leaves them out. */
export interface PipelineContext {
  /** CTR candidates in the panel; undefined = unknown (list view): don't block. */
  ctrCandidates?: number;
  /** Clips loaded for the episode; undefined = fall back to a succeeded shorts job. */
  clipCount?: number;
}

/** The order the stepper suggests tools in: each tool after what it needs. */
export const SUGGESTED_ORDER: readonly ToolType[] = [
  'transcribe',
  'diarize',
  'chapters',
  'relevance',
  'shorts',
  'hook',
  'shownotes',
  'ctr',
];

const byNewest = (a: JobOut, b: JobOut) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime();

export function latestJob(jobs: readonly JobOut[], tool: ToolType): JobOut | undefined {
  return jobs.filter((j) => j.tool_type === tool).sort(byNewest)[0];
}

export function newestSucceededJobId(jobs: readonly JobOut[], tool: ToolType): number | null {
  return jobs.filter((j) => j.tool_type === tool && j.status === 'succeeded').sort(byNewest)[0]?.id ?? null;
}

export function hasSucceeded(jobs: readonly JobOut[], tool: ToolType): boolean {
  return jobs.some((j) => j.tool_type === tool && j.status === 'succeeded');
}

function blocker(
  tool: ToolType,
  jobs: readonly JobOut[],
  ctx: PipelineContext,
): { reason: string; by: ToolType } | null {
  const transcriptTools: ToolType[] = ['diarize', 'chapters', 'relevance', 'shownotes'];
  if (transcriptTools.includes(tool) && !hasSucceeded(jobs, 'transcribe')) {
    return { reason: 'Needs a transcript', by: 'transcribe' };
  }
  if (tool === 'shorts' && !hasSucceeded(jobs, 'relevance')) {
    return { reason: 'Needs relevance scores', by: 'relevance' };
  }
  if (tool === 'hook') {
    const hasClips = ctx.clipCount !== undefined ? ctx.clipCount > 0 : hasSucceeded(jobs, 'shorts');
    if (!hasClips) return { reason: 'Needs shorts', by: 'shorts' };
  }
  if (tool === 'ctr' && ctx.ctrCandidates === 0) {
    return { reason: 'Add title/thumbnail candidates first', by: 'ctr' };
  }
  return null;
}

export function toolState(tool: ToolType, jobs: readonly JobOut[], ctx: PipelineContext = {}): ToolState {
  const job = latestJob(jobs, tool);
  const hasResult = hasSucceeded(jobs, tool);
  const block = blocker(tool, jobs, ctx);
  let status: ToolStatus;
  if (job?.status === 'queued' || job?.status === 'running') status = job.status;
  else if (job?.status === 'failed') status = 'failed';
  else if (job?.status === 'succeeded') status = 'done';
  else status = block ? 'blocked' : 'ready';
  return {
    tool,
    status,
    job,
    hasResult,
    blockedReason: block?.reason ?? null,
    unblockedBy: block?.by ?? null,
  };
}

export function toolStates(jobs: readonly JobOut[], ctx: PipelineContext = {}): Record<ToolType, ToolState> {
  return Object.fromEntries(TOOL_TYPES.map((t) => [t, toolState(t, jobs, ctx)])) as Record<ToolType, ToolState>;
}

/** Run is offered when nothing is in flight and nothing it needs is missing. */
export function canRun(state: ToolState): boolean {
  return state.status !== 'queued' && state.status !== 'running' && state.blockedReason === null;
}

export interface Suggestion {
  state: ToolState;
  /** One sentence: why this step, now. */
  why: string;
}

/**
 * The next step for an episode: a failed tool first (it needs attention),
 * then the first tool in dependency order that has never succeeded and
 * can run. Null when every tool is done or in flight.
 */
export function nextSuggested(states: Record<ToolType, ToolState>): Suggestion | null {
  const failed = SUGGESTED_ORDER.map((t) => states[t]).find((s) => s.status === 'failed');
  if (failed) {
    return { state: failed, why: `${TOOL_LABELS[failed.tool]} failed last time. Check the error and run it again.` };
  }
  const inFlight = SUGGESTED_ORDER.some((t) => states[t].status === 'queued' || states[t].status === 'running');
  for (const tool of SUGGESTED_ORDER) {
    const s = states[tool];
    if (s.hasResult || s.status === 'queued' || s.status === 'running') continue;
    if (s.status === 'blocked') {
      // A blocker that is itself in flight: wait rather than suggest.
      if (inFlight) return null;
      if (s.unblockedBy === tool) return { state: s, why: `${s.blockedReason}, then run ${TOOL_LABELS[tool]}.` };
      continue;
    }
    return { state: s, why: whyNow(tool) };
  }
  return null;
}

function whyNow(tool: ToolType): string {
  switch (tool) {
    case 'transcribe':
      return 'Every other tool works on the transcript, so it comes first.';
    case 'diarize':
      return 'Speaker labels improve the relevance scores and the subtitles.';
    case 'chapters':
      return 'Chapters split the episode by topic and give show notes their structure.';
    case 'relevance':
      return 'Relevance finds the best moments, which shorts are cut from.';
    case 'shorts':
      return 'Relevance is ready: cut the best moments into shorts.';
    case 'hook':
      return 'Shorts are ready: score how strong each one’s opening is.';
    case 'shownotes':
      return 'Draft titles, a description and tags from the full transcript.';
    case 'ctr':
      return 'Rank your title and thumbnail candidates.';
  }
}

/** How many of the 8 tools have results. */
export function doneCount(states: Record<ToolType, ToolState>): number {
  return TOOL_TYPES.filter((t) => states[t].hasResult).length;
}

/** "SHOW 1X09： Title [videoId].mkv" → "SHOW 1X09： Title". */
export function displayName(filename: string): string {
  const cleaned = filename
    .replace(/\.[A-Za-z0-9]{2,4}$/, '')
    .replace(/\s*\[[A-Za-z0-9_-]{6,}\]$/, '')
    .trim();
  return cleaned || filename;
}

export function formatDuration(seconds: number | null): string | null {
  if (seconds == null) return null;
  const total = Math.round(seconds);
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const s = total % 60;
  const mm = h ? m.toString().padStart(2, '0') : m.toString();
  return `${h ? h + ':' : ''}${mm}:${s.toString().padStart(2, '0')}`;
}
