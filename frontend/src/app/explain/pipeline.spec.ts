import { JobOut, ToolType } from '../jobs/job.model';
import { canRun, displayName, doneCount, formatDuration, nextSuggested, toolState, toolStates } from './pipeline';

let nextId = 1;
function job(tool: ToolType, status: JobOut['status'], created_at = '2026-08-14T12:00:00Z'): JobOut {
  return {
    id: nextId++,
    upload_id: 1,
    tool_type: tool,
    status,
    progress_pct: null,
    started_at: null,
    completed_at: null,
    error_message: null,
    created_at,
  };
}

describe('pipeline state', () => {
  it('a fresh episode: transcript ready, everything that needs it blocked', () => {
    const s = toolStates([]);
    expect(s.transcribe.status).toBe('ready');
    for (const t of ['diarize', 'chapters', 'relevance', 'shownotes'] as const) {
      expect(s[t]).toMatchObject({ status: 'blocked', blockedReason: 'Needs a transcript', unblockedBy: 'transcribe' });
    }
    expect(s.shorts).toMatchObject({ status: 'blocked', unblockedBy: 'relevance' });
    expect(s.hook).toMatchObject({ status: 'blocked', unblockedBy: 'shorts' });
    // Unknown candidate count (list view): CTR isn't blocked.
    expect(s.ctr.status).toBe('ready');
  });

  it('uses the latest job for the status, but any success for hasResult', () => {
    const jobs = [job('transcribe', 'succeeded'), job('transcribe', 'running', '2026-08-15T12:00:00Z')];
    const s = toolState('transcribe', jobs);
    expect(s.status).toBe('running');
    expect(s.hasResult).toBe(true);
    expect(canRun(s)).toBe(false);
  });

  it('an episode view knows about clips and candidates', () => {
    const jobs = [job('transcribe', 'succeeded'), job('relevance', 'succeeded'), job('shorts', 'succeeded')];
    expect(toolState('hook', jobs, { clipCount: 0 }).status).toBe('blocked');
    expect(toolState('hook', jobs, { clipCount: 3 }).status).toBe('ready');
    expect(toolState('ctr', jobs, { ctrCandidates: 0 })).toMatchObject({ status: 'blocked', unblockedBy: 'ctr' });
  });

  it('counts tools with results', () => {
    expect(doneCount(toolStates([job('transcribe', 'succeeded'), job('chapters', 'failed')]))).toBe(1);
  });
});

describe('nextSuggested', () => {
  it('starts with the transcript', () => {
    expect(nextSuggested(toolStates([]))?.state.tool).toBe('transcribe');
  });

  it('waits while a prerequisite is running', () => {
    expect(nextSuggested(toolStates([job('transcribe', 'running')]))).toBeNull();
  });

  it('follows dependency order', () => {
    const jobs = [job('transcribe', 'succeeded'), job('diarize', 'succeeded'), job('chapters', 'succeeded')];
    expect(nextSuggested(toolStates(jobs))?.state.tool).toBe('relevance');
  });

  it('puts a failed tool first', () => {
    const jobs = [job('transcribe', 'succeeded'), job('chapters', 'failed')];
    const s = nextSuggested(toolStates(jobs));
    expect(s?.state.tool).toBe('chapters');
    expect(s?.why).toContain('failed');
  });

  it('points at CTR candidates once the rest is done', () => {
    const done: ToolType[] = ['transcribe', 'diarize', 'chapters', 'relevance', 'shorts', 'hook', 'shownotes'];
    const s = nextSuggested(toolStates(done.map((t) => job(t, 'succeeded')), { ctrCandidates: 0, clipCount: 5 }));
    expect(s?.state.tool).toBe('ctr');
    expect(s?.why).toContain('candidates');
  });

  it('returns null when every tool has run', () => {
    const all: ToolType[] = ['transcribe', 'diarize', 'chapters', 'relevance', 'shorts', 'hook', 'shownotes', 'ctr'];
    expect(nextSuggested(toolStates(all.map((t) => job(t, 'succeeded'))))).toBeNull();
  });
});

describe('display helpers', () => {
  it.each([
    ['SASTRE Y MALDONADO 1X09： ¿Torres y Sastre？ [oSCkZ8yVz7M].mkv', 'SASTRE Y MALDONADO 1X09： ¿Torres y Sastre？'],
    ['episode-04.mp4', 'episode-04'],
    ['no extension', 'no extension'],
    ['.mp4', '.mp4'],
  ])('displayName(%s)', (input, expected) => {
    expect(displayName(input)).toBe(expected);
  });

  it.each([
    [null, null],
    [125, '2:05'],
    [3125, '52:05'],
    [4000, '1:06:40'],
  ])('formatDuration(%s)', (input, expected) => {
    expect(formatDuration(input)).toBe(expected);
  });
});
