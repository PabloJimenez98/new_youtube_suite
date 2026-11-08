import { provideHttpClient } from '@angular/common/http';
import { provideHttpClientTesting } from '@angular/common/http/testing';
import { ComponentFixture, TestBed } from '@angular/core/testing';
import { of, throwError } from 'rxjs';

import { TOKEN_STORAGE_KEY } from '../../auth/auth';
import { ChapterOut, Chapters } from '../../chapters/chapters';
import { Explain } from '../../explain/explain';
import { JobOut, ToolType } from '../../jobs/job.model';
import { Jobs } from '../../jobs/jobs';
import { Relevance, RelevanceOut } from '../../relevance/relevance';
import { ClipOut, Shorts } from '../../shorts/shorts';
import { ShowNotes, ShowNotesOut } from '../../shownotes/shownotes';
import { Transcripts } from '../../transcripts/transcripts';
import { UploadOut } from '../../uploads/upload.model';
import { EpisodeWorkspace } from './episode-workspace';

const upload: UploadOut = {
  id: 5,
  filename: 'episode.mp4',
  storage_path: 'x',
  duration_seconds: null,
  uploaded_at: '2026-08-14T12:00:00Z',
  status: 'uploaded',
};

function makeJob(overrides: Partial<JobOut> = {}): JobOut {
  return {
    id: 1,
    upload_id: 5,
    tool_type: 'transcribe',
    status: 'queued',
    progress_pct: null,
    started_at: null,
    completed_at: null,
    error_message: null,
    created_at: '2026-08-14T12:00:00Z',
    ...overrides,
  };
}

describe('EpisodeWorkspace', () => {
  let fixture: ComponentFixture<EpisodeWorkspace>;
  let jobsStub: { createJob: ReturnType<typeof vi.fn> };
  let transcriptsStub: { getSubtitle: ReturnType<typeof vi.fn>; getTranscript: ReturnType<typeof vi.fn> };
  let chaptersStub: { getChapters: ReturnType<typeof vi.fn> };
  let relevanceStub: { getRelevance: ReturnType<typeof vi.fn> };
  let shortsStub: { getClips: ReturnType<typeof vi.fn> };
  let showNotesStub: { getShowNotes: ReturnType<typeof vi.fn>; updateShowNotes: ReturnType<typeof vi.fn> };

  const el = () => fixture.nativeElement as HTMLElement;
  /** The tool bar of one tool (every tab stays mounted, so every bar is in the DOM). */
  const toolBar = (tool: ToolType) => el().querySelector<HTMLElement>(`.tool-bar[data-tool="${tool}"]`)!;
  const runButton = (tool: ToolType) => toolBar(tool).querySelector<HTMLButtonElement>('.tool-bar__run');

  beforeEach(async () => {
    jobsStub = { createJob: vi.fn() };
    transcriptsStub = { getSubtitle: vi.fn(), getTranscript: vi.fn().mockReturnValue(of({ segments: [], speakers: [] })) };
    chaptersStub = { getChapters: vi.fn().mockReturnValue(of([])) };
    showNotesStub = { getShowNotes: vi.fn(), updateShowNotes: vi.fn() };
    relevanceStub = { getRelevance: vi.fn() };
    shortsStub = { getClips: vi.fn() };
    localStorage.setItem(TOKEN_STORAGE_KEY, 'fake-token');

    await TestBed.configureTestingModule({
      imports: [EpisodeWorkspace],
      providers: [
        { provide: Jobs, useValue: jobsStub },
        { provide: Transcripts, useValue: transcriptsStub },
        { provide: Chapters, useValue: chaptersStub },
        { provide: ShowNotes, useValue: showNotesStub },
        { provide: Relevance, useValue: relevanceStub },
        { provide: Shorts, useValue: shortsStub },
        provideHttpClient(),
        provideHttpClientTesting(),
      ],
    }).compileComponents();

    fixture = TestBed.createComponent(EpisodeWorkspace);
    fixture.componentRef.setInput('upload', upload);
  });

  afterEach(() => {
    localStorage.removeItem(TOKEN_STORAGE_KEY);
  });

  it('builds the video preview URL with the upload id and auth token', () => {
    fixture.componentRef.setInput('jobs', []);
    fixture.detectChanges();
    expect(el().querySelector('.ws__video')?.getAttribute('src')).toBe('/uploads/5/file?token=fake-token');
  });

  it('shows a Run button for every tool with no job yet', () => {
    fixture.componentRef.setInput('jobs', []);
    fixture.detectChanges();
    expect(el().querySelectorAll('.tool-bar__run').length).toBe(8);
  });

  it('hides Run (shows only the badge) while a job is queued/running', () => {
    fixture.componentRef.setInput('jobs', [makeJob({ tool_type: 'transcribe', status: 'queued' })]);
    fixture.detectChanges();

    expect(el().querySelectorAll('app-job-status-badge').length).toBe(1);
    expect(el().querySelectorAll('.tool-bar__run').length).toBe(7);
  });

  it.each(['succeeded', 'failed'] as const)('shows Run again alongside the badge once the latest job is %s', (status) => {
    fixture.componentRef.setInput('jobs', [makeJob({ tool_type: 'transcribe', status })]);
    fixture.detectChanges();

    expect(el().querySelectorAll('app-job-status-badge').length).toBe(1);
    expect(el().querySelectorAll('.tool-bar__run').length).toBe(8);
  });

  it('labels Run as Re-run once the tool has a result', () => {
    fixture.componentRef.setInput('jobs', [makeJob({ status: 'succeeded' })]);
    fixture.detectChanges();
    expect(runButton('transcribe')!.textContent!.trim()).toBe('Re-run');
    expect(runButton('chapters')!.textContent!.trim()).toBe('Run');
  });

  it('runTool creates a job and optimistically shows its status', () => {
    fixture.componentRef.setInput('jobs', [makeJob({ status: 'succeeded' })]);
    jobsStub.createJob.mockReturnValue(of(makeJob({ id: 2, tool_type: 'chapters' })));
    fixture.detectChanges();

    fixture.componentInstance['runTool']('chapters');
    fixture.detectChanges();

    expect(jobsStub.createJob).toHaveBeenCalledWith(5, 'chapters');
    expect(toolBar('chapters').querySelector('app-job-status-badge')).not.toBeNull();
  });

  it('does not double-count an optimistic job once the poll returns it', () => {
    const created = makeJob({ id: 2, tool_type: 'chapters' });
    fixture.componentRef.setInput('jobs', [makeJob({ status: 'succeeded' })]);
    jobsStub.createJob.mockReturnValue(of(created));
    fixture.detectChanges();
    fixture.componentInstance['runTool']('chapters');
    fixture.componentRef.setInput('jobs', [makeJob({ status: 'succeeded' }), { ...created, status: 'running' }]);
    fixture.detectChanges();

    expect(fixture.componentInstance['states']().chapters.status).toBe('running');
  });

  it('resets the running state if job creation fails', () => {
    fixture.componentRef.setInput('jobs', []);
    jobsStub.createJob.mockReturnValue(throwError(() => new Error('boom')));
    fixture.detectChanges();

    fixture.componentInstance['runTool']('transcribe');
    expect(fixture.componentInstance['runningToolType']()).toBeNull();
  });

  describe('tabs', () => {
    const panel = (id: string) => el().querySelector<HTMLElement>(`#panel-${id}`)!;

    it('shows the transcript tab first and hides the others', () => {
      fixture.componentRef.setInput('jobs', []);
      fixture.detectChanges();
      expect(panel('transcript').hidden).toBe(false);
      expect(panel('chapters').hidden).toBe(true);
      expect(el().querySelector('#tab-transcript')!.getAttribute('aria-selected')).toBe('true');
    });

    it('switches panels on click', () => {
      fixture.componentRef.setInput('jobs', []);
      fixture.detectChanges();
      el().querySelector<HTMLButtonElement>('#tab-shorts')!.click();
      fixture.detectChanges();
      expect(panel('shorts').hidden).toBe(false);
      expect(panel('transcript').hidden).toBe(true);
    });

    it('a blocked tool links to the tab that unblocks it', () => {
      fixture.componentRef.setInput('jobs', [makeJob({ status: 'succeeded' })]);
      fixture.detectChanges();
      el().querySelector<HTMLButtonElement>('#tab-shorts')!.click();
      fixture.detectChanges();

      const link = toolBar('shorts').querySelector<HTMLButtonElement>('.link-btn')!;
      expect(link.textContent!.trim()).toBe('Relevance →');
      link.click();
      fixture.detectChanges();
      expect(panel('relevance').hidden).toBe(false);
    });

    it('explains an empty tab with the tool’s one-liner', () => {
      fixture.componentRef.setInput('jobs', []);
      fixture.detectChanges();
      expect(panel('chapters').querySelector('.ws__empty')!.textContent).toContain('Needs a transcript');
    });

    it('opens the explanation popup from the ⓘ next to a tool', () => {
      fixture.componentRef.setInput('jobs', []);
      fixture.detectChanges();
      toolBar('relevance').querySelector<HTMLButtonElement>('.info-btn')!.click();
      expect(TestBed.inject(Explain).openKey()).toBe('relevance');
    });
  });

  describe('suggested next step', () => {
    it('suggests the transcript first and runs it from the stepper', () => {
      jobsStub.createJob.mockReturnValue(of(makeJob({ id: 7 })));
      fixture.componentRef.setInput('jobs', []);
      fixture.detectChanges();

      const card = el().querySelector('.stepper__next')!;
      expect(card.textContent).toContain('Transcript');
      card.querySelector<HTMLButtonElement>('.btn--primary')!.click();
      expect(jobsStub.createJob).toHaveBeenCalledWith(5, 'transcribe');
    });

    it('opens the Shorts tab instead of running shorts (they have options)', () => {
      relevanceStub.getRelevance.mockReturnValue(of({ version: 'v0', windows: [], top_moments: [] }));
      fixture.componentRef.setInput(
        'jobs',
        (['transcribe', 'diarize', 'chapters', 'relevance'] as const).map((t, i) =>
          makeJob({ id: i + 1, tool_type: t, status: 'succeeded' }),
        ),
      );
      fixture.detectChanges();

      el().querySelector<HTMLButtonElement>('.stepper__next .btn--primary')!.click();
      fixture.detectChanges();
      expect(jobsStub.createJob).not.toHaveBeenCalled();
      expect(el().querySelector<HTMLElement>('#panel-shorts')!.hidden).toBe(false);
    });

    it('stepper nodes open their tool’s tab', () => {
      fixture.componentRef.setInput('jobs', []);
      fixture.detectChanges();
      const node = [...el().querySelectorAll<HTMLButtonElement>('.stepper__node')].find((b) =>
        b.textContent!.includes('Show notes'),
      )!;
      node.click();
      fixture.detectChanges();
      expect(el().querySelector<HTMLElement>('#panel-shownotes')!.hidden).toBe(false);
    });
  });

  describe('subtitle downloads', () => {
    const downloadButtons = () => el().querySelectorAll<HTMLButtonElement>('.ws__download');

    it.each(['queued', 'running', 'failed'] as const)('are hidden when the only transcribe job is %s', (status) => {
      fixture.componentRef.setInput('jobs', [makeJob({ status })]);
      fixture.detectChanges();
      expect(downloadButtons().length).toBe(0);
    });

    it('are hidden when only another tool has succeeded', () => {
      fixture.componentRef.setInput('jobs', [makeJob({ tool_type: 'chapters', status: 'succeeded' })]);
      fixture.detectChanges();
      expect(downloadButtons().length).toBe(0);
    });

    it('show .srt and .vtt once a transcribe job has succeeded', () => {
      fixture.componentRef.setInput('jobs', [makeJob({ status: 'succeeded' })]);
      fixture.detectChanges();
      expect([...downloadButtons()].map((b) => b.textContent?.trim())).toEqual(['.srt', '.vtt']);
    });

    it('stay available while a re-run is in flight', () => {
      fixture.componentRef.setInput('jobs', [
        makeJob({ id: 2, status: 'running', created_at: '2026-08-15T12:00:00Z' }),
        makeJob({ id: 1, status: 'succeeded' }),
      ]);
      fixture.detectChanges();
      expect(downloadButtons().length).toBe(2);
    });

    it('fetch the subtitle and save it under the video file stem', () => {
      const blob = new Blob(['1\n']);
      transcriptsStub.getSubtitle.mockReturnValue(of(blob));
      const createUrl = vi.spyOn(URL, 'createObjectURL').mockReturnValue('blob:x');
      const revokeUrl = vi.spyOn(URL, 'revokeObjectURL').mockImplementation(() => {});
      const click = vi
        .spyOn(HTMLAnchorElement.prototype, 'click')
        .mockImplementation(function (this: HTMLAnchorElement) {
          expect(this.download).toBe('episode.srt');
          expect(this.href).toBe('blob:x');
        });

      fixture.componentRef.setInput('jobs', [makeJob({ status: 'succeeded' })]);
      fixture.detectChanges();
      downloadButtons()[0].click();

      expect(transcriptsStub.getSubtitle).toHaveBeenCalledWith(5, 'srt');
      expect(createUrl).toHaveBeenCalledWith(blob);
      expect(click).toHaveBeenCalledTimes(1);
      expect(revokeUrl).toHaveBeenCalledWith('blob:x');
      expect(fixture.componentInstance['downloadingFormat']()).toBeNull();

      createUrl.mockRestore();
      revokeUrl.mockRestore();
      click.mockRestore();
    });

    it('resets the downloading state if the request fails', () => {
      transcriptsStub.getSubtitle.mockReturnValue(throwError(() => new Error('409')));
      fixture.componentRef.setInput('jobs', [makeJob({ status: 'succeeded' })]);
      fixture.detectChanges();

      fixture.componentInstance['downloadSubtitle']('vtt');
      expect(fixture.componentInstance['downloadingFormat']()).toBeNull();
    });
  });

  describe('Speakers (diarize) run button', () => {
    it('is disabled with a hint until a transcript exists', () => {
      fixture.componentRef.setInput('jobs', []);
      fixture.detectChanges();
      expect(runButton('diarize')?.disabled).toBe(true);
      expect(runButton('diarize')?.getAttribute('title')).toBe('Needs a transcript');
    });

    it('stays disabled while transcription is still running', () => {
      fixture.componentRef.setInput('jobs', [makeJob({ status: 'running' })]);
      fixture.detectChanges();
      expect(runButton('diarize')?.disabled).toBe(true);
    });

    it('is enabled once a transcribe job has succeeded', () => {
      fixture.componentRef.setInput('jobs', [makeJob({ status: 'succeeded' })]);
      fixture.detectChanges();
      expect(runButton('diarize')?.disabled).toBe(false);
      expect(runButton('diarize')?.hasAttribute('title')).toBe(false);
    });

    it('does not gate the other tools', () => {
      fixture.componentRef.setInput('jobs', []);
      fixture.detectChanges();
      // Speakers, Chapters, Relevance, Show notes, Shorts (needs relevance),
      // CTR (needs candidates) and Hook (needs shorts)
      expect(el().querySelectorAll('.tool-bar__run:disabled').length).toBe(7);
    });
  });

  describe('CTR (M13)', () => {
    it('blocks Run until the panel reports candidates', () => {
      fixture.componentRef.setInput('jobs', []);
      fixture.detectChanges();
      expect(runButton('ctr')!.disabled).toBe(true);
      expect(runButton('ctr')!.getAttribute('title')).toBe('Add title/thumbnail candidates first');

      (fixture.componentInstance as unknown as { ctrCandidates: { set(n: number): void } }).ctrCandidates.set(2);
      fixture.detectChanges();
      expect(runButton('ctr')!.disabled).toBe(false);
    });

    it('always shows the candidates panel', () => {
      fixture.detectChanges();
      expect(el().querySelector('app-ctr-panel')).not.toBeNull();
    });
  });

  describe('Chapters (M5)', () => {
    const chapters: ChapterOut[] = [
      { order_index: 0, start_seconds: 0, title: 'Intro', model_version: 'bilstm-v1' },
      { order_index: 1, start_seconds: 605, title: 'Main topic', model_version: 'bilstm-v1' },
    ];

    it('Run is disabled until a transcript exists', () => {
      fixture.componentRef.setInput('jobs', []);
      fixture.detectChanges();
      expect(runButton('chapters')?.disabled).toBe(true);
      expect(runButton('chapters')?.getAttribute('title')).toBe('Needs a transcript');
    });

    it('does not fetch chapters until a chapters job has succeeded', () => {
      fixture.componentRef.setInput('jobs', [makeJob({ id: 3, tool_type: 'chapters', status: 'running' })]);
      fixture.detectChanges();
      expect(chaptersStub.getChapters).not.toHaveBeenCalled();
      expect(el().querySelector('.ws__chapters')).toBeNull();
    });

    it('fetches once per succeeded job and lists the chapters with their model version', () => {
      chaptersStub.getChapters.mockReturnValue(of(chapters));
      const done = makeJob({ id: 3, tool_type: 'chapters', status: 'succeeded' });
      fixture.componentRef.setInput('jobs', [done]);
      fixture.detectChanges();
      // The parent's 3 s poll hands in an equal-but-new array: no refetch.
      fixture.componentRef.setInput('jobs', [{ ...done }]);
      fixture.detectChanges();

      expect(chaptersStub.getChapters).toHaveBeenCalledTimes(1);
      expect(chaptersStub.getChapters).toHaveBeenCalledWith(5);
      const items = [...el().querySelectorAll('.ws__chapter-list li')].map((li) =>
        li.textContent?.replace(/\s+/g, ' ').trim(),
      );
      expect(items).toEqual(['00:00 Intro', '10:05 Main topic']);
      expect(el().querySelector('.ws__chapters .model-chip')!.textContent).toContain('bilstm-v1');
    });

    it('refetches when a newer chapters job succeeds', () => {
      chaptersStub.getChapters.mockReturnValue(of(chapters));
      fixture.componentRef.setInput('jobs', [makeJob({ id: 3, tool_type: 'chapters', status: 'succeeded' })]);
      fixture.detectChanges();
      fixture.componentRef.setInput('jobs', [
        makeJob({ id: 4, tool_type: 'chapters', status: 'succeeded', created_at: '2026-08-15T12:00:00Z' }),
        makeJob({ id: 3, tool_type: 'chapters', status: 'succeeded' }),
      ]);
      fixture.detectChanges();
      expect(chaptersStub.getChapters).toHaveBeenCalledTimes(2);
    });

    it('copies the YouTube description block', async () => {
      const writeText = vi.fn().mockResolvedValue(undefined);
      Object.defineProperty(navigator, 'clipboard', { value: { writeText }, configurable: true });
      chaptersStub.getChapters.mockReturnValue(of(chapters));
      fixture.componentRef.setInput('jobs', [makeJob({ id: 3, tool_type: 'chapters', status: 'succeeded' })]);
      fixture.detectChanges();

      el().querySelector<HTMLButtonElement>('.ws__copy')!.click();
      await Promise.resolve();
      fixture.detectChanges();

      expect(writeText).toHaveBeenCalledWith('00:00 Intro\n10:05 Main topic');
      expect(el().querySelector('.ws__copy')?.textContent?.trim()).toBe('Copied!');
    });

    it('shows nothing if the fetch fails', () => {
      chaptersStub.getChapters.mockReturnValue(throwError(() => new Error('boom')));
      fixture.componentRef.setInput('jobs', [makeJob({ id: 3, tool_type: 'chapters', status: 'succeeded' })]);
      fixture.detectChanges();
      expect(el().querySelector('.ws__chapters')).toBeNull();
    });
  });

  describe('relevance (M7)', () => {
    const relevance: RelevanceOut = {
      version: 'v0',
      windows: [{ start: 0, end: 15, score: 0.5 }],
      top_moments: [{ start: 42, end: 57, score: 0.9, text: 'Great bit' }],
    };

    it('Run is disabled until a transcript exists', () => {
      fixture.componentRef.setInput('jobs', []);
      fixture.detectChanges();
      expect(runButton('relevance')?.disabled).toBe(true);
    });

    it('fetches the scores once the job succeeds and shows the panel', () => {
      relevanceStub.getRelevance.mockReturnValue(of(relevance));
      fixture.componentRef.setInput('jobs', [makeJob({ id: 3, tool_type: 'relevance', status: 'running' })]);
      fixture.detectChanges();
      expect(relevanceStub.getRelevance).not.toHaveBeenCalled();

      fixture.componentRef.setInput('jobs', [makeJob({ id: 3, tool_type: 'relevance', status: 'succeeded' })]);
      fixture.detectChanges();
      expect(relevanceStub.getRelevance).toHaveBeenCalledWith(5);
      expect(el().querySelector('app-relevance-panel')).not.toBeNull();
    });

    it('clicking a moment seeks the video there', () => {
      relevanceStub.getRelevance.mockReturnValue(of(relevance));
      fixture.componentRef.setInput('jobs', [makeJob({ id: 3, tool_type: 'relevance', status: 'succeeded' })]);
      fixture.detectChanges();
      const video = el().querySelector<HTMLVideoElement>('.ws__video')!;
      const play = vi.spyOn(video, 'play').mockResolvedValue(undefined);

      el().querySelector<HTMLButtonElement>('.relevance__moment')!.click();

      expect(video.currentTime).toBe(42);
      expect(play).toHaveBeenCalled();
    });
  });

  describe('shorts (M8)', () => {
    const clips: ClipOut[] = [
      {
        id: 21,
        rank: 1,
        start: 120,
        end: 151,
        duration: 31,
        score: 0.8,
        aspect_ratio: '1:1',
        caption_burned_in: false,
        framing: { shots: [{ start: 0, end: 31, mode: 'fit' }] },
        hook: null,
      },
    ];
    const relevanceDone = makeJob({ id: 4, tool_type: 'relevance', status: 'succeeded' });

    beforeEach(() => {
      relevanceStub.getRelevance.mockReturnValue(of({ version: 'v0', windows: [], top_moments: [] }));
    });

    it('Run is disabled until relevance scores exist', () => {
      fixture.componentRef.setInput('jobs', [makeJob({ status: 'succeeded' })]);
      fixture.detectChanges();
      expect(runButton('shorts')!.disabled).toBe(true);
      expect(runButton('shorts')!.getAttribute('title')).toBe('Needs relevance scores');

      fixture.componentRef.setInput('jobs', [relevanceDone]);
      fixture.detectChanges();
      expect(runButton('shorts')!.disabled).toBe(false);
    });

    it('sends the chosen aspect ratio and captions toggle', () => {
      jobsStub.createJob.mockReturnValue(of(makeJob({ id: 8, tool_type: 'shorts' })));
      fixture.componentRef.setInput('jobs', [relevanceDone]);
      fixture.detectChanges();

      const select = toolBar('shorts').querySelector('select')!;
      expect([...select.options].map((o) => o.value)).toEqual(['9:16', '1:1', '4:5', '16:9']);
      select.value = '4:5';
      select.dispatchEvent(new Event('change'));
      const box = toolBar('shorts').querySelector<HTMLInputElement>('input[type=checkbox]')!;
      expect(box.checked).toBe(true);
      box.checked = false;
      box.dispatchEvent(new Event('change'));
      runButton('shorts')!.click();

      expect(jobsStub.createJob).toHaveBeenCalledWith(5, 'shorts', { aspect_ratio: '4:5', captions: false });
    });

    it('defaults to 9:16 with captions', () => {
      jobsStub.createJob.mockReturnValue(of(makeJob({ id: 8, tool_type: 'shorts' })));
      fixture.componentRef.setInput('jobs', [relevanceDone]);
      fixture.detectChanges();
      runButton('shorts')!.click();
      expect(jobsStub.createJob).toHaveBeenCalledWith(5, 'shorts', { aspect_ratio: '9:16', captions: true });
    });

    it('fetches the clips once the job succeeds, shows the gallery and seeks the episode', () => {
      shortsStub.getClips.mockReturnValue(of(clips));
      fixture.componentRef.setInput('jobs', [relevanceDone, makeJob({ id: 9, tool_type: 'shorts', status: 'running' })]);
      fixture.detectChanges();
      expect(shortsStub.getClips).not.toHaveBeenCalled();

      fixture.componentRef.setInput('jobs', [relevanceDone, makeJob({ id: 9, tool_type: 'shorts', status: 'succeeded' })]);
      fixture.detectChanges();
      expect(shortsStub.getClips).toHaveBeenCalledWith(5);
      expect(el().querySelector('.shorts video')!.getAttribute('src')).toBe('/clips/21/file?token=fake-token');

      const player = el().querySelector<HTMLVideoElement>('.ws__video')!;
      vi.spyOn(player, 'play').mockResolvedValue(undefined);
      el().querySelector<HTMLButtonElement>('.shorts__seek')!.click();
      expect(player.currentTime).toBe(120);
    });

    describe('hook score (M14)', () => {
      const shortsDone = makeJob({ id: 9, tool_type: 'shorts', status: 'succeeded' });

      it('blocks Run until the episode has shorts', () => {
        fixture.componentRef.setInput('jobs', [relevanceDone]);
        fixture.detectChanges();
        expect(runButton('hook')!.disabled).toBe(true);
        expect(runButton('hook')!.getAttribute('title')).toBe('Needs shorts');

        shortsStub.getClips.mockReturnValue(of(clips));
        fixture.componentRef.setInput('jobs', [relevanceDone, shortsDone]);
        fixture.detectChanges();
        expect(runButton('hook')!.disabled).toBe(false);
      });

      it('refetches the clips when a newer hook job succeeds', () => {
        shortsStub.getClips.mockReturnValue(of(clips));
        fixture.componentRef.setInput('jobs', [relevanceDone, shortsDone]);
        fixture.detectChanges();
        expect(shortsStub.getClips).toHaveBeenCalledTimes(1);

        const scored: ClipOut[] = [
          { ...clips[0], hook: { score: 81, rank: 1, scored_at: '2026-11-08T12:00:00Z', features: {} } },
        ];
        shortsStub.getClips.mockReturnValue(of(scored));
        fixture.componentRef.setInput('jobs', [relevanceDone, shortsDone, makeJob({ id: 12, tool_type: 'hook', status: 'running' })]);
        fixture.detectChanges();
        expect(shortsStub.getClips).toHaveBeenCalledTimes(1);

        fixture.componentRef.setInput('jobs', [relevanceDone, shortsDone, makeJob({ id: 12, tool_type: 'hook', status: 'succeeded' })]);
        fixture.detectChanges();
        expect(shortsStub.getClips).toHaveBeenCalledTimes(2);
        expect(el().querySelector('.shorts__hook-badge')!.textContent!.replace(/\s+/g, ' ').trim()).toBe('Hook 81 · #1');
      });
    });
  });

  describe('show notes (M6)', () => {
    const notes: ShowNotesOut = {
      title_suggestions: ['A', 'B', 'C'],
      description: 'Desc.',
      tags: ['x'],
      summary: 'Sum.',
      language: 'en',
      llm_model_used: 'm+shownotes-v1',
      created_at: '2026-10-16T12:00:00Z',
      updated_at: '2026-10-16T12:00:00Z',
    };

    afterEach(() => vi.restoreAllMocks());

    it('Run is disabled until a transcript exists', () => {
      fixture.componentRef.setInput('jobs', []);
      fixture.detectChanges();
      expect(runButton('shownotes')?.disabled).toBe(true);
      expect(runButton('shownotes')?.getAttribute('title')).toBe('Needs a transcript');
    });

    it('fetches the draft once its job succeeds and shows the panel', () => {
      showNotesStub.getShowNotes.mockReturnValue(of(notes));
      fixture.componentRef.setInput('jobs', [makeJob({ id: 3, tool_type: 'shownotes', status: 'running' })]);
      fixture.detectChanges();
      expect(el().querySelector('app-show-notes-panel')).toBeNull();

      fixture.componentRef.setInput('jobs', [makeJob({ id: 3, tool_type: 'shownotes', status: 'succeeded' })]);
      fixture.detectChanges();

      expect(showNotesStub.getShowNotes).toHaveBeenCalledWith(5);
      expect(el().querySelector('app-show-notes-panel')).not.toBeNull();
    });

    it('runs without asking when there is no draft yet', () => {
      const confirm = vi.spyOn(window, 'confirm');
      jobsStub.createJob.mockReturnValue(of(makeJob({ id: 2, tool_type: 'shownotes' })));
      fixture.componentRef.setInput('jobs', [makeJob({ status: 'succeeded' })]);
      fixture.detectChanges();

      runButton('shownotes')!.click();

      expect(confirm).not.toHaveBeenCalled();
      expect(jobsStub.createJob).toHaveBeenCalledWith(5, 'shownotes');
    });

    it.each([
      [false, 0],
      [true, 1],
    ])('asks before replacing an existing draft (confirmed: %s)', (answer, calls) => {
      const confirm = vi.spyOn(window, 'confirm').mockReturnValue(answer);
      showNotesStub.getShowNotes.mockReturnValue(of(notes));
      jobsStub.createJob.mockReturnValue(of(makeJob({ id: 9, tool_type: 'shownotes' })));
      fixture.componentRef.setInput('jobs', [
        makeJob({ status: 'succeeded' }),
        makeJob({ id: 3, tool_type: 'shownotes', status: 'succeeded' }),
      ]);
      fixture.detectChanges();

      runButton('shownotes')!.click();

      expect(confirm).toHaveBeenCalledTimes(1);
      expect(jobsStub.createJob).toHaveBeenCalledTimes(calls);
    });
  });

  it('passes the job error message to its status badge', () => {
    fixture.componentRef.setInput('jobs', [
      makeJob({ tool_type: 'diarize', status: 'failed', error_message: 'Diarization unavailable' }),
      makeJob({ id: 2, status: 'succeeded' }),
    ]);
    fixture.detectChanges();

    expect(toolBar('diarize').querySelector('.badge--failed')?.getAttribute('title')).toBe('Diarization unavailable');
  });
});
