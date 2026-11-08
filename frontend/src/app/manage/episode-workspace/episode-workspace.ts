import { NgTemplateOutlet } from '@angular/common';
import { Component, ElementRef, computed, effect, inject, input, output, signal, untracked, viewChild } from '@angular/core';

import { Auth } from '../../auth/auth';
import { ChapterOut, Chapters, formatTimestamp, toYouTubeText } from '../../chapters/chapters';
import { ToolState, canRun, newestSucceededJobId, nextSuggested, toolStates } from '../../explain/pipeline';
import { TOOL_DOCS, ToolDoc } from '../../explain/tool-docs';
import { ASPECT_RATIOS, AspectRatio, JobOut, TOOL_LABELS, ToolType } from '../../jobs/job.model';
import { Jobs } from '../../jobs/jobs';
import { Relevance, RelevanceOut } from '../../relevance/relevance';
import { InfoButton } from '../../shared/info-button/info-button';
import { JobStatusBadge } from '../../shared/job-status-badge/job-status-badge';
import { ClipOut, Shorts } from '../../shorts/shorts';
import { ShowNotes, ShowNotesOut } from '../../shownotes/shownotes';
import { SUBTITLE_FORMATS, SubtitleFormat, Transcripts, saveBlob } from '../../transcripts/transcripts';
import { UploadOut } from '../../uploads/upload.model';
import { CtrPanel } from '../ctr-panel/ctr-panel';
import { HookSummary } from '../hook-summary/hook-summary';
import { PipelineStepper } from '../pipeline-stepper/pipeline-stepper';
import { RelevancePanel } from '../relevance-panel/relevance-panel';
import { ShortsPanel } from '../shorts-panel/shorts-panel';
import { ShowNotesPanel } from '../show-notes-panel/show-notes-panel';
import { TranscriptViewer } from '../transcript-viewer/transcript-viewer';

export type TabId = 'transcript' | 'chapters' | 'relevance' | 'shorts' | 'shownotes' | 'ctr';

export interface WorkspaceTab {
  id: TabId;
  label: string;
  tools: ToolType[];
}

/** Six tabs for eight tools: a tool lives with the results it produces. */
export const TABS: readonly WorkspaceTab[] = [
  { id: 'transcript', label: 'Transcript & speakers', tools: ['transcribe', 'diarize'] },
  { id: 'chapters', label: 'Chapters', tools: ['chapters'] },
  { id: 'relevance', label: 'Relevance', tools: ['relevance'] },
  { id: 'shorts', label: 'Shorts & hooks', tools: ['shorts', 'hook'] },
  { id: 'shownotes', label: 'Show notes', tools: ['shownotes'] },
  { id: 'ctr', label: 'Titles & thumbnails', tools: ['ctr'] },
];

export function tabOf(tool: ToolType): TabId {
  return TABS.find((t) => t.tools.includes(tool))!.id;
}

/**
 * One episode's tools (docs/decisions/0021): the preview, a pipeline
 * stepper suggesting the next step, and one tab per result type. All
 * tabs stay mounted (hidden when inactive) so unsaved edits and the CTR
 * candidate count survive a tab switch.
 */
@Component({
  selector: 'app-episode-workspace',
  imports: [
    CtrPanel,
    HookSummary,
    InfoButton,
    JobStatusBadge,
    NgTemplateOutlet,
    PipelineStepper,
    RelevancePanel,
    ShortsPanel,
    ShowNotesPanel,
    TranscriptViewer,
  ],
  templateUrl: './episode-workspace.html',
  styleUrl: './episode-workspace.scss',
})
export class EpisodeWorkspace {
  private readonly jobsService = inject(Jobs);
  private readonly auth = inject(Auth);
  private readonly transcripts = inject(Transcripts);
  private readonly chaptersService = inject(Chapters);
  private readonly showNotesService = inject(ShowNotes);
  private readonly relevanceService = inject(Relevance);
  private readonly shortsService = inject(Shorts);
  private readonly player = viewChild<ElementRef<HTMLVideoElement>>('player');

  readonly upload = input.required<UploadOut>();
  /** This episode's jobs, as last fetched by the parent's poll. */
  readonly jobs = input<readonly JobOut[]>([]);
  readonly deleteRequested = output<UploadOut>();

  protected readonly tabs = TABS;
  protected readonly toolLabels = TOOL_LABELS;
  protected readonly docs = TOOL_DOCS;
  protected readonly activeTab = signal<TabId>('transcript');
  protected readonly runningToolType = signal<ToolType | null>(null);
  protected readonly subtitleFormats = SUBTITLE_FORMATS;
  protected readonly downloadingFormat = signal<SubtitleFormat | null>(null);
  protected readonly chapters = signal<ChapterOut[] | null>(null);
  protected readonly copied = signal(false);
  protected readonly showNotes = signal<ShowNotesOut | null>(null);
  protected readonly relevance = signal<RelevanceOut | null>(null);
  protected readonly clips = signal<ClipOut[] | null>(null);
  /** How many CTR candidates the panel holds (M13): Run is blocked at zero. */
  protected readonly ctrCandidates = signal(0);
  /** Options for the next shorts run (docs/decisions/0013 §8). */
  protected readonly aspectRatios = ASPECT_RATIOS;
  protected readonly shortsAspect = signal<AspectRatio>('9:16');
  protected readonly shortsCaptions = signal(true);
  protected readonly token = computed(() => this.auth.token() ?? '');
  protected readonly chaptersWithHours = computed(() => (this.chapters() ?? []).some((c) => c.start_seconds >= 3600));
  protected readonly formatTimestamp = formatTimestamp;

  /**
   * <video src> requests come straight from the browser, bypassing
   * Angular's HttpClient (and therefore authInterceptor) entirely -- the
   * token travels as a query param instead, only for this endpoint. See
   * docs/decisions/0005-video-preview-streaming.md.
   */
  protected readonly videoUrl = computed(() => `/uploads/${this.upload().id}/file?token=${this.auth.token() ?? ''}`);

  // Jobs created via this page's own Run clicks, merged with whatever the
  // parent last fetched, so the UI is current before the next poll.
  private readonly optimisticJobs = signal<JobOut[]>([]);
  private readonly allJobs = computed(() => {
    const fetched = this.jobs();
    const fetchedIds = new Set(fetched.map((j) => j.id));
    return [...this.optimisticJobs().filter((j) => !fetchedIds.has(j.id)), ...fetched];
  });

  protected readonly states = computed(() =>
    toolStates(this.allJobs(), { ctrCandidates: this.ctrCandidates(), clipCount: this.clips()?.length ?? 0 }),
  );
  protected readonly suggestion = computed(() => nextSuggested(this.states()));

  private readonly succeededChaptersJobId = computed(() => newestSucceededJobId(this.allJobs(), 'chapters'));
  private readonly succeededShowNotesJobId = computed(() => newestSucceededJobId(this.allJobs(), 'shownotes'));
  private readonly succeededRelevanceJobId = computed(() => newestSucceededJobId(this.allJobs(), 'relevance'));
  private readonly succeededShortsJobId = computed(() => newestSucceededJobId(this.allJobs(), 'shorts'));
  /** Hook scores (M14) ride on the clips, so a newer one refetches the clips. */
  private readonly succeededHookJobId = computed(() => newestSucceededJobId(this.allJobs(), 'hook'));
  /** CTR (M13): the panel refetches the ranking. */
  protected readonly succeededCtrJobId = computed(() => newestSucceededJobId(this.allJobs(), 'ctr'));
  /** Refetch key for the transcript viewer: a new transcript or new speaker labels. */
  protected readonly transcriptVersion = computed(
    () => `${newestSucceededJobId(this.allJobs(), 'transcribe')}-${newestSucceededJobId(this.allJobs(), 'diarize')}`,
  );

  constructor() {
    // The parent re-polls jobs every 3 s; only fetch results when the
    // succeeded job actually changes, not on every poll.
    effect(() => {
      if (this.succeededChaptersJobId() === null) return;
      untracked(() =>
        this.chaptersService.getChapters(this.upload().id).subscribe({
          next: (chapters) => this.chapters.set(chapters),
          error: () => this.chapters.set(null),
        }),
      );
    });
    effect(() => {
      if (this.succeededShowNotesJobId() === null) return;
      untracked(() =>
        this.showNotesService.getShowNotes(this.upload().id).subscribe({
          next: (notes) => this.showNotes.set(notes),
          error: () => this.showNotes.set(null),
        }),
      );
    });
    effect(() => {
      if (this.succeededRelevanceJobId() === null) return;
      untracked(() =>
        this.relevanceService.getRelevance(this.upload().id).subscribe({
          next: (relevance) => this.relevance.set(relevance),
          error: () => this.relevance.set(null),
        }),
      );
    });
    effect(() => {
      const jobId = this.succeededShortsJobId();
      this.succeededHookJobId(); // tracked: a newer hook run refetches too
      if (jobId === null) return;
      untracked(() =>
        this.shortsService.getClips(this.upload().id).subscribe({
          next: (clips) => this.clips.set(clips),
          error: () => this.clips.set(null),
        }),
      );
    });
  }

  // Typed lookups for the ng-template tool bars, whose context variable is untyped.
  protected stateOf(tool: ToolType): ToolState {
    return this.states()[tool];
  }

  protected labelOf(tool: ToolType): string {
    return TOOL_LABELS[tool];
  }

  protected docOf(tool: ToolType): ToolDoc {
    return TOOL_DOCS[tool];
  }

  protected selectTab(tab: TabId): void {
    this.activeTab.set(tab);
  }

  protected openToolTab(tool: ToolType): void {
    this.activeTab.set(tabOf(tool));
  }

  /** Worst status among a tab's tools, for the dot on the tab. */
  protected tabStatus(tab: WorkspaceTab): string {
    const order = ['failed', 'running', 'queued', 'ready', 'blocked', 'done'];
    const statuses = tab.tools.map((t) => this.states()[t].status as string);
    return order.find((s) => statuses.includes(s)) ?? 'done';
  }

  /** Jump the preview to a relevance moment or a short's source (docs/decisions/0012 §7, 0013 §8). */
  protected seekTo(seconds: number): void {
    const video = this.player()?.nativeElement;
    if (!video) return;
    video.currentTime = seconds;
    video.play().catch(() => undefined);
  }

  protected canRun(tool: ToolType): boolean {
    return canRun(this.states()[tool]);
  }

  protected runTool(tool: ToolType): void {
    if (!this.canRun(tool)) return;
    // A show-notes re-run replaces the draft, edits included (docs/decisions/0011 §6).
    if (
      tool === 'shownotes' &&
      this.showNotes() !== null &&
      !window.confirm('Run show notes again? This replaces the current draft, including your edits.')
    ) {
      return;
    }
    this.runningToolType.set(tool);
    const request =
      tool === 'shorts'
        ? this.jobsService.createJob(this.upload().id, tool, {
            aspect_ratio: this.shortsAspect(),
            captions: this.shortsCaptions(),
          })
        : this.jobsService.createJob(this.upload().id, tool);
    request.subscribe({
      next: (job) => {
        this.optimisticJobs.update((jobs) => [job, ...jobs]);
        this.runningToolType.set(null);
      },
      error: () => this.runningToolType.set(null),
    });
  }

  /** The stepper's main action: shorts opens its tab first (it has options); the rest run. */
  protected onSuggestedAction(tool: ToolType): void {
    if (tool === 'shorts' || this.states()[tool].blockedReason) {
      this.openToolTab(tool);
      return;
    }
    this.runTool(tool);
  }

  /** Subtitles stay downloadable during a re-run (docs/decisions/0007 §5). */
  protected hasTranscript(): boolean {
    return this.states().transcribe.hasResult;
  }

  protected setShortsAspect(value: string): void {
    if ((ASPECT_RATIOS as readonly string[]).includes(value)) this.shortsAspect.set(value as AspectRatio);
  }

  /** Copy the chapters in YouTube-description format (docs/decisions/0009 §7). */
  protected copyChapters(): void {
    const chapters = this.chapters();
    if (!chapters?.length) return;
    navigator.clipboard.writeText(toYouTubeText(chapters)).then(() => {
      this.copied.set(true);
      setTimeout(() => this.copied.set(false), 2000);
    });
  }

  protected downloadSubtitle(format: SubtitleFormat): void {
    this.downloadingFormat.set(format);
    const stem = this.upload().filename.replace(/\.[^.]+$/, '') || 'transcript';
    this.transcripts.getSubtitle(this.upload().id, format).subscribe({
      next: (blob) => {
        saveBlob(blob, `${stem}.${format}`);
        this.downloadingFormat.set(null);
      },
      error: () => this.downloadingFormat.set(null),
    });
  }
}
