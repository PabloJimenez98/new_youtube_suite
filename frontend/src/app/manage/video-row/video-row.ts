import { Component, ElementRef, computed, effect, inject, input, output, signal, untracked, viewChild } from '@angular/core';

import { Auth } from '../../auth/auth';
import { ChapterOut, Chapters, formatTimestamp, toYouTubeText } from '../../chapters/chapters';
import { JobOut, TOOL_LABELS, TOOL_TYPES, ToolType } from '../../jobs/job.model';
import { Jobs } from '../../jobs/jobs';
import { UploadOut } from '../../uploads/upload.model';
import { Relevance, RelevanceOut } from '../../relevance/relevance';
import { JobStatusBadge } from '../../shared/job-status-badge/job-status-badge';
import { ShowNotes, ShowNotesOut } from '../../shownotes/shownotes';
import { RelevancePanel } from '../relevance-panel/relevance-panel';
import { ShowNotesPanel } from '../show-notes-panel/show-notes-panel';
import {
  SUBTITLE_FORMATS,
  SubtitleFormat,
  Transcripts,
  saveBlob,
} from '../../transcripts/transcripts';

@Component({
  selector: 'app-video-row',
  imports: [JobStatusBadge, RelevancePanel, ShowNotesPanel],
  templateUrl: './video-row.html',
  styleUrl: './video-row.scss',
})
export class VideoRow {
  private readonly jobsService = inject(Jobs);
  private readonly auth = inject(Auth);
  private readonly transcripts = inject(Transcripts);
  private readonly chaptersService = inject(Chapters);
  private readonly showNotesService = inject(ShowNotes);
  private readonly relevanceService = inject(Relevance);
  private readonly player = viewChild<ElementRef<HTMLVideoElement>>('player');

  readonly upload = input.required<UploadOut>();
  /** This video's jobs, as last fetched by the parent's batched call. */
  readonly jobs = input<readonly JobOut[]>([]);
  readonly deleteRequested = output<UploadOut>();

  protected readonly toolTypes = TOOL_TYPES;
  protected readonly toolLabels = TOOL_LABELS;
  protected readonly runningToolType = signal<ToolType | null>(null);
  protected readonly subtitleFormats = SUBTITLE_FORMATS;
  protected readonly downloadingFormat = signal<SubtitleFormat | null>(null);
  protected readonly chapters = signal<ChapterOut[] | null>(null);
  protected readonly copied = signal(false);
  protected readonly showNotes = signal<ShowNotesOut | null>(null);
  protected readonly relevance = signal<RelevanceOut | null>(null);
  protected readonly chaptersWithHours = computed(() =>
    (this.chapters() ?? []).some((c) => c.start_seconds >= 3600),
  );
  protected readonly formatTimestamp = formatTimestamp;

  /**
   * <video src> requests come straight from the browser, bypassing
   * Angular's HttpClient (and therefore authInterceptor) entirely --
   * no way to attach an Authorization header. The token travels as a
   * query param instead, only for this endpoint. See
   * docs/decisions/0005-video-preview-streaming.md.
   */
  protected readonly videoUrl = computed(
    () => `/uploads/${this.upload().id}/file?token=${this.auth.token() ?? ''}`,
  );

  // Jobs created via this row's own "Run" clicks, merged with whatever
  // the parent last fetched -- the parent doesn't refetch after every
  // click, so this keeps the UI current without an extra round trip.
  private readonly optimisticJobs = signal<JobOut[]>([]);
  private readonly allJobs = computed(() => [...this.optimisticJobs(), ...this.jobs()]);

  /** Newest succeeded chapters job -- a change (first success, or a re-run) triggers a (re)fetch. */
  private readonly succeededChaptersJobId = computed(() => this.newestSucceededJobId('chapters'));
  /** Same for show notes (M6). */
  private readonly succeededShowNotesJobId = computed(() => this.newestSucceededJobId('shownotes'));
  /** Same for relevance (M7). */
  private readonly succeededRelevanceJobId = computed(() => this.newestSucceededJobId('relevance'));

  constructor() {
    // The parent re-polls jobs every 3 s; only fetch chapters when the
    // succeeded job actually changes, not on every poll.
    effect(() => {
      const jobId = this.succeededChaptersJobId();
      if (jobId === null) return;
      untracked(() =>
        this.chaptersService.getChapters(this.upload().id).subscribe({
          next: (chapters) => this.chapters.set(chapters),
          error: () => this.chapters.set(null),
        }),
      );
    });
    effect(() => {
      const jobId = this.succeededShowNotesJobId();
      if (jobId === null) return;
      untracked(() =>
        this.showNotesService.getShowNotes(this.upload().id).subscribe({
          next: (notes) => this.showNotes.set(notes),
          error: () => this.showNotes.set(null),
        }),
      );
    });
    effect(() => {
      const jobId = this.succeededRelevanceJobId();
      if (jobId === null) return;
      untracked(() =>
        this.relevanceService.getRelevance(this.upload().id).subscribe({
          next: (relevance) => this.relevance.set(relevance),
          error: () => this.relevance.set(null),
        }),
      );
    });
  }

  /** Jump the preview to a relevance moment (docs/decisions/0012 §7). */
  protected seekTo(seconds: number): void {
    const video = this.player()?.nativeElement;
    if (!video) return;
    video.currentTime = seconds;
    video.play().catch(() => undefined);
  }

  private newestSucceededJobId(tool: ToolType): number | null {
    return (
      this.allJobs()
        .filter((job) => job.tool_type === tool && job.status === 'succeeded')
        .sort((a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime())[0]?.id ??
      null
    );
  }

  protected latestJobFor(tool: ToolType): JobOut | undefined {
    return this.allJobs()
      .filter((job) => job.tool_type === tool)
      .sort((a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime())[0];
  }

  /**
   * "Run" is available with no job yet, and again once the latest job
   * has reached a terminal state -- hidden only while queued/running.
   * Retry policy (backoff, auto-retry) is explicitly M9's territory;
   * this is just letting the operator ask for another run by hand.
   */
  protected canRun(tool: ToolType): boolean {
    const job = this.latestJobFor(tool);
    return !job || job.status === 'succeeded' || job.status === 'failed';
  }

  protected runTool(tool: ToolType): void {
    // A show-notes re-run replaces the draft, edits included (docs/decisions/0011 §6).
    if (
      tool === 'shownotes' &&
      this.showNotes() !== null &&
      !window.confirm('Run show notes again? This replaces the current draft, including your edits.')
    ) {
      return;
    }
    this.runningToolType.set(tool);
    this.jobsService.createJob(this.upload().id, tool).subscribe({
      next: (job) => {
        this.optimisticJobs.update((jobs) => [job, ...jobs]);
        this.runningToolType.set(null);
      },
      error: () => {
        this.runningToolType.set(null);
      },
    });
  }

  /**
   * Subtitles are downloadable once any transcribe job has succeeded --
   * the backend keeps serving that transcript during a re-run, so this
   * isn't tied to the *latest* job (docs/decisions/0007 §5).
   */
  protected hasTranscript(): boolean {
    return this.allJobs().some(
      (job) => job.tool_type === 'transcribe' && job.status === 'succeeded',
    );
  }

  /** Diarization (M4), chapters (M5), relevance (M7) and show notes (M6) work on an existing transcript. */
  protected needsTranscript(tool: ToolType): boolean {
    return (
      (tool === 'diarize' || tool === 'chapters' || tool === 'relevance' || tool === 'shownotes') &&
      !this.hasTranscript()
    );
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
      error: () => {
        this.downloadingFormat.set(null);
      },
    });
  }
}
