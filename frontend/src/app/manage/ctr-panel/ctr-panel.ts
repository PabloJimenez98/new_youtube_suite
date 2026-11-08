import {
  Component,
  computed,
  effect,
  inject,
  input,
  output,
  signal,
  untracked,
} from '@angular/core';

import {
  CandidateOut,
  Ctr,
  MAX_CANDIDATES,
  MAX_THUMBNAIL_BYTES,
  MAX_TITLE_CHARS,
  SUB_SCORE_LABELS,
  thumbnailUrl,
} from '../../ctr/ctr';

/**
 * Title/thumbnail candidates for one episode and their heuristic v0
 * ranking (M13, docs/decisions/0019 §5).
 *
 * - The row's existing CTR "Run" button scores them; this panel refetches
 *   whenever a newer `ctr` job succeeds (`scoredJobId`).
 * - The score is relative between this episode's candidates, not a CTR %.
 */
@Component({
  selector: 'app-ctr-panel',
  imports: [],
  templateUrl: './ctr-panel.html',
  styleUrl: './ctr-panel.scss',
})
export class CtrPanel {
  private readonly ctr = inject(Ctr);

  readonly uploadId = input.required<number>();
  readonly token = input.required<string>();
  /** Newest succeeded `ctr` job id; a change triggers a refetch. */
  readonly scoredJobId = input<number | null>(null);
  readonly hasShowNotes = input(false);
  /** How many candidates exist, so the row can block Run on zero. */
  readonly candidateCount = output<number>();

  protected readonly candidates = signal<CandidateOut[]>([]);
  protected readonly loadError = signal(false);
  protected readonly error = signal<string | null>(null);
  protected readonly busy = signal(false);
  protected readonly newTitle = signal('');
  protected readonly newThumbnail = signal<File | null>(null);
  protected readonly editing = signal<number | null>(null);
  protected readonly editTitle = signal('');
  protected readonly expanded = signal<ReadonlySet<number>>(new Set());

  protected readonly maxTitle = MAX_TITLE_CHARS;
  protected readonly maxCandidates = MAX_CANDIDATES;
  protected readonly subScoreLabels = SUB_SCORE_LABELS;
  protected readonly full = computed(() => this.candidates().length >= MAX_CANDIDATES);
  protected readonly canAdd = computed(() => {
    const t = this.newTitle().trim();
    return t.length > 0 && t.length <= MAX_TITLE_CHARS && !this.full() && !this.busy();
  });
  protected readonly anyStale = computed(() =>
    this.candidates().some((c) => c.group === 'unscored' && !c.features?.error),
  );

  constructor() {
    effect(() => {
      this.uploadId();
      this.scoredJobId();
      untracked(() => this.refresh());
    });
  }

  protected refresh(): void {
    this.ctr.list(this.uploadId()).subscribe({
      next: (list) => this.setList(list),
      error: () => this.loadError.set(true),
    });
  }

  private setList(list: CandidateOut[]): void {
    this.loadError.set(false);
    this.candidates.set(list);
    this.candidateCount.emit(list.length);
  }

  private fail(err: unknown, fallback: string): void {
    this.busy.set(false);
    const detail = (err as { error?: { detail?: unknown } })?.error?.detail;
    this.error.set(typeof detail === 'string' ? detail : fallback);
  }

  protected pickThumbnail(event: Event): void {
    const file = (event.target as HTMLInputElement).files?.[0] ?? null;
    if (file && file.size > MAX_THUMBNAIL_BYTES) {
      this.error.set('Thumbnails can be at most 2 MB (YouTube’s limit).');
      this.newThumbnail.set(null);
      return;
    }
    this.error.set(null);
    this.newThumbnail.set(file);
  }

  protected add(): void {
    if (!this.canAdd()) return;
    this.busy.set(true);
    this.error.set(null);
    this.ctr.create(this.uploadId(), this.newTitle().trim(), this.newThumbnail()).subscribe({
      next: () => {
        this.busy.set(false);
        this.newTitle.set('');
        this.newThumbnail.set(null);
        this.refresh();
      },
      error: (err) => this.fail(err, 'Adding the candidate failed.'),
    });
  }

  protected addFromShowNotes(): void {
    this.busy.set(true);
    this.error.set(null);
    this.ctr.fromShowNotes(this.uploadId()).subscribe({
      next: (list) => {
        this.busy.set(false);
        this.setList(list);
      },
      error: (err) => this.fail(err, 'Could not read the show-notes titles.'),
    });
  }

  protected startEdit(c: CandidateOut): void {
    this.editing.set(c.id);
    this.editTitle.set(c.candidate_title);
  }

  protected saveEdit(c: CandidateOut): void {
    const title = this.editTitle().trim();
    if (!title || title.length > MAX_TITLE_CHARS) return;
    this.ctr.update(c.id, { title }).subscribe({
      next: () => {
        this.editing.set(null);
        this.refresh();
      },
      error: (err) => this.fail(err, 'Saving the title failed.'),
    });
  }

  protected replaceThumbnail(c: CandidateOut, event: Event): void {
    const file = (event.target as HTMLInputElement).files?.[0];
    if (!file) return;
    if (file.size > MAX_THUMBNAIL_BYTES) {
      this.error.set('Thumbnails can be at most 2 MB (YouTube’s limit).');
      return;
    }
    this.ctr.update(c.id, { thumbnail: file }).subscribe({
      next: () => this.refresh(),
      error: (err) => this.fail(err, 'Replacing the thumbnail failed.'),
    });
  }

  protected remove(c: CandidateOut): void {
    this.ctr.remove(c.id).subscribe({
      next: () => this.refresh(),
      error: (err) => this.fail(err, 'Deleting the candidate failed.'),
    });
  }

  protected toggle(id: number): void {
    this.expanded.update((set) => {
      const next = new Set(set);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }

  protected thumbnailUrl(c: CandidateOut): string {
    // `scored_at`/id bust the browser cache after a replacement clears and re-scores it.
    return thumbnailUrl(c.id, this.token(), c.scored_at);
  }

  protected subScores(
    c: CandidateOut,
  ): { key: string; label: string; value: number; weight: number }[] {
    const subs = c.features?.sub_scores ?? {};
    const weights = c.features?.weights ?? {};
    return Object.entries(subs).map(([key, value]) => ({
      key,
      label: SUB_SCORE_LABELS[key] ?? key,
      value,
      weight: weights[key] ?? 0,
    }));
  }

  /** The zero-weight features, shown so the owner sees what was measured but not counted. */
  protected loggedFeatures(c: CandidateOut): { key: string; value: string }[] {
    const t = c.features?.title ?? {};
    const th = c.features?.thumbnail ?? {};
    const out: { key: string; value: string }[] = [
      { key: 'characters', value: String(t['title_chars'] ?? '–') },
    ];
    for (const key of [
      'has_digit',
      'has_question',
      'has_exclamation',
      'has_brackets',
      'second_person',
    ]) {
      if (t[key]) out.push({ key: key.replace('has_', '').replace('_', ' '), value: 'yes' });
    }
    if (t['caps_words']) out.push({ key: 'all-caps words', value: String(t['caps_words']) });
    if (c.features?.thumbnail) {
      out.push(
        { key: 'faces', value: String(th['face_count']) },
        { key: 'largest face', value: `${Math.round((th['largest_face_area'] ?? 0) * 100)} %` },
        { key: 'text area', value: `${Math.round((th['text_area'] ?? 0) * 100)} %` },
        { key: 'brightness', value: (th['brightness'] ?? 0).toFixed(2) },
        { key: 'contrast', value: (th['contrast'] ?? 0).toFixed(2) },
        { key: 'saturation', value: (th['saturation'] ?? 0).toFixed(2) },
      );
      const ratio = th['aspect_ratio'] ?? 0;
      if (Math.abs(ratio - 16 / 9) > 0.05)
        out.push({ key: 'aspect ratio', value: `${ratio.toFixed(2)} (YouTube expects 16:9)` });
    }
    return out;
  }
}
