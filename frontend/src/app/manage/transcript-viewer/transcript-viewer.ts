import { DecimalPipe } from '@angular/common';
import { Component, computed, effect, inject, input, output, signal, untracked } from '@angular/core';

import { formatTimestamp } from '../../chapters/chapters';
import { SERIES } from '../../charts/viz';
import { TranscriptOut, TranscriptSegment, Transcripts } from '../../transcripts/transcripts';

export interface Turn {
  speaker: string | null;
  start: number;
  end: number;
  text: string;
}

/** Without speakers, paragraphs break after this many seconds. */
const PARAGRAPH_S = 60;

/** Consecutive segments of one speaker become one turn (docs/decisions/0022 §4). */
export function groupTurns(segments: TranscriptSegment[], hasSpeakers: boolean): Turn[] {
  const turns: Turn[] = [];
  for (const s of segments) {
    const last = turns[turns.length - 1];
    const same = last && (hasSpeakers ? last.speaker === s.speaker : s.start - last.start < PARAGRAPH_S);
    if (same) {
      last.end = s.end;
      last.text += ` ${s.text}`;
    } else {
      turns.push({ speaker: hasSpeakers ? s.speaker : null, start: s.start, end: s.end, text: s.text });
    }
  }
  return turns;
}

/** Share of speaking time per speaker, from segment durations. */
export function talkShare(segments: TranscriptSegment[], speakers: string[]): { speaker: string; seconds: number; share: number }[] {
  const secs = new Map(speakers.map((s) => [s, 0]));
  for (const s of segments) {
    if (s.speaker) secs.set(s.speaker, (secs.get(s.speaker) ?? 0) + Math.max(0, s.end - s.start));
  }
  const total = [...secs.values()].reduce((a, b) => a + b, 0) || 1;
  return speakers.map((speaker) => ({ speaker, seconds: secs.get(speaker)!, share: secs.get(speaker)! / total }));
}

/** The episode's transcript, readable in the app, with speakers once diarized. */
@Component({
  selector: 'app-transcript-viewer',
  imports: [DecimalPipe],
  templateUrl: './transcript-viewer.html',
  styleUrl: './transcript-viewer.scss',
})
export class TranscriptViewer {
  private readonly transcripts = inject(Transcripts);

  readonly uploadId = input.required<number>();
  /** Changes when a newer transcribe or diarize job succeeds: refetch. */
  readonly version = input<string>('');
  readonly seek = output<number>();

  protected readonly data = signal<TranscriptOut | null>(null);
  protected readonly loading = signal(false);
  protected readonly error = signal(false);
  protected readonly query = signal('');

  protected readonly hasSpeakers = computed(() => (this.data()?.speakers.length ?? 0) > 0);
  protected readonly turns = computed(() => {
    const d = this.data();
    return d ? groupTurns(d.segments, this.hasSpeakers()) : [];
  });
  protected readonly visible = computed(() => {
    const q = this.query().trim().toLowerCase();
    return q ? this.turns().filter((t) => t.text.toLowerCase().includes(q)) : this.turns();
  });
  protected readonly share = computed(() => {
    const d = this.data();
    return d && this.hasSpeakers() ? talkShare(d.segments, d.speakers) : [];
  });
  protected readonly withHours = computed(() => this.turns().some((t) => t.start >= 3600));
  protected readonly words = computed(() => this.data()?.segments.reduce((n, s) => n + s.text.split(/\s+/).length, 0) ?? 0);

  constructor() {
    effect(() => {
      const id = this.uploadId();
      this.version();
      untracked(() => this.load(id));
    });
  }

  private load(id: number): void {
    this.loading.set(true);
    this.error.set(false);
    this.transcripts.getTranscript(id).subscribe({
      next: (d) => {
        this.data.set(d);
        this.loading.set(false);
      },
      error: () => {
        this.error.set(true);
        this.loading.set(false);
      },
    });
  }

  /** "SPEAKER_00" → "Speaker 1". */
  protected speakerName(label: string | null): string {
    const n = label?.match(/(\d+)$/);
    return n ? `Speaker ${Number(n[1]) + 1}` : (label ?? '');
  }

  /** Fixed colour per speaker by order of first appearance; past 8 they share the last. */
  protected speakerColor(label: string | null): string {
    const i = this.data()?.speakers.indexOf(label ?? '') ?? -1;
    return i < 0 ? 'var(--color-text-muted)' : SERIES[Math.min(i, SERIES.length - 1)];
  }

  protected ts(seconds: number): string {
    return formatTimestamp(seconds, this.withHours());
  }

  /** Text split around the search query, for highlighting. */
  protected parts(text: string): { text: string; hit: boolean }[] {
    const q = this.query().trim();
    if (!q) return [{ text, hit: false }];
    const out: { text: string; hit: boolean }[] = [];
    const lower = text.toLowerCase();
    const ql = q.toLowerCase();
    let i = 0;
    for (let at = lower.indexOf(ql); at >= 0; at = lower.indexOf(ql, i)) {
      if (at > i) out.push({ text: text.slice(i, at), hit: false });
      out.push({ text: text.slice(at, at + q.length), hit: true });
      i = at + q.length;
    }
    if (i < text.length) out.push({ text: text.slice(i), hit: false });
    return out;
  }
}
