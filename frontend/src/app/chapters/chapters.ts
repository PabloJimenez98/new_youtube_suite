import { HttpClient } from '@angular/common/http';
import { Service, inject } from '@angular/core';
import { Observable } from 'rxjs';

/** Mirrors backend/app/schemas/chapter.py's `ChapterOut`. */
export interface ChapterOut {
  order_index: number;
  start_seconds: number;
  title: string;
  model_version: string;
}

@Service()
export class Chapters {
  private readonly http = inject(HttpClient);

  /** GET /uploads/{id}/chapters -- see docs/decisions/0009 §7. */
  getChapters(uploadId: number): Observable<ChapterOut[]> {
    return this.http.get<ChapterOut[]>(`/uploads/${uploadId}/chapters`);
  }
}

const pad = (n: number) => String(n).padStart(2, '0');

/**
 * YouTube chapter timestamp: `MM:SS`, or `H:MM:SS` for every line once
 * the episode passes an hour (consistent width reads better in a
 * description; YouTube accepts both).
 */
export function formatTimestamp(seconds: number, withHours: boolean): string {
  const s = Math.floor(seconds);
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  return withHours ? `${h}:${pad(m)}:${pad(s % 60)}` : `${pad(m + h * 60)}:${pad(s % 60)}`;
}

/** The block to paste into a YouTube description: one `timestamp title` line per chapter. */
export function toYouTubeText(chapters: readonly ChapterOut[]): string {
  const withHours = chapters.some((c) => c.start_seconds >= 3600);
  return chapters.map((c) => `${formatTimestamp(c.start_seconds, withHours)} ${c.title}`).join('\n');
}
