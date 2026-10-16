import { HttpClient } from '@angular/common/http';
import { Service, inject } from '@angular/core';
import { Observable, map } from 'rxjs';

import { AspectRatio } from '../jobs/job.model';

/** Mirrors backend/app/schemas/clip.py. */
export interface ClipShot {
  start: number;
  end: number;
  mode: 'crop' | 'fit';
}

export interface ClipOut {
  id: number;
  rank: number;
  start: number;
  end: number;
  duration: number;
  score: number;
  aspect_ratio: AspectRatio;
  caption_burned_in: boolean;
  framing: { source?: number[]; output?: number[]; shots?: ClipShot[] };
}

@Service()
export class Shorts {
  private readonly http = inject(HttpClient);

  /** GET /uploads/{id}/clips -- see docs/decisions/0013 §7. */
  getClips(uploadId: number): Observable<ClipOut[]> {
    return this.http.get<ClipOut[]>(`/uploads/${uploadId}/clips`);
  }

  /** GET /clips/count -- every rendered short, for the dashboard. */
  countClips(): Observable<number> {
    return this.http.get<{ count: number }>('/clips/count').pipe(map((r) => r.count));
  }
}

/**
 * The clip file URL for a <video src> or a download link. Like the
 * upload preview, it carries the token as a query param because the
 * browser fetches it without Angular's interceptor (docs/decisions/0005).
 */
export function clipFileUrl(clipId: number, token: string, download = false): string {
  return `/clips/${clipId}/file?token=${encodeURIComponent(token)}${download ? '&download=true' : ''}`;
}
