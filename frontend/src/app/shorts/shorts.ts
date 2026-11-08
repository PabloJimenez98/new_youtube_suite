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
  /** Latest hook score (M14), null until the `hook` job has scored the clip. */
  hook: HookScore | null;
}

/** Mirrors `HookOut` / the stored feature vector (docs/decisions/0020 §2-5). */
export interface HookScore {
  score: number; // 0-100, relative among this episode's clips
  rank: number; // ties share a rank
  scored_at: string;
  features: {
    version?: string;
    language?: string;
    language_fallback?: boolean;
    title_source?: string;
    sub_scores?: Record<string, number>;
    weights?: Record<string, number>;
    features?: Record<string, unknown>;
  };
}

export const HOOK_SUB_SCORE_LABELS: Record<string, string> = {
  start: 'Starts at once (no dead air)',
  pace: 'Pace (words/s)',
  framing: 'Opens with a question / curiosity gap',
  address: 'Direct address ("you", imperatives)',
};

/** The measured inputs of the four sub-scores (0020 §3). */
export const HOOK_SCORED_INPUTS = [
  'lead_silence_s',
  'words_per_second',
  'first_is_question',
  'first_curiosity',
  'first_unresolved_demonstratives',
  'second_person',
  'imperatives',
] as const;

/** Measured but weight 0 in v0, shown in the breakdown (0020 §2). */
export const HOOK_LOGGED_FEATURES = [
  'emotion_max',
  'exclamations',
  'superlatives',
  'numerics',
  'entity_density',
  'readability',
  'title_overlap',
  'words',
  'sentences',
] as const;

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
