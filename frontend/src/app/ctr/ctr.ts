import { HttpClient } from '@angular/common/http';
import { Service, inject } from '@angular/core';
import { Observable } from 'rxjs';

export const MAX_TITLE_CHARS = 100;
export const MAX_CANDIDATES = 10;
export const MAX_THUMBNAIL_BYTES = 2 * 1024 * 1024;

export type CandidateGroup = 'full' | 'partial' | 'unscored';

/** The stored breakdown (backend/app/ctr/scorer.py `features_json`); `error` when the thumbnail was unreadable. */
export interface CtrFeatures {
  version: string;
  error?: string;
  title?: Record<string, number | boolean>;
  thumbnail?: Record<string, number> | null;
  sub_scores?: Record<string, number>;
  weights?: Record<string, number>;
}

/** Mirrors backend/app/routers/ctr.py's `CandidateOut`. */
export interface CandidateOut {
  id: number;
  upload_id: number | null;
  candidate_title: string;
  has_thumbnail: boolean;
  predicted_score: number | null;
  partial: boolean;
  features: CtrFeatures | null;
  source: 'manual' | 'show_notes';
  scored_at: string | null;
  group: CandidateGroup;
  rank: number | null;
}

/** Title/thumbnail candidates for the CTR predictor v0 (M13, docs/decisions/0019 §4-5). */
@Service()
export class Ctr {
  private readonly http = inject(HttpClient);

  list(uploadId: number): Observable<CandidateOut[]> {
    return this.http.get<CandidateOut[]>(`/uploads/${uploadId}/ctr-candidates`);
  }

  create(uploadId: number, title: string, thumbnail: File | null): Observable<CandidateOut> {
    const form = new FormData();
    form.append('title', title);
    if (thumbnail) form.append('thumbnail', thumbnail);
    return this.http.post<CandidateOut>(`/uploads/${uploadId}/ctr-candidates`, form);
  }

  fromShowNotes(uploadId: number): Observable<CandidateOut[]> {
    return this.http.post<CandidateOut[]>(
      `/uploads/${uploadId}/ctr-candidates/from-show-notes`,
      {},
    );
  }

  update(
    id: number,
    changes: { title?: string; thumbnail?: File; removeThumbnail?: boolean },
  ): Observable<CandidateOut> {
    const form = new FormData();
    if (changes.title !== undefined) form.append('title', changes.title);
    if (changes.thumbnail) form.append('thumbnail', changes.thumbnail);
    if (changes.removeThumbnail) form.append('remove_thumbnail', 'true');
    return this.http.patch<CandidateOut>(`/ctr-candidates/${id}`, form);
  }

  remove(id: number): Observable<void> {
    return this.http.delete<void>(`/ctr-candidates/${id}`);
  }
}

/** `<img src>` can't send the Authorization header: the token travels as a query param (0005). */
export function thumbnailUrl(id: number, token: string, version: string | null = null): string {
  const bust = version ? `&v=${encodeURIComponent(version)}` : '';
  return `/ctr-candidates/${id}/thumbnail?token=${encodeURIComponent(token)}${bust}`;
}

/** Labels for the breakdown (sub-scores carry weight in v0; the rest are logged only). */
export const SUB_SCORE_LABELS: Record<string, string> = {
  title_length: 'Title length',
  faces: 'Faces',
  face_size: 'Face size',
  text_coverage: 'Text on thumbnail',
};
