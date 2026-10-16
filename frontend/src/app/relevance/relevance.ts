import { HttpClient } from '@angular/common/http';
import { Service, inject } from '@angular/core';
import { Observable } from 'rxjs';

/** Mirrors backend/app/schemas/relevance.py. */
export interface RelevanceWindow {
  start: number;
  end: number;
  score: number;
}

export interface TopMoment extends RelevanceWindow {
  text: string;
}

export interface RelevanceOut {
  version: string;
  windows: RelevanceWindow[];
  top_moments: TopMoment[];
}

@Service()
export class Relevance {
  private readonly http = inject(HttpClient);

  /** GET /uploads/{id}/relevance -- see docs/decisions/0012 §6. */
  getRelevance(uploadId: number): Observable<RelevanceOut> {
    return this.http.get<RelevanceOut>(`/uploads/${uploadId}/relevance`);
  }
}
