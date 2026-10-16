import { HttpClient } from '@angular/common/http';
import { Service, inject } from '@angular/core';
import { Observable } from 'rxjs';

/** Mirrors backend/app/schemas/show_notes.py's `ShowNotesOut`. */
export interface ShowNotesOut {
  title_suggestions: string[];
  description: string;
  tags: string[];
  summary: string;
  language: string;
  llm_model_used: string;
  created_at: string;
  updated_at: string;
}

/** `PATCH` body: any subset of the editable fields. */
export type ShowNotesUpdate = Partial<
  Pick<ShowNotesOut, 'title_suggestions' | 'description' | 'tags' | 'summary'>
>;

@Service()
export class ShowNotes {
  private readonly http = inject(HttpClient);

  /** GET /uploads/{id}/shownotes -- see docs/decisions/0011 §5. */
  getShowNotes(uploadId: number): Observable<ShowNotesOut> {
    return this.http.get<ShowNotesOut>(`/uploads/${uploadId}/shownotes`);
  }

  /** PATCH /uploads/{id}/shownotes -- the backend enforces YouTube's limits (422). */
  updateShowNotes(uploadId: number, body: ShowNotesUpdate): Observable<ShowNotesOut> {
    return this.http.patch<ShowNotesOut>(`/uploads/${uploadId}/shownotes`, body);
  }
}

/** "a, b ,, c" -> ["a", "b", "c"]. */
export function parseTags(text: string): string[] {
  return text
    .split(',')
    .map((t) => t.trim())
    .filter(Boolean);
}
