import { HttpClient } from '@angular/common/http';
import { Service, inject, signal } from '@angular/core';
import { Observable, tap } from 'rxjs';

/** Which LLM the chapter-titles, show-notes and relevance tools use (M16, docs/decisions/0025 §11). */
export type LlmChoice = 'qwen' | 'own';

/** The tools that run an LLM, so the only ones offering the Qwen / own choice. */
export const LLM_TOOLS = ['chapters', 'shownotes', 'relevance'] as const;

/** Mirrors backend/app/routers/settings.py's `LlmSettingsOut`. */
export interface LlmSettingsOut {
  default: LlmChoice;
  /** False when no own model is exported on this machine: the option is hidden. */
  own_available: boolean;
  own_label: string | null;
  qwen_label: string | null;
}

/**
 * GET / PUT /settings/llm. The last response is kept in `current` so the Settings page and the
 * episode workspace share one value; `load()` is cheap to call again.
 */
@Service()
export class LlmSettings {
  private readonly http = inject(HttpClient);
  readonly current = signal<LlmSettingsOut | null>(null);

  load(): Observable<LlmSettingsOut> {
    return this.http.get<LlmSettingsOut>('/settings/llm').pipe(tap((s) => this.current.set(s)));
  }

  setDefault(choice: LlmChoice): Observable<LlmSettingsOut> {
    return this.http.put<LlmSettingsOut>('/settings/llm', { default: choice }).pipe(tap((s) => this.current.set(s)));
  }
}

/** The label shown for a choice: the model's own name when the backend knows it. */
export function llmLabel(choice: LlmChoice, s: LlmSettingsOut | null): string {
  if (choice === 'qwen') return s?.qwen_label ? `Qwen (${s.qwen_label})` : 'Qwen';
  return s?.own_label ? `Our model (${s.own_label})` : 'Our model';
}
