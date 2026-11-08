import { Component, computed, input, output, signal } from '@angular/core';

import { formatTimestamp } from '../../chapters/chapters';
import {
  ClipOut,
  HOOK_LOGGED_FEATURES,
  HOOK_SCORED_INPUTS,
  HOOK_SUB_SCORE_LABELS,
  clipFileUrl,
} from '../../shorts/shorts';

/**
 * The rendered shorts of one episode: a player per clip, its source time
 * range and score, a download link, and "Show in episode", which asks the
 * parent to seek its own player there. See docs/decisions/0013 §8.
 *
 * Since M14 each card also shows the clip's hook score (relative among
 * this episode's clips, not a retention %) with a breakdown
 * (docs/decisions/0020 §6).
 */
@Component({
  selector: 'app-shorts-panel',
  imports: [],
  templateUrl: './shorts-panel.html',
  styleUrl: './shorts-panel.scss',
})
export class ShortsPanel {
  readonly clips = input.required<readonly ClipOut[]>();
  readonly token = input.required<string>();
  readonly seek = output<number>();

  protected readonly withHours = computed(() => this.clips().some((c) => c.end >= 3600));
  protected readonly anyHook = computed(() => this.clips().some((c) => c.hook !== null));
  protected readonly expanded = signal<ReadonlySet<number>>(new Set());
  protected readonly formatTimestamp = formatTimestamp;

  protected videoUrl(clip: ClipOut): string {
    return clipFileUrl(clip.id, this.token());
  }

  protected downloadUrl(clip: ClipOut): string {
    return clipFileUrl(clip.id, this.token(), true);
  }

  /** CSS aspect-ratio for the player box, e.g. "9 / 16". */
  protected ratio(clip: ClipOut): string {
    return clip.aspect_ratio.replace(':', ' / ');
  }

  protected toggle(id: number): void {
    this.expanded.update((set) => {
      const next = new Set(set);
      if (!next.delete(id)) next.add(id);
      return next;
    });
  }

  protected subScores(clip: ClipOut): { key: string; label: string; value: number; weight: number }[] {
    const f = clip.hook?.features;
    const subs = f?.sub_scores ?? {};
    return Object.keys(HOOK_SUB_SCORE_LABELS)
      .filter((key) => key in subs)
      .map((key) => ({
        key,
        label: HOOK_SUB_SCORE_LABELS[key],
        value: subs[key],
        weight: f?.weights?.[key] ?? 0,
      }));
  }

  /** The measured inputs behind the four sub-scores. */
  protected scoredInputs(clip: ClipOut): { key: string; value: string }[] {
    return this.pick(clip, HOOK_SCORED_INPUTS);
  }

  /** Measured, weight 0 in v0. */
  protected loggedFeatures(clip: ClipOut): { key: string; value: string }[] {
    return this.pick(clip, HOOK_LOGGED_FEATURES);
  }

  private pick(clip: ClipOut, keys: readonly string[]): { key: string; value: string }[] {
    const raw = clip.hook?.features.features ?? {};
    return keys.filter((k) => k in raw).map((k) => ({ key: k, value: this.show(raw[k]) }));
  }

  protected languageNote(clip: ClipOut): string | null {
    const f = clip.hook?.features;
    if (!f?.language) return null;
    return f.language_fallback ? `language: ${f.language} (fallback)` : `language: ${f.language}`;
  }

  private show(v: unknown): string {
    if (v === null || v === undefined) return '–';
    if (typeof v === 'number') return Number.isInteger(v) ? String(v) : v.toFixed(2);
    return String(v);
  }
}
