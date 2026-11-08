import { Component, computed, input, output } from '@angular/core';

import { formatTimestamp } from '../../chapters/chapters';
import { ClipOut, HOOK_SUB_SCORE_LABELS } from '../../shorts/shorts';
import { InfoButton } from '../../shared/info-button/info-button';

const SUB_KEYS = ['start', 'pace', 'framing', 'address'] as const;
const SHORT_LABELS: Record<(typeof SUB_KEYS)[number], string> = {
  start: 'Starts at once',
  pace: 'Pace',
  framing: 'Question / curiosity',
  address: 'Talks to “you”',
};

export interface HookRow {
  clip: ClipOut;
  score: number;
  rank: number;
  subs: { key: string; label: string; value: number; weight: number }[];
  leadSilence: number | null;
  wordsPerSecond: number | null;
  question: boolean;
  strongest: string | null;
  weakest: string | null;
}

/** Rank the episode's clips by hook score, with what makes each opening strong or weak. */
export function hookRows(clips: readonly ClipOut[]): HookRow[] {
  return clips
    .filter((c) => c.hook)
    .map((c) => {
      const f = c.hook!.features;
      const subs = SUB_KEYS.filter((k) => f.sub_scores?.[k] != null).map((k) => ({
        key: k,
        label: SHORT_LABELS[k],
        value: f.sub_scores![k],
        weight: f.weights?.[k] ?? 0,
      }));
      const sorted = [...subs].sort((a, b) => b.value - a.value);
      const raw = f.features ?? {};
      const num = (k: string) => (typeof raw[k] === 'number' ? (raw[k] as number) : null);
      return {
        clip: c,
        score: c.hook!.score,
        rank: c.hook!.rank,
        subs,
        leadSilence: num('lead_silence_s'),
        wordsPerSecond: num('words_per_second'),
        question: raw['first_is_question'] === true,
        strongest: sorted[0] && sorted[0].value >= 0.7 ? sorted[0].label : null,
        weakest: sorted.length && sorted[sorted.length - 1].value < 0.5 ? sorted[sorted.length - 1].label : null,
      };
    })
    .sort((a, b) => a.rank - b.rank || b.score - a.score);
}

/**
 * The Shorts tab's hook overview (docs/decisions/0022 §4): every scored
 * clip ranked, its four sub-scores as bars, and the measured inputs.
 */
@Component({
  selector: 'app-hook-summary',
  imports: [InfoButton],
  templateUrl: './hook-summary.html',
  styleUrl: './hook-summary.scss',
})
export class HookSummary {
  readonly clips = input.required<readonly ClipOut[]>();
  readonly seek = output<number>();

  protected readonly rows = computed(() => hookRows(this.clips()));
  protected readonly subKeys = SUB_KEYS;
  protected readonly shortLabels = SHORT_LABELS;
  protected readonly longLabels = HOOK_SUB_SCORE_LABELS;
  protected readonly weights = computed(() => this.rows()[0]?.subs.reduce<Record<string, number>>((m, s) => ({ ...m, [s.key]: s.weight }), {}) ?? {});
  protected readonly unscored = computed(() => this.clips().filter((c) => !c.hook).length);
  protected readonly withHours = computed(() => this.clips().some((c) => c.end >= 3600));

  protected ts(s: number): string {
    return formatTimestamp(s, this.withHours());
  }

  protected sub(row: HookRow, key: string): number | null {
    return row.subs.find((s) => s.key === key)?.value ?? null;
  }
}
