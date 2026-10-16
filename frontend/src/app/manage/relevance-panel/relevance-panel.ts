import { Component, computed, input, output } from '@angular/core';

import { formatTimestamp } from '../../chapters/chapters';
import { RelevanceOut } from '../../relevance/relevance';

const WIDTH = 600;
const HEIGHT = 80;

/**
 * Relevance v0 curve (fixed 0-1 scale) plus the top moments; clicking a
 * moment asks the parent to seek its video there. See docs/decisions/0012 §7.
 */
@Component({
  selector: 'app-relevance-panel',
  imports: [],
  templateUrl: './relevance-panel.html',
  styleUrl: './relevance-panel.scss',
})
export class RelevancePanel {
  readonly relevance = input.required<RelevanceOut>();
  readonly seek = output<number>();

  protected readonly width = WIDTH;
  protected readonly height = HEIGHT;

  private readonly duration = computed(() => {
    const windows = this.relevance().windows;
    return windows.length ? windows[windows.length - 1].end : 0;
  });

  protected readonly withHours = computed(() => this.duration() >= 3600);

  /** Each window plotted at its midpoint; y is the score on a fixed 0-1 scale. */
  protected readonly points = computed(() => {
    const d = this.duration() || 1;
    return this.relevance()
      .windows.map((w) => {
        const x = (((w.start + w.end) / 2) / d) * WIDTH;
        const y = HEIGHT - Math.min(Math.max(w.score, 0), 1) * HEIGHT;
        return `${x.toFixed(1)},${y.toFixed(1)}`;
      })
      .join(' ');
  });

  /** Time-axis ticks every 10 min (every 5 min under 30 min). */
  protected readonly ticks = computed(() => {
    const d = this.duration();
    const step = d < 1800 ? 300 : 600;
    const out: { x: number; label: string }[] = [];
    for (let t = 0; t <= d; t += step) {
      out.push({ x: (t / (d || 1)) * WIDTH, label: formatTimestamp(t, this.withHours()) });
    }
    return out;
  });

  protected readonly markers = computed(() => {
    const d = this.duration() || 1;
    return this.relevance().top_moments.map((m) => ({
      x: (m.start / d) * WIDTH,
      y: HEIGHT - m.score * HEIGHT,
    }));
  });

  protected readonly formatTimestamp = formatTimestamp;

  protected trim(text: string): string {
    return text.length > 140 ? `${text.slice(0, 139).trimEnd()}…` : text;
  }
}
