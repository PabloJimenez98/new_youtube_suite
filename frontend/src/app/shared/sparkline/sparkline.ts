import { Component, computed, input } from '@angular/core';

/** Minimal SVG polyline sparkline -- no charting library needed for this. */
@Component({
  selector: 'app-sparkline',
  imports: [],
  templateUrl: './sparkline.html',
  styleUrl: './sparkline.scss',
})
export class Sparkline {
  readonly values = input.required<readonly number[]>();
  readonly width = input(160);
  readonly height = input(40);

  protected readonly points = computed(() => {
    const vals = this.values();
    if (vals.length === 0) {
      return '';
    }
    const w = this.width();
    const h = this.height();
    const min = Math.min(...vals);
    const max = Math.max(...vals);
    const range = max - min || 1;
    const step = vals.length > 1 ? w / (vals.length - 1) : 0;

    return vals
      .map((v, i) => {
        const x = i * step;
        const y = h - ((v - min) / range) * h;
        return `${x.toFixed(1)},${y.toFixed(1)}`;
      })
      .join(' ');
  });
}
