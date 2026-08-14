import { Component, computed, input } from '@angular/core';

import { MarketTrend } from '../analytics-dummy-data';

/** Plain div bars -- no charting library warranted for a dummy/preview chart. */
@Component({
  selector: 'app-market-chart',
  imports: [],
  templateUrl: './market-chart.html',
  styleUrl: './market-chart.scss',
})
export class MarketChart {
  readonly trends = input.required<readonly MarketTrend[]>();

  protected readonly maxScore = computed(() =>
    Math.max(...this.trends().map((t) => t.trendScore), 1),
  );
}
