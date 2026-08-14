import { DecimalPipe } from '@angular/common';
import { Component } from '@angular/core';

import { PreviewBanner } from '../shared/preview-banner/preview-banner';
import { DUMMY_CHANNEL_METRICS, DUMMY_MARKET_TRENDS } from './analytics-dummy-data';
import { MarketChart } from './market-chart/market-chart';

@Component({
  selector: 'app-analytics',
  imports: [PreviewBanner, MarketChart, DecimalPipe],
  templateUrl: './analytics.html',
  styleUrl: './analytics.scss',
})
export class Analytics {
  protected readonly trends = DUMMY_MARKET_TRENDS;
  protected readonly channelMetrics = DUMMY_CHANNEL_METRICS;
}
