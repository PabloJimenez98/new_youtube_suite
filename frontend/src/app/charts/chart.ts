import { Component, ElementRef, OnDestroy, afterNextRender, effect, input, output, viewChild } from '@angular/core';
import { BarChart, BoxplotChart, CustomChart, HeatmapChart, LineChart, ScatterChart } from 'echarts/charts';
import {
  GridComponent,
  LegendComponent,
  MarkAreaComponent,
  MarkLineComponent,
  TooltipComponent,
  VisualMapComponent,
} from 'echarts/components';
import * as echarts from 'echarts/core';
import { CanvasRenderer } from 'echarts/renderers';
import type { EChartsCoreOption } from 'echarts/core';

// Modular build: only what the dashboards use (docs/decisions/0022 §5).
// This file is only imported by lazily loaded pages, so ECharts stays out
// of the initial bundle.
echarts.use([
  BarChart,
  BoxplotChart,
  CustomChart,
  HeatmapChart,
  LineChart,
  ScatterChart,
  GridComponent,
  LegendComponent,
  MarkAreaComponent,
  MarkLineComponent,
  TooltipComponent,
  VisualMapComponent,
  CanvasRenderer,
]);

function canvasSupported(): boolean {
  try {
    return !!document.createElement('canvas').getContext('2d');
  } catch {
    return false;
  }
}

/** One ECharts instance, resized with its container and disposed with the component. */
@Component({
  selector: 'app-chart',
  imports: [],
  template: `<div #host class="chart" [style.height.px]="height()" role="img" [attr.aria-label]="label()"></div>`,
  styles: `
    :host {
      display: block;
    }
    .chart {
      width: 100%;
    }
  `,
})
export class Chart implements OnDestroy {
  readonly option = input.required<EChartsCoreOption>();
  readonly height = input(280);
  /** Accessible name; the card's table view carries the values. */
  readonly label = input('Chart');
  /** Emits the clicked datum's `name` (e.g. to open an episode). */
  readonly pointClick = output<unknown>();

  private readonly host = viewChild.required<ElementRef<HTMLDivElement>>('host');
  private instance?: echarts.ECharts;
  private resizeObserver?: ResizeObserver;
  private visibility?: IntersectionObserver;

  constructor() {
    // Charts are only drawn when they come near the viewport (docs/decisions/0023):
    // a page of ~15 charts no longer builds them all on load.
    afterNextRender(() => {
      const el = this.host().nativeElement;
      if (typeof IntersectionObserver === 'undefined') {
        this.draw();
        return;
      }
      this.visibility = new IntersectionObserver(
        (entries) => {
          if (entries.some((e) => e.isIntersecting)) {
            this.visibility?.disconnect();
            this.draw();
          }
        },
        { rootMargin: '300px' },
      );
      this.visibility.observe(el);
    });
    effect(() => {
      const option = this.option();
      this.instance?.setOption(option, true);
    });
  }

  private draw(): void {
    const el = this.host().nativeElement;
    // No 2D canvas (jsdom in tests, some print paths): skip the drawing; the
    // card's table view still has the data. ECharts would throw later, mid-render.
    if (!canvasSupported()) return;
    try {
      // Canvas: thousands of scatter points stay fast (SVG made one DOM node each).
      this.instance = echarts.init(el, undefined, { renderer: 'canvas' });
    } catch {
      return; // no layout (tests, print): the table view still has the data
    }
    this.instance.on('click', (p) => this.pointClick.emit((p as { data?: unknown }).data));
    this.instance.setOption(this.option(), true);
    if (typeof ResizeObserver !== 'undefined') {
      this.resizeObserver = new ResizeObserver(() => this.instance?.resize());
      this.resizeObserver.observe(el);
    }
  }

  ngOnDestroy(): void {
    this.visibility?.disconnect();
    this.resizeObserver?.disconnect();
    this.instance?.dispose();
  }
}
