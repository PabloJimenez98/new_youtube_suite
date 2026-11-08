import { Component, input, signal } from '@angular/core';

export interface ChartHelp {
  /** What the chart shows, in one or two sentences. */
  what: string;
  /** The method: how the numbers were computed. */
  method?: string;
  /** How to read it, including what it does not say. */
  read: string;
}

export interface ChartTable {
  columns: string[];
  rows: (string | number)[][];
}

let nextId = 0;

/**
 * A chart's frame (docs/decisions/0022 §5): title, an ⓘ explanation
 * tooltip (what / method / how to read), an empty state, and a table
 * view so a value is never readable only through a hover tooltip.
 */
@Component({
  selector: 'app-chart-card',
  imports: [],
  template: `
    <section class="chart-card glass-panel" [attr.aria-labelledby]="titleId">
      <header class="chart-card__head">
        <h3 [id]="titleId">{{ title() }}</h3>
        @if (subtitle()) {
          <span class="chart-card__sub">{{ subtitle() }}</span>
        }
        <span class="chart-card__spacer"></span>
        @if (table() && !empty()) {
          <button type="button" class="chart-card__toggle" [attr.aria-pressed]="showTable()" (click)="showTable.set(!showTable())">
            {{ showTable() ? 'Chart' : 'Table' }}
          </button>
        }
        <span class="chart-help" tabindex="0" [attr.aria-describedby]="helpId">
          <span class="info-btn" aria-hidden="true">i</span>
          <span class="chart-help__tip" role="tooltip" [id]="helpId">
            <strong>What it shows</strong> {{ help().what }}
            @if (help().method) {
              <strong>Method</strong> {{ help().method }}
            }
            <strong>How to read it</strong> {{ help().read }}
          </span>
        </span>
      </header>

      @if (empty()) {
        <p class="chart-card__empty">{{ emptyText() }}</p>
      } @else if (showTable() && table(); as t) {
        <div class="chart-card__table-wrap">
          <table class="chart-card__table">
            <thead>
              <tr>
                @for (c of t.columns; track $index) {
                  <th scope="col">{{ c }}</th>
                }
              </tr>
            </thead>
            <tbody>
              @for (row of t.rows; track $index) {
                <tr>
                  @for (cell of row; track $index) {
                    <td>{{ cell }}</td>
                  }
                </tr>
              }
            </tbody>
          </table>
        </div>
      } @else {
        <ng-content />
      }
      @if (footnote() && !empty()) {
        <p class="chart-card__foot">{{ footnote() }}</p>
      }
    </section>
  `,
  styleUrl: './chart-card.scss',
})
export class ChartCard {
  readonly title = input.required<string>();
  readonly subtitle = input<string | null>(null);
  readonly help = input.required<ChartHelp>();
  readonly table = input<ChartTable | null>(null);
  readonly empty = input(false);
  readonly emptyText = input('No data yet.');
  readonly footnote = input<string | null>(null);

  protected readonly showTable = signal(false);
  private readonly id = nextId++;
  protected readonly titleId = `chart-title-${this.id}`;
  protected readonly helpId = `chart-help-${this.id}`;
}
