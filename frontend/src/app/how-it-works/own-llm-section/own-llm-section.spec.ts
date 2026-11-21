import { ComponentFixture, TestBed } from '@angular/core/testing';

import { Explain } from '../../explain/explain';
import { GATES, ablations, gateRates, paramBreakdown, passes, scaling } from '../own-llm-charts';
import { OWN_LLM_DATA } from '../own-llm-data';
import { OwnLlmSection } from './own-llm-section';

type Any = any; // eslint-disable-line @typescript-eslint/no-explicit-any -- loosely typed ECharts options

describe('own LLM data (generated from the measured runs)', () => {
  it('has the stage A architecture and an exact parameter count', () => {
    const p = OWN_LLM_DATA.arch.params;
    expect(p.total).toBe(100_682_496);
    expect(p.embedding + p.attention + p.ffn + p.norms).toBe(p.total);
    expect(OWN_LLM_DATA.arch.n_layers).toBe(12);
  });

  it('ends pretraining at the recorded 5B tokens and 0.963 bpb', () => {
    const val = OWN_LLM_DATA.pretrain.val;
    expect(val[val.length - 1][0]).toBeCloseTo(5, 1);
    expect(val[val.length - 1][1]).toBeCloseTo(0.9629, 4);
  });
});

describe('own LLM charts', () => {
  it('splits the parameters into three direct-labelled parts that add up', () => {
    const b = paramBreakdown();
    const series = (b.option as Any).series as Any[];
    expect(series.length).toBe(3);
    expect(series.every((s) => s.label.show)).toBe(true);
    expect(b.table.rows[b.table.rows.length - 1][1]).toBe('100,682,496');
  });

  it('marks the adopted ablation and colours better vs worse', () => {
    const s = (ablations().option as Any).series[0];
    const chosen = s.data.find((d: Any) => d.itemStyle.borderWidth === 2);
    expect(chosen.value).toBeLessThan(0);
    expect(s.data.length).toBe(OWN_LLM_DATA.ablations.length - 1);
  });

  it('shows bpb falling as tokens grow', () => {
    const pts = (scaling().option as Any).series[0].data as number[][];
    for (let i = 1; i < pts.length; i++) expect(pts[i][1]).toBeLessThan(pts[i - 1][1]);
  });

  it('grades the gates: titles pass, show notes and relevance fail', () => {
    const verdict = Object.fromEntries(GATES.map((g) => [g.gate, passes(g, g.dpo)]));
    expect(verdict['Titles: no fallback to mT5']).toBe(true);
    expect(verdict['Titles: right language']).toBe(true);
    expect(verdict['Show notes: valid within 3 tries']).toBe(false);
    expect(verdict['Relevance: ranks like Qwen (Spearman)']).toBe(false);
    expect(gateRates().table.rows.length).toBe(3);
  });
});

describe('OwnLlmSection', () => {
  let fixture: ComponentFixture<OwnLlmSection>;
  const el = () => fixture.nativeElement as HTMLElement;

  beforeEach(async () => {
    await TestBed.configureTestingModule({ imports: [OwnLlmSection] }).compileComponents();
    fixture = TestBed.createComponent(OwnLlmSection);
    fixture.detectChanges();
  });

  it('is the #own-llm anchor and opens the model explanation', () => {
    expect(el().querySelector('#own-llm')).not.toBeNull();
    el().querySelector<HTMLButtonElement>('.own__head .info-btn')!.click();
    expect(TestBed.inject(Explain).openKey()).toBe('ownllm');
  });

  it('gives the verdict per task at a glance', () => {
    const verdicts = [...el().querySelectorAll('.own__verdict')];
    expect(verdicts.map((v) => v.classList.contains('is-ok'))).toEqual([true, false, false]);
  });

  it('draws the architecture with the 12-block repeat and every chart card', () => {
    expect(el().querySelector('.arch__repeat')?.textContent).toContain('12');
    expect(el().querySelectorAll('app-chart-card').length).toBe(13);
  });

  it('lists every gate with a pass/fail badge', () => {
    const rows = el().querySelectorAll('.own__gates tbody tr');
    expect(rows.length).toBe(GATES.length);
    expect(el().querySelectorAll('.own__gates .badge--succeeded').length).toBe(3);
    expect(el().querySelectorAll('.own__gates .badge--failed').length).toBe(2);
  });
});
