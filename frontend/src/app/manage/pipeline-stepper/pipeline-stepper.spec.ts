import { ComponentFixture, TestBed } from '@angular/core/testing';

import { Explain } from '../../explain/explain';
import { nextSuggested, toolStates } from '../../explain/pipeline';
import { PipelineStepper } from './pipeline-stepper';

describe('PipelineStepper', () => {
  let fixture: ComponentFixture<PipelineStepper>;
  const el = () => fixture.nativeElement as HTMLElement;

  beforeEach(async () => {
    await TestBed.configureTestingModule({ imports: [PipelineStepper] }).compileComponents();
    fixture = TestBed.createComponent(PipelineStepper);
    const states = toolStates([]);
    fixture.componentRef.setInput('states', states);
    fixture.componentRef.setInput('suggestion', nextSuggested(states));
    fixture.detectChanges();
  });

  it('puts an ⓘ next to every tool, opening that tool’s explanation', () => {
    const items = el().querySelectorAll('.stepper__item');
    expect(items.length).toBe(8);
    for (const item of items) {
      expect(item.querySelector('.info-btn')).not.toBeNull();
    }
    const relevance = [...items].find((i) => i.textContent!.includes('Relevance'))!;
    relevance.querySelector<HTMLButtonElement>('.info-btn')!.click();
    expect(TestBed.inject(Explain).openKey()).toBe('relevance');
  });

  it('the node itself still selects the tool', () => {
    const spy = vi.fn();
    fixture.componentInstance.selectTool.subscribe(spy);
    el().querySelector<HTMLButtonElement>('.stepper__node')!.click();
    expect(spy).toHaveBeenCalledWith('transcribe');
  });
});
