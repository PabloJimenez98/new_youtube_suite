import { ComponentFixture, TestBed } from '@angular/core/testing';

import { Explain } from '../explain/explain';
import { GLOSSARY } from '../explain/glossary';
import { SYSTEM_FLOW } from '../explain/tool-docs';
import { HowItWorks } from './how-it-works';

describe('HowItWorks', () => {
  let fixture: ComponentFixture<HowItWorks>;
  const el = () => fixture.nativeElement as HTMLElement;

  beforeEach(async () => {
    await TestBed.configureTestingModule({ imports: [HowItWorks] }).compileComponents();
    fixture = TestBed.createComponent(HowItWorks);
    fixture.detectChanges();
  });

  it('draws the whole pipeline with clickable nodes that open their explanation', () => {
    const nodes = el().querySelectorAll<SVGGElement>('.flow__node--link');
    expect(nodes.length).toBe(SYSTEM_FLOW.nodes.length);

    const relevance = [...nodes].find((n) => n.getAttribute('aria-label') === 'Explain Relevance')!;
    relevance.dispatchEvent(new MouseEvent('click', { bubbles: true }));
    expect(TestBed.inject(Explain).openKey()).toBe('relevance');
  });

  it('lists the model status and the full glossary', () => {
    expect(el().querySelectorAll('.hiw__table tbody tr').length).toBe(7);
    expect(el().querySelectorAll('.hiw__glossary dt').length).toBe(Object.keys(GLOSSARY).length);
  });
});
