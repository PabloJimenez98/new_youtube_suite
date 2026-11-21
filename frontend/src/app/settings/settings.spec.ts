import { ComponentFixture, TestBed } from '@angular/core/testing';
import { of, throwError } from 'rxjs';
import { WritableSignal, signal } from '@angular/core';
import { provideRouter } from '@angular/router';

import { LlmSettings, LlmSettingsOut } from './llm-settings';
import { Settings } from './settings';

function out(overrides: Partial<LlmSettingsOut> = {}): LlmSettingsOut {
  return { default: 'qwen', own_available: true, own_label: 'yts-110m-a3', qwen_label: 'Qwen3.5-9B-Q4_K_M', ...overrides };
}

describe('Settings', () => {
  let fixture: ComponentFixture<Settings>;
  let stub: { current: WritableSignal<LlmSettingsOut | null>; load: ReturnType<typeof vi.fn>; setDefault: ReturnType<typeof vi.fn> };
  const el = () => fixture.nativeElement as HTMLElement;
  const radio = (v: string) => el().querySelector<HTMLInputElement>(`input[value="${v}"]`)!;

  async function setup(initial: LlmSettingsOut) {
    const current = signal<LlmSettingsOut | null>(null);
    stub = {
      current,
      load: vi.fn(() => {
        current.set(initial);
        return of(initial);
      }),
      setDefault: vi.fn(),
    };
    await TestBed.configureTestingModule({
      imports: [Settings],
      providers: [provideRouter([]), { provide: LlmSettings, useValue: stub }],
    }).compileComponents();
    fixture = TestBed.createComponent(Settings);
    fixture.detectChanges();
  }

  it('shows both models with the current default checked', async () => {
    await setup(out());
    expect(radio('qwen').checked).toBe(true);
    expect(radio('own').checked).toBe(false);
    expect(el().textContent).toContain('Our model (yts-110m-a3)');
  });

  it('disables our model while it is not installed', async () => {
    await setup(out({ own_available: false, own_label: null }));
    expect(radio('own').disabled).toBe(true);
    expect(el().textContent).toContain('Not installed');
    expect(el().querySelector('.settings__more')).toBeNull();
  });

  it('links to the explanation of our model when it is installed', async () => {
    await setup(out());
    const more = el().querySelector('.settings__more')!;
    expect(more.querySelector('.info-btn')?.getAttribute('aria-label')).toBe('How Our own language model works');
    expect(more.querySelector('a')?.getAttribute('href')).toBe('/how-it-works#own-llm');
  });

  it('saves a new default', async () => {
    await setup(out());
    stub.setDefault.mockImplementation(() => {
      stub.current.set(out({ default: 'own' }));
      return of(out({ default: 'own' }));
    });
    radio('own').click();
    fixture.detectChanges();

    expect(stub.setDefault).toHaveBeenCalledWith('own');
    expect(el().querySelector('.settings__status')?.textContent).toContain('Saved');
  });

  it('shows the backend error when saving fails', async () => {
    await setup(out());
    stub.setDefault.mockReturnValue(throwError(() => ({ error: { detail: "Our own LLM isn't installed on this machine" } })));
    radio('own').click();
    fixture.detectChanges();

    expect(el().querySelector('[role="alert"]')?.textContent).toContain("isn't installed");
  });
});
