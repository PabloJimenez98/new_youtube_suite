import { provideHttpClient } from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { TestBed } from '@angular/core/testing';

import { LlmSettings, LlmSettingsOut, llmLabel } from './llm-settings';

const out: LlmSettingsOut = { default: 'qwen', own_available: true, own_label: 'yts-110m-a3', qwen_label: 'Qwen3.5-9B-Q4_K_M' };

describe('LlmSettings', () => {
  let service: LlmSettings;
  let httpMock: HttpTestingController;

  beforeEach(() => {
    TestBed.configureTestingModule({ providers: [provideHttpClient(), provideHttpClientTesting()] });
    service = TestBed.inject(LlmSettings);
    httpMock = TestBed.inject(HttpTestingController);
  });

  afterEach(() => httpMock.verify());

  it('load GETs /settings/llm and keeps the result in current', () => {
    service.load().subscribe();
    const req = httpMock.expectOne('/settings/llm');
    expect(req.request.method).toBe('GET');
    req.flush(out);
    expect(service.current()).toEqual(out);
  });

  it('setDefault PUTs { default } and updates current', () => {
    service.setDefault('own').subscribe();
    const req = httpMock.expectOne('/settings/llm');
    expect(req.request.method).toBe('PUT');
    expect(req.request.body).toEqual({ default: 'own' });
    req.flush({ ...out, default: 'own' });
    expect(service.current()?.default).toBe('own');
  });
});

describe('llmLabel', () => {
  it('names the model when the backend knows it', () => {
    expect(llmLabel('qwen', out)).toBe('Qwen (Qwen3.5-9B-Q4_K_M)');
    expect(llmLabel('own', out)).toBe('Our model (yts-110m-a3)');
  });

  it('falls back to a generic name', () => {
    expect(llmLabel('qwen', null)).toBe('Qwen');
    expect(llmLabel('own', { ...out, own_label: null })).toBe('Our model');
  });
});
