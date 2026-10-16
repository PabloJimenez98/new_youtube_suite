import { provideHttpClient } from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { TestBed } from '@angular/core/testing';

import { ShowNotes, ShowNotesOut, parseTags } from './shownotes';

const notes: ShowNotesOut = {
  title_suggestions: ['A', 'B', 'C'],
  description: 'Desc.',
  tags: ['x'],
  summary: 'Sum.',
  language: 'en',
  llm_model_used: 'm+shownotes-v1',
  created_at: '2026-10-16T12:00:00Z',
  updated_at: '2026-10-16T12:00:00Z',
};

describe('ShowNotes', () => {
  let service: ShowNotes;
  let httpMock: HttpTestingController;

  beforeEach(() => {
    TestBed.configureTestingModule({ providers: [provideHttpClient(), provideHttpClientTesting()] });
    service = TestBed.inject(ShowNotes);
    httpMock = TestBed.inject(HttpTestingController);
  });

  afterEach(() => httpMock.verify());

  it('getShowNotes GETs /uploads/{id}/shownotes', () => {
    let received: ShowNotesOut | undefined;
    service.getShowNotes(5).subscribe((n) => (received = n));
    const req = httpMock.expectOne('/uploads/5/shownotes');
    expect(req.request.method).toBe('GET');
    req.flush(notes);
    expect(received).toEqual(notes);
  });

  it('updateShowNotes PATCHes only the given fields', () => {
    service.updateShowNotes(5, { summary: 'New.' }).subscribe();
    const req = httpMock.expectOne('/uploads/5/shownotes');
    expect(req.request.method).toBe('PATCH');
    expect(req.request.body).toEqual({ summary: 'New.' });
    req.flush({ ...notes, summary: 'New.' });
  });

  it('parseTags splits on commas and drops blanks', () => {
    expect(parseTags(' ai , deep learning,, ,podcast ')).toEqual(['ai', 'deep learning', 'podcast']);
    expect(parseTags('')).toEqual([]);
  });
});
