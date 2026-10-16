import { provideHttpClient } from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { TestBed } from '@angular/core/testing';

import { Relevance, RelevanceOut } from './relevance';

describe('Relevance', () => {
  it('getRelevance GETs /uploads/{id}/relevance', () => {
    TestBed.configureTestingModule({ providers: [provideHttpClient(), provideHttpClientTesting()] });
    const service = TestBed.inject(Relevance);
    const httpMock = TestBed.inject(HttpTestingController);
    const body: RelevanceOut = { version: 'v', windows: [], top_moments: [] };
    let received: RelevanceOut | undefined;
    service.getRelevance(5).subscribe((r) => (received = r));
    const req = httpMock.expectOne('/uploads/5/relevance');
    expect(req.request.method).toBe('GET');
    req.flush(body);
    expect(received).toEqual(body);
    httpMock.verify();
  });
});
