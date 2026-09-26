import { provideHttpClient } from '@angular/common/http';
import {
  HttpTestingController,
  provideHttpClientTesting,
} from '@angular/common/http/testing';
import { TestBed } from '@angular/core/testing';

import { Transcripts } from './transcripts';

describe('Transcripts', () => {
  let service: Transcripts;
  let httpMock: HttpTestingController;

  beforeEach(() => {
    TestBed.configureTestingModule({
      providers: [provideHttpClient(), provideHttpClientTesting()],
    });
    service = TestBed.inject(Transcripts);
    httpMock = TestBed.inject(HttpTestingController);
  });

  afterEach(() => {
    httpMock.verify();
  });

  it.each(['srt', 'vtt'] as const)('getSubtitle GETs /uploads/{id}/transcript.%s as a blob', (format) => {
    let received: Blob | undefined;
    service.getSubtitle(5, format).subscribe((blob) => (received = blob));

    const req = httpMock.expectOne(`/uploads/5/transcript.${format}`);
    expect(req.request.method).toBe('GET');
    expect(req.request.responseType).toBe('blob');

    const body = new Blob(['WEBVTT\n\n']);
    req.flush(body);
    expect(received).toBe(body);
  });
});
