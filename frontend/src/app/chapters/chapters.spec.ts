import { provideHttpClient } from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { TestBed } from '@angular/core/testing';

import { ChapterOut, Chapters, formatTimestamp, toYouTubeText } from './chapters';

const ch = (start_seconds: number, title: string): ChapterOut => ({
  order_index: 0,
  start_seconds,
  title,
  model_version: 'v',
});

describe('Chapters', () => {
  it('getChapters GETs /uploads/{id}/chapters', () => {
    TestBed.configureTestingModule({ providers: [provideHttpClient(), provideHttpClientTesting()] });
    const service = TestBed.inject(Chapters);
    const httpMock = TestBed.inject(HttpTestingController);
    let received: ChapterOut[] | undefined;
    service.getChapters(5).subscribe((c) => (received = c));
    const req = httpMock.expectOne('/uploads/5/chapters');
    expect(req.request.method).toBe('GET');
    req.flush([ch(0, 'Intro')]);
    expect(received).toEqual([ch(0, 'Intro')]);
    httpMock.verify();
  });

  it.each([
    [0, false, '00:00'],
    [75.9, false, '01:15'],
    [3599, false, '59:59'],
    [0, true, '0:00:00'],
    [3725, true, '1:02:05'],
  ])('formatTimestamp(%s, %s) = %s', (s, h, out) => {
    expect(formatTimestamp(s, h)).toBe(out);
  });

  it('toYouTubeText uses MM:SS under an hour', () => {
    expect(toYouTubeText([ch(0, 'Intro'), ch(605, 'Main')])).toBe('00:00 Intro\n10:05 Main');
  });

  it('toYouTubeText switches every line to H:MM:SS past an hour', () => {
    expect(toYouTubeText([ch(0, 'Intro'), ch(3725, 'Late')])).toBe('0:00:00 Intro\n1:02:05 Late');
  });
});
