import { provideHttpClient } from '@angular/common/http';
import {
  HttpTestingController,
  provideHttpClientTesting,
} from '@angular/common/http/testing';
import { TestBed } from '@angular/core/testing';

import { Uploads } from './uploads';

describe('Uploads', () => {
  let service: Uploads;
  let httpMock: HttpTestingController;

  beforeEach(() => {
    TestBed.configureTestingModule({
      providers: [provideHttpClient(), provideHttpClientTesting()],
    });
    service = TestBed.inject(Uploads);
    httpMock = TestBed.inject(HttpTestingController);
  });

  afterEach(() => {
    httpMock.verify();
  });

  it('should be created', () => {
    expect(service).toBeTruthy();
  });

  it('uploadFile POSTs multipart/form-data with the file under the `file` key', () => {
    const file = new File([new Uint8Array(4)], 'clip.mp4', { type: 'video/mp4' });

    service.uploadFile(file).subscribe();

    const req = httpMock.expectOne('/uploads');
    expect(req.request.method).toBe('POST');
    expect(req.request.body instanceof FormData).toBe(true);
    expect((req.request.body as FormData).get('file')).toBe(file);

    req.flush({
      id: 1,
      filename: 'clip.mp4',
      storage_path: 'x_clip.mp4',
      duration_seconds: null,
      uploaded_at: '2026-08-14T12:00:00Z',
      status: 'uploaded',
    });
  });

  it('listUploads GETs /uploads', () => {
    service.listUploads().subscribe();

    const req = httpMock.expectOne('/uploads');
    expect(req.request.method).toBe('GET');
    req.flush([]);
  });
});
