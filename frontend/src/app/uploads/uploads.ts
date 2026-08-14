import { HttpClient, HttpEvent, HttpRequest } from '@angular/common/http';
import { Service, inject } from '@angular/core';
import { Observable } from 'rxjs';

import { UploadOut } from './upload.model';

@Service()
export class Uploads {
  private readonly http = inject(HttpClient);

  /**
   * POST /uploads (multipart/form-data, file field literally named
   * `file`). Returns a raw event stream so the caller can subscribe to
   * upload-progress events as well as the final response.
   */
  uploadFile(file: File): Observable<HttpEvent<UploadOut>> {
    const formData = new FormData();
    formData.append('file', file);

    const req = new HttpRequest<FormData>('POST', '/uploads', formData, {
      reportProgress: true,
    });
    return this.http.request<UploadOut>(req);
  }

  listUploads(): Observable<UploadOut[]> {
    return this.http.get<UploadOut[]>('/uploads');
  }
}
