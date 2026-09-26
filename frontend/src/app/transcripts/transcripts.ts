import { HttpClient } from '@angular/common/http';
import { Service, inject } from '@angular/core';
import { Observable } from 'rxjs';

export type SubtitleFormat = 'srt' | 'vtt';

export const SUBTITLE_FORMATS: readonly SubtitleFormat[] = ['srt', 'vtt'];

@Service()
export class Transcripts {
  private readonly http = inject(HttpClient);

  /**
   * GET /uploads/{id}/transcript.{srt,vtt}. Fetched as a blob through
   * HttpClient (so authInterceptor attaches the token) rather than a
   * plain link -- see docs/decisions/0007-transcription-pipeline.md §6.
   */
  getSubtitle(uploadId: number, format: SubtitleFormat): Observable<Blob> {
    return this.http.get(`/uploads/${uploadId}/transcript.${format}`, {
      responseType: 'blob',
    });
  }
}

/** Hand a blob to the browser as a file download. */
export function saveBlob(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  link.click();
  URL.revokeObjectURL(url);
}
