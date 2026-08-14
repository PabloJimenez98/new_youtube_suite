import { HttpEventType } from '@angular/common/http';
import { Component, inject, output, signal } from '@angular/core';

import { Uploads } from '../uploads';

type UploadState = 'idle' | 'selecting' | 'uploading' | 'success' | 'error';

const MAX_FILE_SIZE_BYTES = 2 * 1024 ** 3; // 2 GB -- see docs/decisions/0003-frontend-skeleton.md

@Component({
  selector: 'app-upload',
  imports: [],
  templateUrl: './upload.html',
  styleUrl: './upload.scss',
})
export class Upload {
  private readonly uploads = inject(Uploads);

  /** Emitted after a successful upload so the parent can refresh the list. */
  readonly uploaded = output<void>();

  protected readonly state = signal<UploadState>('idle');
  protected readonly progress = signal(0);
  protected readonly errorMessage = signal<string | null>(null);
  protected selectedFile: File | null = null;

  protected onFileSelected(event: Event): void {
    const input = event.target as HTMLInputElement;
    const file = input.files?.[0] ?? null;
    this.errorMessage.set(null);

    if (!file) {
      this.selectedFile = null;
      this.state.set('idle');
      return;
    }

    if (!file.type.startsWith('video/')) {
      this.selectedFile = null;
      this.state.set('error');
      this.errorMessage.set('Please choose a video file.');
      return;
    }

    if (file.size > MAX_FILE_SIZE_BYTES) {
      this.selectedFile = null;
      this.state.set('error');
      this.errorMessage.set('That file is larger than the 2 GB limit.');
      return;
    }

    this.selectedFile = file;
    this.state.set('selecting');
  }

  protected submit(): void {
    if (!this.selectedFile || this.state() === 'uploading') {
      return;
    }

    this.state.set('uploading');
    this.progress.set(0);
    this.errorMessage.set(null);

    this.uploads.uploadFile(this.selectedFile).subscribe({
      next: (event) => {
        if (event.type === HttpEventType.UploadProgress && event.total) {
          this.progress.set(Math.round((100 * event.loaded) / event.total));
        } else if (event.type === HttpEventType.Response) {
          this.state.set('success');
          this.selectedFile = null;
          this.uploaded.emit();
        }
      },
      error: (err) => {
        this.state.set('error');
        this.errorMessage.set(
          err?.error?.detail ?? 'Upload failed. Please try again.',
        );
        // Keep selectedFile so the user can retry without reselecting.
      },
    });
  }

  protected reset(): void {
    this.selectedFile = null;
    this.progress.set(0);
    this.errorMessage.set(null);
    this.state.set('idle');
  }
}
