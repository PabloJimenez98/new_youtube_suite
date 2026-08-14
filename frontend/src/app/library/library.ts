import { Component, OnInit, inject, signal } from '@angular/core';

import { Uploads } from '../uploads/uploads';
import { UploadOut } from '../uploads/upload.model';
import { VideoCard } from './video-card/video-card';

@Component({
  selector: 'app-library',
  imports: [VideoCard],
  templateUrl: './library.html',
  styleUrl: './library.scss',
})
export class Library implements OnInit {
  private readonly uploads = inject(Uploads);

  protected readonly videos = signal<UploadOut[]>([]);
  protected readonly loading = signal(true);
  protected readonly errorMessage = signal<string | null>(null);

  ngOnInit(): void {
    this.uploads.listUploads().subscribe({
      next: (videos) => {
        this.videos.set(videos);
        this.loading.set(false);
      },
      error: () => {
        this.errorMessage.set('Could not load your videos.');
        this.loading.set(false);
      },
    });
  }
}
