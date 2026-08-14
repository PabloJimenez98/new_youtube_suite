import { DatePipe } from '@angular/common';
import { Component, OnInit, inject, signal } from '@angular/core';

import { UploadOut } from '../upload.model';
import { Uploads } from '../uploads';

@Component({
  selector: 'app-uploads-list',
  imports: [DatePipe],
  templateUrl: './uploads-list.html',
  styleUrl: './uploads-list.scss',
})
export class UploadsList implements OnInit {
  private readonly uploads = inject(Uploads);

  protected readonly items = signal<UploadOut[]>([]);
  protected readonly loading = signal(false);
  protected readonly errorMessage = signal<string | null>(null);

  ngOnInit(): void {
    this.refresh();
  }

  /** Public so the dashboard can trigger it after a successful upload. */
  refresh(): void {
    this.loading.set(true);
    this.errorMessage.set(null);

    this.uploads.listUploads().subscribe({
      next: (items) => {
        this.items.set(items);
        this.loading.set(false);
      },
      error: () => {
        this.errorMessage.set('Could not load uploads.');
        this.loading.set(false);
      },
    });
  }
}
