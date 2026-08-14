import { DatePipe } from '@angular/common';
import { Component, computed, input } from '@angular/core';

import { UploadOut } from '../../uploads/upload.model';

@Component({
  selector: 'app-video-card',
  imports: [DatePipe],
  templateUrl: './video-card.html',
  styleUrl: './video-card.scss',
})
export class VideoCard {
  readonly upload = input.required<UploadOut>();

  protected readonly durationLabel = computed(() => {
    const seconds = this.upload().duration_seconds;
    if (seconds == null) {
      return null;
    }
    const minutes = Math.floor(seconds / 60);
    const remaining = Math.round(seconds % 60);
    return `${minutes}:${remaining.toString().padStart(2, '0')}`;
  });
}
