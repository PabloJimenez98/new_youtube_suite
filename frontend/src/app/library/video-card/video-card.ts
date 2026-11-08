import { DatePipe } from '@angular/common';
import { Component, computed, input } from '@angular/core';
import { RouterLink } from '@angular/router';

import { displayName, formatDuration } from '../../explain/pipeline';
import { UploadOut } from '../../uploads/upload.model';

/**
 * A library card with the episode's YouTube thumbnail (docs/decisions/0022 §1).
 *
 * Not a <video> frame: each one opened a stream of the whole episode file,
 * and a dozen of them took every browser connection to the host, so the
 * API calls queued behind them and the app looked frozen (0023 §5).
 */
@Component({
  selector: 'app-video-card',
  imports: [DatePipe, RouterLink],
  templateUrl: './video-card.html',
  styleUrl: './video-card.scss',
})
export class VideoCard {
  readonly upload = input.required<UploadOut>();

  protected readonly name = computed(() => displayName(this.upload().filename));
  protected readonly durationLabel = computed(() => formatDuration(this.upload().duration_seconds));
  protected readonly thumbnail = computed(() => this.upload().youtube?.thumbnail_url ?? null);
}
