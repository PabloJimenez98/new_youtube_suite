import { Component, computed, input, output } from '@angular/core';

import { formatTimestamp } from '../../chapters/chapters';
import { ClipOut, clipFileUrl } from '../../shorts/shorts';

/**
 * The rendered shorts of one episode: a player per clip, its source time
 * range and score, a download link, and "Show in episode", which asks the
 * parent to seek its own player there. See docs/decisions/0013 §8.
 */
@Component({
  selector: 'app-shorts-panel',
  imports: [],
  templateUrl: './shorts-panel.html',
  styleUrl: './shorts-panel.scss',
})
export class ShortsPanel {
  readonly clips = input.required<readonly ClipOut[]>();
  readonly token = input.required<string>();
  readonly seek = output<number>();

  protected readonly withHours = computed(() => this.clips().some((c) => c.end >= 3600));
  protected readonly formatTimestamp = formatTimestamp;

  protected videoUrl(clip: ClipOut): string {
    return clipFileUrl(clip.id, this.token());
  }

  protected downloadUrl(clip: ClipOut): string {
    return clipFileUrl(clip.id, this.token(), true);
  }

  /** CSS aspect-ratio for the player box, e.g. "9 / 16". */
  protected ratio(clip: ClipOut): string {
    return clip.aspect_ratio.replace(':', ' / ');
  }
}
