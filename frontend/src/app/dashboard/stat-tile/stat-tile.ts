import { Component, input } from '@angular/core';

import { PreviewBadge } from '../../shared/preview-badge/preview-badge';

@Component({
  selector: 'app-stat-tile',
  imports: [PreviewBadge],
  templateUrl: './stat-tile.html',
  styleUrl: './stat-tile.scss',
})
export class StatTile {
  readonly label = input.required<string>();
  readonly value = input.required<string>();
  readonly hint = input<string | null>(null);
  /** True for dummy/placeholder tiles -- gets a visible "Preview" badge. */
  readonly preview = input(false);
}
