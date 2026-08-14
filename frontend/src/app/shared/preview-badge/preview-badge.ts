import { Component, input } from '@angular/core';

/** Small pill marking a widget's data as placeholder/dummy content. */
@Component({
  selector: 'app-preview-badge',
  imports: [],
  templateUrl: './preview-badge.html',
  styleUrl: './preview-badge.scss',
})
export class PreviewBadge {
  readonly label = input('Preview');
}
