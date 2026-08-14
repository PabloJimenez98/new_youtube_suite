import { Component, input } from '@angular/core';

/** Page-wide banner marking an entire page's content as preview/dummy. */
@Component({
  selector: 'app-preview-banner',
  imports: [],
  templateUrl: './preview-banner.html',
  styleUrl: './preview-banner.scss',
})
export class PreviewBanner {
  readonly message = input.required<string>();
}
