import { Component, input, output } from '@angular/core';

import { UploadOut } from '../../uploads/upload.model';

@Component({
  selector: 'app-delete-confirm-dialog',
  imports: [],
  templateUrl: './delete-confirm-dialog.html',
  styleUrl: './delete-confirm-dialog.scss',
})
export class DeleteConfirmDialog {
  /** null = closed; a video = open, confirming deletion of that video. */
  readonly upload = input<UploadOut | null>(null);
  readonly confirmed = output<void>();
  readonly cancelled = output<void>();
}
