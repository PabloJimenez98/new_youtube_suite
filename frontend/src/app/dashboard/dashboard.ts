import { Component, viewChild } from '@angular/core';

import { Upload } from '../uploads/upload/upload';
import { UploadsList } from '../uploads/uploads-list/uploads-list';

@Component({
  selector: 'app-dashboard',
  imports: [Upload, UploadsList],
  templateUrl: './dashboard.html',
  styleUrl: './dashboard.scss',
})
export class Dashboard {
  private readonly uploadsList = viewChild.required(UploadsList);

  protected onUploaded(): void {
    this.uploadsList().refresh();
  }
}
