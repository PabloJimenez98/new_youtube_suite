import { ComponentFixture, TestBed } from '@angular/core/testing';

import { UploadOut } from '../../uploads/upload.model';
import { DeleteConfirmDialog } from './delete-confirm-dialog';

const upload: UploadOut = {
  id: 1,
  filename: 'episode.mp4',
  storage_path: 'x',
  duration_seconds: null,
  uploaded_at: '2026-08-14T12:00:00Z',
  status: 'uploaded',
};

describe('DeleteConfirmDialog', () => {
  let fixture: ComponentFixture<DeleteConfirmDialog>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [DeleteConfirmDialog],
    }).compileComponents();
    fixture = TestBed.createComponent(DeleteConfirmDialog);
  });

  it('renders nothing when upload is null', () => {
    fixture.detectChanges();
    expect((fixture.nativeElement as HTMLElement).querySelector('.dialog-backdrop')).toBeNull();
  });

  it('renders the filename and emits confirmed/cancelled', () => {
    fixture.componentRef.setInput('upload', upload);
    fixture.detectChanges();

    const el = fixture.nativeElement as HTMLElement;
    expect(el.textContent).toContain('episode.mp4');

    const confirmedSpy = vi.fn();
    const cancelledSpy = vi.fn();
    fixture.componentInstance.confirmed.subscribe(confirmedSpy);
    fixture.componentInstance.cancelled.subscribe(cancelledSpy);

    el.querySelector<HTMLButtonElement>('.btn--danger')!.click();
    expect(confirmedSpy).toHaveBeenCalled();

    el.querySelector<HTMLButtonElement>('.btn--ghost')!.click();
    expect(cancelledSpy).toHaveBeenCalled();
  });
});
