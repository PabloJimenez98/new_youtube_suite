import { HttpEventType } from '@angular/common/http';
import { ComponentFixture, TestBed } from '@angular/core/testing';
import { Subject, throwError } from 'rxjs';

import { Uploads } from '../uploads';
import { Upload } from './upload';

function makeFile(name: string, type: string, sizeBytes: number): File {
  return new File([new Uint8Array(sizeBytes)], name, { type });
}

describe('Upload', () => {
  let component: Upload;
  let fixture: ComponentFixture<Upload>;
  let uploadsStub: { uploadFile: ReturnType<typeof vi.fn> };

  beforeEach(async () => {
    uploadsStub = { uploadFile: vi.fn() };

    await TestBed.configureTestingModule({
      imports: [Upload],
      providers: [{ provide: Uploads, useValue: uploadsStub }],
    }).compileComponents();

    fixture = TestBed.createComponent(Upload);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  function selectFile(file: File | null): void {
    const input = { files: file ? [file] : [] } as unknown as HTMLInputElement;
    component['onFileSelected']({ target: input } as unknown as Event);
  }

  it('starts idle with submit effectively disabled (no file selected)', () => {
    expect(component['state']()).toBe('idle');
    const button: HTMLButtonElement = fixture.nativeElement.querySelector('button');
    expect(button.disabled).toBe(true);
  });

  it('rejects a non-video file without calling the service', () => {
    selectFile(makeFile('notes.txt', 'text/plain', 10));

    expect(component['state']()).toBe('error');
    expect(component['errorMessage']()).toContain('video file');
    expect(uploadsStub.uploadFile).not.toHaveBeenCalled();
  });

  it('rejects an oversized file without calling the service', () => {
    selectFile(makeFile('huge.mp4', 'video/mp4', 2 * 1024 ** 3 + 1));

    expect(component['state']()).toBe('error');
    expect(component['errorMessage']()).toContain('2 GB');
    expect(uploadsStub.uploadFile).not.toHaveBeenCalled();
  });

  it('accepts a valid file and moves to the selecting state', () => {
    selectFile(makeFile('episode.mp4', 'video/mp4', 1024));

    expect(component['state']()).toBe('selecting');
  });

  it('renders progress and reaches success, emitting `uploaded`', () => {
    const events$ = new Subject<unknown>();
    uploadsStub.uploadFile.mockReturnValue(events$.asObservable());
    selectFile(makeFile('episode.mp4', 'video/mp4', 1024));

    const uploadedSpy = vi.fn();
    component.uploaded.subscribe(uploadedSpy);

    component['submit']();
    expect(component['state']()).toBe('uploading');

    events$.next({ type: HttpEventType.UploadProgress, loaded: 50, total: 100 });
    expect(component['progress']()).toBe(50);

    events$.next({ type: HttpEventType.Response, status: 201 });
    expect(component['state']()).toBe('success');
    expect(uploadedSpy).toHaveBeenCalledTimes(1);
  });

  it('shows an inline error and keeps the file selected for retry on failure', () => {
    uploadsStub.uploadFile.mockReturnValue(
      throwError(() => ({ error: { detail: 'Upload failed. Please try again.' } })),
    );
    selectFile(makeFile('episode.mp4', 'video/mp4', 1024));

    component['submit']();

    expect(component['state']()).toBe('error');
    expect(component['errorMessage']()).toBe('Upload failed. Please try again.');
    expect(component['selectedFile']).not.toBeNull();
  });
});
