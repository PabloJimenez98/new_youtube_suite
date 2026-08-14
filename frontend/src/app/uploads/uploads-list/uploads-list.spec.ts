import { ComponentFixture, TestBed } from '@angular/core/testing';
import { of } from 'rxjs';

import { UploadOut } from '../upload.model';
import { Uploads } from '../uploads';
import { UploadsList } from './uploads-list';

const sample: UploadOut = {
  id: 1,
  filename: 'episode.mp4',
  storage_path: 'abc_episode.mp4',
  duration_seconds: null,
  uploaded_at: '2026-08-14T12:00:00Z',
  status: 'uploaded',
};

describe('UploadsList', () => {
  let component: UploadsList;
  let fixture: ComponentFixture<UploadsList>;
  let uploadsStub: { listUploads: ReturnType<typeof vi.fn> };

  beforeEach(async () => {
    uploadsStub = { listUploads: vi.fn().mockReturnValue(of([])) };

    await TestBed.configureTestingModule({
      imports: [UploadsList],
      providers: [{ provide: Uploads, useValue: uploadsStub }],
    }).compileComponents();

    fixture = TestBed.createComponent(UploadsList);
    component = fixture.componentInstance;
  });

  it('should create', () => {
    fixture.detectChanges();
    expect(component).toBeTruthy();
  });

  it('shows an empty state when there are no uploads', () => {
    fixture.detectChanges();
    const text = (fixture.nativeElement as HTMLElement).textContent ?? '';
    expect(text).toContain('No uploads yet');
  });

  it('renders fetched items', () => {
    uploadsStub.listUploads.mockReturnValue(of([sample]));
    fixture.detectChanges();

    const text = (fixture.nativeElement as HTMLElement).textContent ?? '';
    expect(text).toContain('episode.mp4');
  });

  it('refresh() re-fetches and updates the rendered list', () => {
    fixture.detectChanges();
    expect(component['items']()).toEqual([]);

    uploadsStub.listUploads.mockReturnValue(of([sample]));
    component.refresh();

    expect(component['items']()).toEqual([sample]);
  });
});
