import { ComponentFixture, TestBed } from '@angular/core/testing';
import { of } from 'rxjs';

import { Uploads } from '../uploads/uploads';
import { Dashboard } from './dashboard';

describe('Dashboard', () => {
  let component: Dashboard;
  let fixture: ComponentFixture<Dashboard>;
  let uploadsStub: { listUploads: ReturnType<typeof vi.fn>; uploadFile: ReturnType<typeof vi.fn> };

  beforeEach(async () => {
    uploadsStub = {
      listUploads: vi.fn().mockReturnValue(of([])),
      uploadFile: vi.fn(),
    };

    await TestBed.configureTestingModule({
      imports: [Dashboard],
      providers: [{ provide: Uploads, useValue: uploadsStub }],
    }).compileComponents();

    fixture = TestBed.createComponent(Dashboard);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });

  it('refreshes the uploads list when the upload component emits `uploaded`', () => {
    expect(uploadsStub.listUploads).toHaveBeenCalledTimes(1);

    component['onUploaded']();

    expect(uploadsStub.listUploads).toHaveBeenCalledTimes(2);
  });
});
