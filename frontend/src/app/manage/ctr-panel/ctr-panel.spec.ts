import { provideHttpClient } from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { ComponentFixture, TestBed } from '@angular/core/testing';
import { of, throwError } from 'rxjs';

import { CandidateOut, Ctr, thumbnailUrl } from '../../ctr/ctr';
import { CtrPanel } from './ctr-panel';

function candidate(over: Partial<CandidateOut> = {}): CandidateOut {
  return {
    id: 1,
    upload_id: 7,
    candidate_title: 'How we built it',
    has_thumbnail: true,
    predicted_score: 82.5,
    partial: false,
    features: {
      version: 'heuristic-v0',
      title: { title_chars: 15, has_digit: false, has_question: true, caps_words: 0 },
      thumbnail: {
        face_count: 2,
        largest_face_area: 0.2,
        text_area: 0.04,
        brightness: 0.5,
        contrast: 0.4,
        saturation: 0.3,
        aspect_ratio: 1.7778,
      },
      sub_scores: { title_length: 0.29, faces: 1, face_size: 1, text_coverage: 1 },
      weights: { title_length: 0.35, faces: 0.3, face_size: 0.2, text_coverage: 0.15 },
    },
    source: 'manual',
    scored_at: '2026-11-08T12:00:00Z',
    group: 'full',
    rank: 1,
    ...over,
  };
}

describe('Ctr service', () => {
  it('calls the candidate endpoints with multipart bodies', () => {
    TestBed.configureTestingModule({
      providers: [provideHttpClient(), provideHttpClientTesting()],
    });
    const service = TestBed.inject(Ctr);
    const http = TestBed.inject(HttpTestingController);
    const file = new File([new Uint8Array([1])], 't.png', { type: 'image/png' });

    service.list(7).subscribe();
    service.create(7, 'A title', file).subscribe();
    service.fromShowNotes(7).subscribe();
    service.update(3, { title: 'B', removeThumbnail: true }).subscribe();
    service.remove(3).subscribe();

    http.expectOne({ url: '/uploads/7/ctr-candidates', method: 'GET' });
    const create = http.expectOne({ url: '/uploads/7/ctr-candidates', method: 'POST' });
    const body = create.request.body as FormData;
    expect(body.get('title')).toBe('A title');
    expect(body.get('thumbnail')).toBe(file);
    http.expectOne({ url: '/uploads/7/ctr-candidates/from-show-notes', method: 'POST' });
    const patch = http.expectOne({ url: '/ctr-candidates/3', method: 'PATCH' });
    expect((patch.request.body as FormData).get('remove_thumbnail')).toBe('true');
    http.expectOne({ url: '/ctr-candidates/3', method: 'DELETE' });
    http.verify();
  });

  it('builds the thumbnail URL with the token', () => {
    expect(thumbnailUrl(3, 'a b')).toBe('/ctr-candidates/3/thumbnail?token=a%20b');
    expect(thumbnailUrl(3, 't', '2026')).toContain('&v=2026');
  });
});

describe('CtrPanel', () => {
  let fixture: ComponentFixture<CtrPanel>;
  let stub: Record<
    'list' | 'create' | 'fromShowNotes' | 'update' | 'remove',
    ReturnType<typeof vi.fn>
  >;
  let counts: number[];
  const el = () => fixture.nativeElement as HTMLElement;
  const text = () => el().textContent!.replace(/\s+/g, ' ');

  async function setup(list: CandidateOut[], opts: { hasShowNotes?: boolean } = {}): Promise<void> {
    stub = {
      list: vi.fn().mockReturnValue(of(list)),
      create: vi.fn().mockReturnValue(of(candidate())),
      fromShowNotes: vi
        .fn()
        .mockReturnValue(
          of([
            candidate({
              source: 'show_notes',
              group: 'unscored',
              predicted_score: null,
              rank: null,
            }),
          ]),
        ),
      update: vi.fn().mockReturnValue(of(candidate())),
      remove: vi.fn().mockReturnValue(of(undefined)),
    };
    await TestBed.configureTestingModule({
      imports: [CtrPanel],
      providers: [{ provide: Ctr, useValue: stub }],
    }).compileComponents();
    fixture = TestBed.createComponent(CtrPanel);
    fixture.componentRef.setInput('uploadId', 7);
    fixture.componentRef.setInput('token', 'tok');
    fixture.componentRef.setInput('hasShowNotes', opts.hasShowNotes ?? false);
    counts = [];
    fixture.componentInstance.candidateCount.subscribe((n) => counts.push(n));
    fixture.detectChanges();
    await fixture.whenStable();
    fixture.detectChanges();
  }

  it('empty: explains how to start and reports zero candidates', async () => {
    await setup([]);
    expect(text()).toContain('No candidates yet');
    expect(counts).toEqual([0]);
    expect(el().querySelector('.ctr-panel__from-notes')).toBeNull();
  });

  it('shows the ranking with rank, score and the not-a-CTR label', async () => {
    await setup([
      candidate(),
      candidate({ id: 2, rank: 2, predicted_score: 60, candidate_title: 'Second' }),
    ]);
    expect(text()).toContain('not a CTR %');
    const ranks = [...el().querySelectorAll('.ctr-panel__rank')].map((n) => n.textContent);
    expect(ranks).toEqual(['#1', '#2']);
    expect(el().querySelector('img')!.getAttribute('src')).toContain(
      '/ctr-candidates/1/thumbnail?token=tok',
    );
    expect(counts).toEqual([2]);
  });

  it('badges partial and unscored candidates, and shows thumbnail errors', async () => {
    await setup([
      candidate({ partial: true, group: 'partial', has_thumbnail: false }),
      candidate({ id: 2, predicted_score: null, group: 'unscored', rank: null }),
      candidate({
        id: 3,
        predicted_score: null,
        group: 'unscored',
        rank: null,
        features: { version: 'heuristic-v0', error: 'thumbnail unreadable: x' },
      }),
    ]);
    expect(text()).toContain('title only');
    expect(text()).toContain('not scored');
    expect(text()).toContain('thumbnail unreadable');
    expect(text()).toContain('run CTR to score them');
    expect(text()).toContain('No thumbnail');
  });

  it('expands the breakdown: weighted sub-scores and the logged features', async () => {
    await setup([candidate()]);
    (el().querySelector('.ctr-panel__why') as HTMLButtonElement).click();
    fixture.detectChanges();
    const breakdown = el().querySelector('.ctr-panel__breakdown')!.textContent!;
    expect(breakdown).toContain('Faces');
    expect(breakdown).toContain('1.00 × 0.3');
    expect(breakdown).toContain('question: yes');
    expect(breakdown).toContain('text area: 4 %');
    expect(breakdown).not.toContain('aspect ratio'); // 16:9 needs no warning
  });

  it('adds a candidate with the title counter, then refetches', async () => {
    await setup([]);
    const input = el().querySelector('.ctr-panel__new-title') as HTMLInputElement;
    input.value = '  Fresh title ';
    input.dispatchEvent(new Event('input'));
    fixture.detectChanges();
    expect(el().querySelector('.ctr-panel__add .ctr-panel__count')!.textContent).toContain(
      '11/100',
    );
    (el().querySelector('.ctr-panel__add-btn') as HTMLButtonElement).click();
    expect(stub.create).toHaveBeenCalledWith(7, 'Fresh title', null);
    expect(stub.list).toHaveBeenCalledTimes(2);
  });

  it('shows the backend reason when adding fails', async () => {
    await setup([]);
    stub.create.mockReturnValue(
      throwError(() => ({
        error: { detail: 'thumbnail: only JPEG, PNG and WebP images are accepted' },
      })),
    );
    const input = el().querySelector('.ctr-panel__new-title') as HTMLInputElement;
    input.value = 'x';
    input.dispatchEvent(new Event('input'));
    fixture.detectChanges();
    (el().querySelector('.ctr-panel__add-btn') as HTMLButtonElement).click();
    fixture.detectChanges();
    expect(el().querySelector('[role="alert"]')!.textContent).toContain('only JPEG, PNG and WebP');
  });

  it('takes titles from the show notes when a draft exists', async () => {
    await setup([], { hasShowNotes: true });
    (el().querySelector('.ctr-panel__from-notes') as HTMLButtonElement).click();
    fixture.detectChanges();
    expect(stub.fromShowNotes).toHaveBeenCalledWith(7);
    expect(text()).toContain('from show notes');
    expect(counts).toEqual([0, 1]);
  });

  it('edits a title and deletes a candidate', async () => {
    await setup([candidate()]);
    (el().querySelector('.ctr-panel__edit-btn') as HTMLButtonElement).click();
    fixture.detectChanges();
    const input = el().querySelector('.ctr-panel__edit input') as HTMLInputElement;
    input.value = 'Better title';
    input.dispatchEvent(new Event('input'));
    (el().querySelector('.ctr-panel__save') as HTMLButtonElement).click();
    expect(stub.update).toHaveBeenCalledWith(1, { title: 'Better title' });

    (el().querySelector('.ctr-panel__delete') as HTMLButtonElement).click();
    expect(stub.remove).toHaveBeenCalledWith(1);
  });

  it('refetches when a newer ctr job succeeds', async () => {
    await setup([]);
    fixture.componentRef.setInput('scoredJobId', 42);
    fixture.detectChanges();
    await fixture.whenStable();
    expect(stub.list).toHaveBeenCalledTimes(2);
  });

  it('caps the list at ten candidates', async () => {
    await setup(Array.from({ length: 10 }, (_, i) => candidate({ id: i + 1 })));
    expect(text()).toContain('At most 10 candidates');
    expect((el().querySelector('.ctr-panel__new-title') as HTMLInputElement).disabled).toBe(true);
  });
});
