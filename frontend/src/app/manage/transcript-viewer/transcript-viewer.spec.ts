import { ComponentFixture, TestBed } from '@angular/core/testing';
import { of, throwError } from 'rxjs';

import { TranscriptOut, Transcripts } from '../../transcripts/transcripts';
import { TranscriptViewer, groupTurns, talkShare } from './transcript-viewer';

const seg = (start: number, end: number, text: string, speaker: string | null = null) => ({ start, end, text, speaker });

const DIARIZED: TranscriptOut = {
  speakers: ['SPEAKER_00', 'SPEAKER_01'],
  segments: [
    seg(0, 4, 'Hola a todos.', 'SPEAKER_00'),
    seg(4, 6, 'Bienvenidos.', 'SPEAKER_00'),
    seg(6, 8, '¿Qué tal?', 'SPEAKER_01'),
    seg(3700, 3702, 'Adiós.', 'SPEAKER_00'),
  ],
};

describe('transcript grouping', () => {
  it('merges consecutive segments of the same speaker into turns', () => {
    const turns = groupTurns(DIARIZED.segments, true);
    expect(turns.map((t) => [t.speaker, t.start, t.text])).toEqual([
      ['SPEAKER_00', 0, 'Hola a todos. Bienvenidos.'],
      ['SPEAKER_01', 6, '¿Qué tal?'],
      ['SPEAKER_00', 3700, 'Adiós.'],
    ]);
  });

  it('without speakers, breaks paragraphs every minute', () => {
    const turns = groupTurns([seg(0, 5, 'a'), seg(30, 35, 'b'), seg(61, 65, 'c')], false);
    expect(turns.map((t) => t.text)).toEqual(['a b', 'c']);
    expect(turns[0].speaker).toBeNull();
  });

  it('computes each speaker’s share of speaking time', () => {
    const share = talkShare(DIARIZED.segments, DIARIZED.speakers);
    expect(share.map((s) => s.seconds)).toEqual([8, 2]);
    expect(share[0].share).toBeCloseTo(0.8);
  });
});

describe('TranscriptViewer', () => {
  let fixture: ComponentFixture<TranscriptViewer>;
  let getTranscript: ReturnType<typeof vi.fn>;
  const el = () => fixture.nativeElement as HTMLElement;

  async function setup(response = of(DIARIZED)) {
    getTranscript = vi.fn().mockReturnValue(response);
    await TestBed.configureTestingModule({
      imports: [TranscriptViewer],
      providers: [{ provide: Transcripts, useValue: { getTranscript } }],
    }).compileComponents();
    fixture = TestBed.createComponent(TranscriptViewer);
    fixture.componentRef.setInput('uploadId', 7);
    fixture.componentRef.setInput('version', '1-2');
    fixture.detectChanges();
  }

  it('shows the turns with speaker names and hour-aware timestamps', async () => {
    await setup();
    expect(getTranscript).toHaveBeenCalledWith(7);
    const turns = el().querySelectorAll('.tv__turn');
    expect(turns.length).toBe(3);
    expect(turns[0].querySelector('.tv__speaker')!.textContent).toBe('Speaker 1');
    expect(turns[0].querySelector('.tv__time')!.textContent).toBe('0:00:00');
    expect(el().querySelector('.tv__legend')!.textContent).toContain('80% of speaking time');
  });

  it('seeks the episode from a timestamp', async () => {
    await setup();
    const spy = vi.fn();
    fixture.componentInstance.seek.subscribe(spy);
    el().querySelectorAll<HTMLButtonElement>('.tv__time')[1].click();
    expect(spy).toHaveBeenCalledWith(6);
  });

  it('filters and highlights by the search box', async () => {
    await setup();
    const box = el().querySelector<HTMLInputElement>('.tv__search')!;
    box.value = 'bienvenidos';
    box.dispatchEvent(new Event('input'));
    fixture.detectChanges();
    expect(el().querySelectorAll('.tv__turn').length).toBe(1);
    expect(el().querySelector('mark')!.textContent).toBe('Bienvenidos');
  });

  it('refetches when a newer transcript or diarization lands', async () => {
    await setup();
    fixture.componentRef.setInput('version', '1-3');
    fixture.detectChanges();
    expect(getTranscript).toHaveBeenCalledTimes(2);
  });

  it('says how to get speakers when there are none, and reports errors', async () => {
    await setup(of({ speakers: [], segments: [seg(0, 2, 'x')] }));
    expect(el().textContent).toContain('run Speakers');
    TestBed.resetTestingModule();
    await setup(throwError(() => new Error('409')));
    expect(el().textContent).toContain('Could not load the transcript');
  });
});
