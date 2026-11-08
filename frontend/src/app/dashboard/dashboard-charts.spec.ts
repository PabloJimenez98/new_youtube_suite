import { JobOut } from '../jobs/job.model';
import { UploadOut } from '../uploads/upload.model';
import { episodesWithStats, jobsPerDay, lengthVsViews, viewsOverTime } from './dashboard-charts';

type Any = any; // eslint-disable-line @typescript-eslint/no-explicit-any -- loosely typed ECharts options

const upload = (id: number, published: string | null, views: number, minutes = 60): UploadOut => ({
  id,
  filename: `Ep ${id} [abcdefghij${id}].mkv`,
  storage_path: 'k',
  duration_seconds: minutes * 60,
  uploaded_at: '2026-11-01T00:00:00Z',
  status: 'uploaded',
  youtube: published
    ? { title: 't', published_at: published, view_count: views, like_count: 10, comment_count: 1, thumbnail_url: null, snapshot_date: '2026-11-08' }
    : null,
});

describe('dashboard chart builders', () => {
  const eps = episodesWithStats([upload(1, '2026-05-01T00:00:00Z', 100), upload(2, '2026-01-01T00:00:00Z', 300, 90), upload(3, null, 0)]);

  it('keeps linked episodes only, in publication order', () => {
    expect(eps.map((e) => e.id)).toEqual([2, 1]);
    expect(eps[0].name).toBe('Ep 2');
  });

  it('views over time: one bar per episode, with the id for click-through and a median line', () => {
    const o = viewsOverTime(eps).option as Any;
    expect(o['series'][0].data).toEqual([
      { value: 300, id: 2 },
      { value: 100, id: 1 },
    ]);
    expect(o['series'][0].markLine.data).toEqual([{ yAxis: 300 }]);
  });

  it('length vs views: minutes on x', () => {
    const o = lengthVsViews(eps).option as Any;
    expect(o['series'][0].data[0]).toEqual({ value: [90, 300], id: 2 });
  });

  it('jobs per day: done and failed stacked over 14 days', () => {
    const job = (status: JobOut['status'], day: string): JobOut => ({
      id: 1,
      upload_id: 1,
      tool_type: 'transcribe',
      status,
      progress_pct: null,
      started_at: null,
      completed_at: `${day}T10:00:00Z`,
      error_message: null,
      created_at: `${day}T09:00:00Z`,
    });
    const b = jobsPerDay([job('succeeded', '2026-11-08'), job('succeeded', '2026-11-08'), job('failed', '2026-11-07')], new Date('2026-11-08T12:00:00Z'));
    const o = b.option as Any;
    expect(o['xAxis'].data.length).toBe(14);
    expect(o['series'][0].data.at(-1)).toBe(2);
    expect(o['series'][1].data.at(-2)).toBe(1);
    expect(b.table.rows.at(-1)).toEqual(['2026-11-08', 2, 0]);
  });
});
