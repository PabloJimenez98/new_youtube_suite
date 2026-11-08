/** Mirrors backend/app/schemas/upload.py's `YoutubeStatsOut` (docs/decisions/0022 §1). */
export interface YoutubeStats {
  title: string;
  published_at: string;
  view_count: number | null;
  like_count: number | null;
  comment_count: number | null;
  thumbnail_url: string | null;
  snapshot_date: string;
}

/** Mirrors backend/app/schemas/upload.py's `UploadOut`. */
export interface UploadOut {
  id: number;
  filename: string;
  storage_path: string;
  duration_seconds: number | null;
  uploaded_at: string;
  status: string;
  /** The linked YouTube video (from the filename or set by hand); optional for older fixtures. */
  youtube_video_id?: string | null;
  /** Its latest public numbers from the market pipeline; null until tracked. */
  youtube?: YoutubeStats | null;
}
