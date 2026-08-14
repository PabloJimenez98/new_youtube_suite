/** Mirrors backend/app/schemas/upload.py's `UploadOut`. */
export interface UploadOut {
  id: number;
  filename: string;
  storage_path: string;
  duration_seconds: number | null;
  uploaded_at: string;
  status: string;
}
