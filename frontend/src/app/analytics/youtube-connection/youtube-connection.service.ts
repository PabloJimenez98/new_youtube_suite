import { HttpClient } from '@angular/common/http';
import { Service, inject } from '@angular/core';
import { Observable } from 'rxjs';

export type ConnectionStatus = 'not_configured' | 'disconnected' | 'connected' | 'needs_reconsent';

/** Mirrors backend/app/routers/youtube.py's `ConnectionOut` (never holds tokens). */
export interface ConnectionOut {
  status: ConnectionStatus;
  channel_id: string | null;
  channel_title: string | null;
  scopes: string[];
  connected_at: string | null;
  last_refresh_at: string | null;
  last_error: string | null;
}

/** The connected YouTube channel (M11, docs/decisions/0017 §1, §6). */
@Service()
export class YoutubeConnectionService {
  private readonly http = inject(HttpClient);

  getConnection(): Observable<ConnectionOut> {
    return this.http.get<ConnectionOut>('/youtube/connection');
  }

  /** The Google consent URL; the caller sends the browser there. */
  startConsent(): Observable<{ authorization_url: string }> {
    return this.http.post<{ authorization_url: string }>('/youtube/oauth/start', {});
  }

  disconnect(): Observable<void> {
    return this.http.delete<void>('/youtube/connection');
  }
}

/** `?youtube=error&reason=...` from the OAuth callback -> a readable message. */
export function callbackErrorMessage(reason: string | null): string {
  switch (reason) {
    case 'access_denied':
      return 'Consent was cancelled on Google’s screen.';
    case 'scopes':
      return 'Both permissions are needed (YouTube Analytics and YouTube read-only). Try again and leave both ticked.';
    case 'no_channel':
      return 'That Google account has no YouTube channel. Create one, then connect again.';
    case 'no_refresh_token':
      return 'Google did not return a refresh token. Try again.';
    case 'state':
      return 'The consent link expired or was not started here. Try again.';
    case 'not_configured':
      return 'YouTube OAuth is not configured on the server.';
    default:
      return `Connecting failed (${reason ?? 'unknown error'}).`;
  }
}
