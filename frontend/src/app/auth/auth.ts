import { HttpClient, HttpHeaders } from '@angular/common/http';
import { Service, computed, inject, signal } from '@angular/core';
import { Observable, tap } from 'rxjs';

export const TOKEN_STORAGE_KEY = 'ys_token';

interface TokenResponse {
  access_token: string;
  token_type: string;
}

@Service()
export class Auth {
  private readonly http = inject(HttpClient);

  private readonly tokenSignal = signal<string | null>(
    localStorage.getItem(TOKEN_STORAGE_KEY),
  );

  /** Current JWT, or null when logged out. */
  readonly token = this.tokenSignal.asReadonly();
  readonly isAuthenticated = computed(() => this.tokenSignal() !== null);

  /**
   * POST /auth/login expects `application/x-www-form-urlencoded` with
   * fields `username`/`password` (FastAPI's OAuth2PasswordRequestForm),
   * not JSON. `username` carries the user's email.
   */
  login(email: string, password: string): Observable<TokenResponse> {
    const body = new URLSearchParams();
    body.set('username', email);
    body.set('password', password);

    return this.http
      .post<TokenResponse>('/auth/login', body.toString(), {
        headers: new HttpHeaders({
          'Content-Type': 'application/x-www-form-urlencoded',
        }),
      })
      .pipe(tap((response) => this.setToken(response.access_token)));
  }

  logout(): void {
    this.setToken(null);
  }

  private setToken(token: string | null): void {
    this.tokenSignal.set(token);
    if (token) {
      localStorage.setItem(TOKEN_STORAGE_KEY, token);
    } else {
      localStorage.removeItem(TOKEN_STORAGE_KEY);
    }
  }
}
