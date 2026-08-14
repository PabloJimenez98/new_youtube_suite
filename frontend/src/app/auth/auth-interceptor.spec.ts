import { HttpInterceptorFn, HttpRequest } from '@angular/common/http';
import { provideHttpClient } from '@angular/common/http';
import { provideHttpClientTesting } from '@angular/common/http/testing';
import { TestBed } from '@angular/core/testing';
import { Router, provideRouter } from '@angular/router';
import { of, throwError } from 'rxjs';

import { Auth, TOKEN_STORAGE_KEY } from './auth';
import { authInterceptor } from './auth-interceptor';

describe('authInterceptor', () => {
  const runInterceptor: HttpInterceptorFn = (req, next) =>
    TestBed.runInInjectionContext(() => authInterceptor(req, next));

  beforeEach(() => {
    localStorage.removeItem(TOKEN_STORAGE_KEY);
    TestBed.configureTestingModule({
      providers: [provideHttpClient(), provideHttpClientTesting(), provideRouter([])],
    });
  });

  afterEach(() => {
    localStorage.removeItem(TOKEN_STORAGE_KEY);
  });

  it('attaches the Authorization header when a token exists', () => {
    localStorage.setItem(TOKEN_STORAGE_KEY, 'fake-token');
    const req = new HttpRequest('GET', '/uploads');
    let seenAuthHeader: string | null = null;

    runInterceptor(req, (r) => {
      seenAuthHeader = r.headers.get('Authorization');
      return of();
    }).subscribe();

    expect(seenAuthHeader).toBe('Bearer fake-token');
  });

  it('does not attach the header when no token exists', () => {
    const req = new HttpRequest('GET', '/uploads');
    let sawHeader = true;

    runInterceptor(req, (r) => {
      sawHeader = r.headers.has('Authorization');
      return of();
    }).subscribe();

    expect(sawHeader).toBe(false);
  });

  it('logs out and redirects to /login on a 401 from a non-login request', async () => {
    localStorage.setItem(TOKEN_STORAGE_KEY, 'stale-token');
    const req = new HttpRequest('GET', '/uploads');

    await TestBed.runInInjectionContext(async () => {
      const auth = TestBed.inject(Auth);
      const router = TestBed.inject(Router);
      const navigateSpy = vi.spyOn(router, 'navigateByUrl');

      await new Promise<void>((resolve) => {
        authInterceptor(req, () => throwError(() => ({ status: 401 }))).subscribe({
          error: () => {
            expect(auth.isAuthenticated()).toBe(false);
            expect(navigateSpy).toHaveBeenCalledWith('/login');
            resolve();
          },
        });
      });
    });
  });

  it('does not force logout on a 401 from the login request itself', async () => {
    const req = new HttpRequest('POST', '/auth/login', 'username=a&password=b');

    await TestBed.runInInjectionContext(async () => {
      const router = TestBed.inject(Router);
      const navigateSpy = vi.spyOn(router, 'navigateByUrl');

      await new Promise<void>((resolve) => {
        authInterceptor(req, () => throwError(() => ({ status: 401 }))).subscribe({
          error: () => {
            expect(navigateSpy).not.toHaveBeenCalled();
            resolve();
          },
        });
      });
    });
  });
});
