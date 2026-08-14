import { provideHttpClient } from '@angular/common/http';
import { provideHttpClientTesting } from '@angular/common/http/testing';
import { TestBed } from '@angular/core/testing';
import { CanActivateFn, Router, UrlTree, provideRouter } from '@angular/router';

import { TOKEN_STORAGE_KEY } from './auth';
import { authGuard } from './auth-guard';

describe('authGuard', () => {
  const executeGuard: CanActivateFn = (...guardParameters) =>
    TestBed.runInInjectionContext(() => authGuard(...guardParameters));

  beforeEach(() => {
    TestBed.configureTestingModule({
      providers: [provideHttpClient(), provideHttpClientTesting(), provideRouter([])],
    });
  });

  afterEach(() => {
    localStorage.removeItem(TOKEN_STORAGE_KEY);
  });

  it('allows activation when a token is present', () => {
    localStorage.setItem(TOKEN_STORAGE_KEY, 'fake-token');

    const result = executeGuard({} as never, { url: '/dashboard' } as never);

    expect(result).toBe(true);
  });

  it('redirects to /login when no token is present', () => {
    localStorage.removeItem(TOKEN_STORAGE_KEY);

    const result = executeGuard({} as never, { url: '/dashboard' } as never);
    const router = TestBed.inject(Router);

    expect(result instanceof UrlTree).toBe(true);
    expect(router.serializeUrl(result as UrlTree)).toBe('/login');
  });
});
