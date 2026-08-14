import { provideHttpClient } from '@angular/common/http';
import {
  HttpTestingController,
  provideHttpClientTesting,
} from '@angular/common/http/testing';
import { TestBed } from '@angular/core/testing';

import { Auth, TOKEN_STORAGE_KEY } from './auth';

describe('Auth', () => {
  let service: Auth;
  let httpMock: HttpTestingController;

  beforeEach(() => {
    localStorage.removeItem(TOKEN_STORAGE_KEY);
    TestBed.configureTestingModule({
      providers: [provideHttpClient(), provideHttpClientTesting()],
    });
    service = TestBed.inject(Auth);
    httpMock = TestBed.inject(HttpTestingController);
  });

  afterEach(() => {
    httpMock.verify();
  });

  it('starts unauthenticated when localStorage has no token', () => {
    expect(service.isAuthenticated()).toBe(false);
    expect(service.token()).toBeNull();
  });

  it('POSTs form-urlencoded username/password and stores the token on success', () => {
    service.login('dev@example.com', 's3cret').subscribe();

    const req = httpMock.expectOne('/auth/login');
    expect(req.request.method).toBe('POST');
    expect(req.request.headers.get('Content-Type')).toBe(
      'application/x-www-form-urlencoded',
    );
    expect(req.request.body).toBe('username=dev%40example.com&password=s3cret');

    req.flush({ access_token: 'abc123', token_type: 'bearer' });

    expect(service.isAuthenticated()).toBe(true);
    expect(service.token()).toBe('abc123');
    expect(localStorage.getItem(TOKEN_STORAGE_KEY)).toBe('abc123');
  });

  it('does not store a token when login fails', () => {
    service.login('dev@example.com', 'wrong').subscribe({ error: () => {} });

    const req = httpMock.expectOne('/auth/login');
    req.flush(
      { detail: 'Incorrect email or password' },
      { status: 401, statusText: 'Unauthorized' },
    );

    expect(service.isAuthenticated()).toBe(false);
    expect(localStorage.getItem(TOKEN_STORAGE_KEY)).toBeNull();
  });

  it('logout clears the signal and localStorage', () => {
    service.login('dev@example.com', 's3cret').subscribe();
    httpMock.expectOne('/auth/login').flush({ access_token: 'abc123', token_type: 'bearer' });
    expect(service.isAuthenticated()).toBe(true);

    service.logout();

    expect(service.isAuthenticated()).toBe(false);
    expect(service.token()).toBeNull();
    expect(localStorage.getItem(TOKEN_STORAGE_KEY)).toBeNull();
  });
});
