import { provideHttpClient } from '@angular/common/http';
import { provideHttpClientTesting } from '@angular/common/http/testing';
import { ComponentFixture, TestBed } from '@angular/core/testing';
import { Router, provideRouter } from '@angular/router';

import { Auth, TOKEN_STORAGE_KEY } from '../auth/auth';
import { Shell } from './shell';

describe('Shell', () => {
  let fixture: ComponentFixture<Shell>;
  let router: Router;

  beforeEach(async () => {
    localStorage.setItem(TOKEN_STORAGE_KEY, 'fake-token');

    await TestBed.configureTestingModule({
      imports: [Shell],
      providers: [provideHttpClient(), provideHttpClientTesting(), provideRouter([])],
    }).compileComponents();

    fixture = TestBed.createComponent(Shell);
    router = TestBed.inject(Router);
  });

  afterEach(() => {
    localStorage.removeItem(TOKEN_STORAGE_KEY);
  });

  it('should create', () => {
    expect(fixture.componentInstance).toBeTruthy();
  });

  it('renders the wordmark and all four nav links', () => {
    fixture.detectChanges();
    const compiled = fixture.nativeElement as HTMLElement;
    expect(compiled.querySelector('.shell-nav__wordmark')?.textContent).toContain(
      'YouTube Suite',
    );
    const links = Array.from(compiled.querySelectorAll('.shell-nav__links a')).map(
      (a) => a.getAttribute('href'),
    );
    expect(links).toEqual(['/library', '/dashboard', '/analytics', '/manage']);
  });

  it('logs out and navigates to /login on click', () => {
    fixture.detectChanges();
    const auth = TestBed.inject(Auth);
    const navigateSpy = vi.spyOn(router, 'navigateByUrl');

    const button: HTMLButtonElement = fixture.nativeElement.querySelector('button');
    button.click();

    expect(auth.isAuthenticated()).toBe(false);
    expect(navigateSpy).toHaveBeenCalledWith('/login');
  });
});
