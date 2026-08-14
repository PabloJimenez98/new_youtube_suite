import { provideHttpClient } from '@angular/common/http';
import {
  HttpTestingController,
  provideHttpClientTesting,
} from '@angular/common/http/testing';
import { ComponentFixture, TestBed } from '@angular/core/testing';
import { Router, provideRouter } from '@angular/router';

import { Login } from './login';

describe('Login', () => {
  let component: Login;
  let fixture: ComponentFixture<Login>;
  let httpMock: HttpTestingController;
  let router: Router;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [Login],
      providers: [provideHttpClient(), provideHttpClientTesting(), provideRouter([])],
    }).compileComponents();

    fixture = TestBed.createComponent(Login);
    component = fixture.componentInstance;
    httpMock = TestBed.inject(HttpTestingController);
    router = TestBed.inject(Router);
    fixture.detectChanges();
  });

  afterEach(() => {
    httpMock.verify();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });

  it('disables submit when the form is invalid', () => {
    const button: HTMLButtonElement = fixture.nativeElement.querySelector(
      'button[type="submit"]',
    );
    expect(component['form'].invalid).toBe(true);
    expect(button.disabled).toBe(true);
  });

  it('maps the email field to `username` and navigates to /dashboard on success', () => {
    const navigateSpy = vi.spyOn(router, 'navigateByUrl');
    component['form'].setValue({ email: 'dev@example.com', password: 's3cret' });

    component['submit']();

    const req = httpMock.expectOne('/auth/login');
    expect(req.request.body).toBe('username=dev%40example.com&password=s3cret');
    req.flush({ access_token: 'abc', token_type: 'bearer' });

    expect(navigateSpy).toHaveBeenCalledWith('/dashboard');
  });

  it('shows the backend error message on 401 and does not navigate', () => {
    const navigateSpy = vi.spyOn(router, 'navigateByUrl');
    component['form'].setValue({ email: 'dev@example.com', password: 'wrong' });

    component['submit']();

    const req = httpMock.expectOne('/auth/login');
    req.flush(
      { detail: 'Incorrect email or password' },
      { status: 401, statusText: 'Unauthorized' },
    );

    expect(component['errorMessage']()).toBe('Incorrect email or password');
    expect(navigateSpy).not.toHaveBeenCalled();
  });
});
