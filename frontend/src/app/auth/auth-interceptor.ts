import { HttpInterceptorFn } from '@angular/common/http';
import { inject } from '@angular/core';
import { Router } from '@angular/router';
import { catchError, throwError } from 'rxjs';

import { Auth } from './auth';

/**
 * Attaches `Authorization: Bearer <token>` to outgoing requests when a
 * token is present, and forces logout + redirect to /login on any 401
 * response (M0 issues stateless JWTs only -- no refresh-token flow to
 * fall back to).
 */
export const authInterceptor: HttpInterceptorFn = (req, next) => {
  const auth = inject(Auth);
  const router = inject(Router);

  const token = auth.token();
  const authedReq = token
    ? req.clone({ setHeaders: { Authorization: `Bearer ${token}` } })
    : req;

  const isLoginRequest = req.url.endsWith('/auth/login');

  return next(authedReq).pipe(
    catchError((error) => {
      // Don't force a redirect for the login request's own 401 (wrong
      // password) -- LoginComponent shows that inline. Only a 401 on
      // an already-authenticated request means the session has gone bad.
      if (error?.status === 401 && !isLoginRequest) {
        auth.logout();
        router.navigateByUrl('/login');
      }
      return throwError(() => error);
    }),
  );
};
