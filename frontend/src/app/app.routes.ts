import { Routes } from '@angular/router';

import { Analytics } from './analytics/analytics';
import { authGuard } from './auth/auth-guard';
import { Dashboard } from './dashboard/dashboard';
import { Library } from './library/library';
import { Login } from './login/login';
import { Manage } from './manage/manage';
import { Shell } from './shell/shell';

export const routes: Routes = [
  { path: 'login', component: Login },
  {
    path: '',
    component: Shell,
    canActivate: [authGuard],
    children: [
      { path: 'library', component: Library },
      { path: 'dashboard', component: Dashboard },
      { path: 'analytics', component: Analytics },
      { path: 'manage', component: Manage },
      { path: '', pathMatch: 'full', redirectTo: 'library' },
    ],
  },
  { path: '**', redirectTo: 'library' },
];
