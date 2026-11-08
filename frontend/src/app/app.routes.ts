import { Routes } from '@angular/router';

import { authGuard } from './auth/auth-guard';
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
      { path: 'dashboard', loadComponent: () => import('./dashboard/dashboard').then((m) => m.Dashboard) },
      // Lazy: the heavier pages load on first visit (docs/decisions/0021).
      { path: 'analytics', loadComponent: () => import('./analytics/analytics').then((m) => m.Analytics) },
      { path: 'manage', component: Manage },
      {
        path: 'manage/:id',
        loadComponent: () => import('./manage/episode-detail/episode-detail').then((m) => m.EpisodeDetail),
      },
      { path: 'how-it-works', loadComponent: () => import('./how-it-works/how-it-works').then((m) => m.HowItWorks) },
      { path: '', pathMatch: 'full', redirectTo: 'library' },
    ],
  },
  { path: '**', redirectTo: 'library' },
];
