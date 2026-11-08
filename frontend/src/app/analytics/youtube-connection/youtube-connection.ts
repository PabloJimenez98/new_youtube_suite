import { DatePipe } from '@angular/common';
import { Component, OnInit, inject, signal } from '@angular/core';
import { ActivatedRoute, Router } from '@angular/router';

import { ConnectionOut, YoutubeConnectionService, callbackErrorMessage } from './youtube-connection.service';

/** Navigation seam, so tests can assert the redirect without leaving the page. */
export const browserNavigation = {
  assign: (url: string) => window.location.assign(url),
};

/**
 * Connection card for the single platform-level YouTube account (M11,
 * docs/decisions/0017 §6). `needs_reconsent` is a visible warning with
 * the stored reason, so a dead token is never silent.
 */
@Component({
  selector: 'app-youtube-connection',
  imports: [DatePipe],
  templateUrl: './youtube-connection.html',
  styleUrl: './youtube-connection.scss',
})
export class YoutubeConnection implements OnInit {
  private readonly service = inject(YoutubeConnectionService);
  private readonly route = inject(ActivatedRoute, { optional: true });
  private readonly router = inject(Router, { optional: true });

  protected readonly connection = signal<ConnectionOut | null>(null);
  protected readonly loadError = signal(false);
  protected readonly busy = signal(false);
  protected readonly confirmingDisconnect = signal(false);
  protected readonly notice = signal<{ kind: 'success' | 'error'; text: string } | null>(null);

  ngOnInit(): void {
    this.readCallbackOutcome();
    this.refresh();
  }

  protected refresh(): void {
    this.loadError.set(false);
    this.service.getConnection().subscribe({
      next: (c) => this.connection.set(c),
      error: () => this.loadError.set(true),
    });
  }

  protected connect(): void {
    if (this.busy()) return;
    this.busy.set(true);
    this.service.startConsent().subscribe({
      next: ({ authorization_url }) => browserNavigation.assign(authorization_url),
      error: () => {
        this.busy.set(false);
        this.notice.set({ kind: 'error', text: 'Could not start the Google consent. Is the server configured?' });
      },
    });
  }

  protected disconnect(): void {
    this.confirmingDisconnect.set(false);
    this.busy.set(true);
    this.service.disconnect().subscribe({
      next: () => {
        this.busy.set(false);
        this.notice.set({ kind: 'success', text: 'YouTube channel disconnected.' });
        this.refresh();
      },
      error: () => {
        this.busy.set(false);
        this.notice.set({ kind: 'error', text: 'Disconnecting failed. Try again.' });
      },
    });
  }

  private readCallbackOutcome(): void {
    const params = this.route?.snapshot.queryParamMap;
    const outcome = params?.get('youtube');
    if (!outcome) return;
    this.notice.set(
      outcome === 'connected'
        ? { kind: 'success', text: 'YouTube channel connected.' }
        : { kind: 'error', text: callbackErrorMessage(params!.get('reason')) },
    );
    // Drop the query params so a reload doesn't repeat the notice.
    void this.router?.navigate([], { relativeTo: this.route, queryParams: {}, replaceUrl: true });
  }
}
