import { provideHttpClient } from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { ComponentFixture, TestBed } from '@angular/core/testing';
import { ActivatedRoute, Router, convertToParamMap } from '@angular/router';
import { of, throwError } from 'rxjs';

import { YoutubeConnection, browserNavigation } from './youtube-connection';
import { ConnectionOut, YoutubeConnectionService, callbackErrorMessage } from './youtube-connection.service';

const base: ConnectionOut = {
  status: 'connected',
  channel_id: 'UCtttttttttttttttttttttt',
  channel_title: 'Test channel',
  scopes: [],
  connected_at: '2026-11-07T10:00:00Z',
  last_refresh_at: '2026-11-07T11:00:00Z',
  last_error: null,
};

describe('YoutubeConnectionService', () => {
  it('calls the three /youtube endpoints', () => {
    TestBed.configureTestingModule({ providers: [provideHttpClient(), provideHttpClientTesting()] });
    const service = TestBed.inject(YoutubeConnectionService);
    const http = TestBed.inject(HttpTestingController);
    service.getConnection().subscribe();
    service.startConsent().subscribe();
    service.disconnect().subscribe();
    expect(http.expectOne({ url: '/youtube/connection', method: 'GET' })).toBeTruthy();
    expect(http.expectOne({ url: '/youtube/oauth/start', method: 'POST' })).toBeTruthy();
    expect(http.expectOne({ url: '/youtube/connection', method: 'DELETE' })).toBeTruthy();
    http.verify();
  });

  it('maps callback reasons to messages', () => {
    expect(callbackErrorMessage('scopes')).toContain('Both permissions');
    expect(callbackErrorMessage('weird')).toContain('weird');
  });
});

describe('YoutubeConnection', () => {
  let fixture: ComponentFixture<YoutubeConnection>;
  let stub: {
    getConnection: ReturnType<typeof vi.fn>;
    startConsent: ReturnType<typeof vi.fn>;
    disconnect: ReturnType<typeof vi.fn>;
  };
  let navigate: ReturnType<typeof vi.fn>;
  const el = () => fixture.nativeElement as HTMLElement;

  async function setup(connection: ConnectionOut, query: Record<string, string> = {}): Promise<void> {
    stub = {
      getConnection: vi.fn().mockReturnValue(of(connection)),
      startConsent: vi.fn(),
      disconnect: vi.fn(),
    };
    navigate = vi.fn().mockResolvedValue(true);
    await TestBed.configureTestingModule({
      imports: [YoutubeConnection],
      providers: [
        { provide: YoutubeConnectionService, useValue: stub },
        { provide: ActivatedRoute, useValue: { snapshot: { queryParamMap: convertToParamMap(query) } } },
        { provide: Router, useValue: { navigate } },
      ],
    }).compileComponents();
    fixture = TestBed.createComponent(YoutubeConnection);
    fixture.detectChanges();
  }

  it('not configured: points to the README, no connect button', async () => {
    await setup({ ...base, status: 'not_configured' });
    expect(el().textContent).toContain('GOOGLE_OAUTH_CLIENT_ID');
    expect(el().querySelector('.yt-connection__connect')).toBeNull();
  });

  it('disconnected: Connect starts the consent and redirects to Google', async () => {
    await setup({ ...base, status: 'disconnected', channel_id: null, channel_title: null });
    const assign = vi.spyOn(browserNavigation, 'assign').mockImplementation(() => undefined);
    stub.startConsent.mockReturnValue(of({ authorization_url: 'https://accounts.google.com/o/oauth2/v2/auth?x=1' }));
    el().querySelector<HTMLButtonElement>('.yt-connection__connect')!.click();
    expect(stub.startConsent).toHaveBeenCalled();
    expect(assign).toHaveBeenCalledWith('https://accounts.google.com/o/oauth2/v2/auth?x=1');
    assign.mockRestore();
  });

  it('needs re-consent: a visible warning with the reason and a Reconnect button', async () => {
    await setup({ ...base, status: 'needs_reconsent', last_error: 'invalid_grant: Token has been expired or revoked.' });
    const alert = el().querySelector('[role="alert"]')!;
    expect(alert.textContent).toContain('Reconnect needed');
    expect(alert.textContent).toContain('invalid_grant');
    expect(el().querySelector('.yt-connection__connect')!.textContent).toContain('Reconnect');
  });

  it('connected: shows the channel; Disconnect asks for confirmation first', async () => {
    await setup(base);
    expect(el().textContent).toContain('Test channel');
    expect(el().textContent).toContain('UCtttttttttttttttttttttt');
    stub.disconnect.mockReturnValue(of(undefined));
    el().querySelector<HTMLButtonElement>('.yt-connection__disconnect')!.click();
    fixture.detectChanges();
    expect(stub.disconnect).not.toHaveBeenCalled();
    stub.getConnection.mockReturnValue(of({ ...base, status: 'disconnected' }));
    el().querySelector<HTMLButtonElement>('.yt-connection__confirm-yes')!.click();
    fixture.detectChanges();
    expect(stub.disconnect).toHaveBeenCalled();
    expect(el().textContent).toContain('disconnected');
    expect(el().querySelector('.yt-connection__connect')).not.toBeNull();
  });

  it('shows the callback outcome and clears the query params', async () => {
    await setup(base, { youtube: 'connected' });
    expect(el().querySelector('[role="status"]')!.textContent).toContain('YouTube channel connected');
    expect(navigate).toHaveBeenCalledWith([], expect.objectContaining({ queryParams: {}, replaceUrl: true }));
  });

  it('shows a readable callback error', async () => {
    await setup({ ...base, status: 'disconnected' }, { youtube: 'error', reason: 'access_denied' });
    expect(el().querySelector('[role="status"]')!.textContent).toContain('cancelled');
  });

  it('a failed status load offers a retry', async () => {
    stub = { getConnection: vi.fn(), startConsent: vi.fn(), disconnect: vi.fn() };
    await setup(base);
    stub.getConnection.mockReturnValue(throwError(() => new Error('down')));
    (fixture.componentInstance as unknown as { refresh: () => void }).refresh();
    fixture.detectChanges();
    expect(el().textContent).toContain('Could not load');
  });
});
