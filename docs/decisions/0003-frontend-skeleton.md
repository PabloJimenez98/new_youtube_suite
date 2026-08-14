# Decision: M1 Frontend Skeleton — Architecture, Connectivity, Upload Contract

**Date:** 2026-08-14
**Status:** Decided (M1 planning sub-phase)

## Context

M1 builds the Angular skeleton — app shell, login screen, upload
screen, a bare uploads-list view — against M0's already-shipped
`/auth/login`, `POST /uploads`, `GET /uploads` endpoints.
`docs/project-description.md` says nothing Angular-specific beyond
"Frontend (Angular)" and a full-v1 feature list; nothing in the repo
decides component/state architecture, how the dev server reaches the
backend (M0's `main.py` has no CORS middleware), or the upload UI's
accepted file types/size limits — all genuinely open, and M1's own
planning bullet explicitly requires deciding the upload contract.

## Decision

**Angular style:** latest stable Angular CLI, standalone components,
routing + SCSS. No NgRx — a single `AuthService` holds the JWT as a
`signal<string | null>`, mirrored to `localStorage` so a page refresh
doesn't drop the session. Functional guards/interceptors
(`CanActivateFn`, `HttpInterceptorFn`), not class-based.

**Dev-server↔backend connectivity:** Angular CLI's dev-server proxy
(`proxy.conf.json` forwarding `/auth` and `/uploads` to
`http://127.0.0.1:8000`), not backend CORS middleware. The browser
only ever talks to `localhost:4200` in dev; zero backend changes.
Production serving topology (same-origin reverse proxy vs. genuinely
cross-origin) is still undecided — deferred, not solved here.

**Routes/components:** `/login` (public) and `/dashboard` (guarded).
The dashboard hosts both the upload form and the uploads list on one
page — the list refreshes in place after a successful upload. Built as
three independent components (`LoginComponent`, `UploadComponent`,
`UploadsListComponent`) composed inside a `DashboardComponent`, so a
future milestone (M2, once the list needs live job status) can promote
`/uploads` to its own route without a rewrite.

**Upload UI contract** (the milestone's actual planning requirement):
- Accepted types: `accept="video/*"` hint on the file input —
  client-side only; the backend enforces nothing yet. Known gap, left
  for a later hardening milestone.
- Max size: client-side guard at 2 GB, friendly inline error, rejected
  before any HTTP call is made.
- Progress: `HttpClient` with `reportProgress: true, observe: 'events'`,
  subscribing to `UploadProgress`/`Response` events for a live percent.
- States: `idle → selecting → uploading (with %) → success (list
  refreshes, form resets) → error (inline message, file retained for
  retry without reselecting)`.
- "Upload succeeded" = the POST resolves 201 **and** the subsequent
  `GET /uploads` re-fetch shows the new file in the list.

**Auth token storage:** JWT in the signal + `localStorage`; any 401
response forces logout + redirect to `/login` (no refresh-token flow,
matching M0's stateless JWT design).

## Alternatives considered

- **NgRx / class-based guards & interceptors** — rejected as overbuilt
  for one shared piece of auth state and three components.
- **CORS middleware on the backend** — rejected for M1: touches M0's
  shipped backend code for a dev-only concern and pre-commits to an
  allowed-origins policy before production topology is decided. The
  proxy solves the same problem with zero backend risk.
- **Separate `/upload` and `/uploads` routes** — rejected for M1: the
  list is deliberately non-live until M2, so a single page keeps the
  "log in, upload, see it listed" flow (M1's own testing bullet) to
  one screen. Revisit when M2 adds live status.
- **Server-side upload validation in this milestone** — rejected as
  out of scope: M1 was asked to define the *UI's* contract, not extend
  M0's frozen backend behavior. Flagged as a real gap for a future
  hardening milestone (M9).

## Consequences

- Hitting the backend directly at `localhost:8000` from a browser
  (bypassing `ng serve`) will fail with no CORS headers — this is
  expected, dev-server-mediated only, not a bug.
- The 2 GB / `video/*` guard is UI-only; a non-browser client can still
  upload anything of any size to `POST /uploads` — acceptable for M1,
  revisit alongside backend validation later.
- Token storage in `localStorage` (not an httpOnly cookie) means the
  app has no defense against XSS-based token theft — accepted for the
  same single-operator/stateless-JWT reasoning as M0; worth revisiting
  if the security posture changes.
